from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torch
import os
import tempfile
import logging
from pydub import AudioSegment
from pypinyin import lazy_pinyin, Style
from opencc import OpenCC
import requests
from fastapi.middleware.cors import CORSMiddleware
import json

# -------------------------------
# 1. 設置日誌
# -------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Whisper ASR 與 LLM 整合服務", version="2.0")

# 啟用 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 2. 模型配置
# -------------------------------
MODEL_PATH = r"D:\whisper_training5\checkpoints\checkpoint-1000"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# 3. 加載 Whisper 模型
# -------------------------------
processor = None
model = None

try:
    logger.info(f"🔍 加載模型: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ 模型路徑不存在: {MODEL_PATH}")

    processor = WhisperProcessor.from_pretrained(MODEL_PATH)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()
    logger.info(f"✅ 模型加載成功，運行於 {DEVICE}")

except Exception as e:
    logger.critical(f"🛑 模型加載失敗: {e}")
    raise

# -------------------------------
# 4. 轉錄與分析端點
# -------------------------------
LLM_ANALYZE_URL = "http://localhost:8080/diagnose-pronunciation"

@app.post("/transcribe", response_class=JSONResponse)
async def transcribe(
    file: UploadFile = File(...),
    target_word: str = Form(None),
    difficulty_level: str = Form("小學")
):
    temp_input_path = None
    temp_wav_path = None
    response = None 

    try:
        # 1. 讀取內容
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="音檔為空")

        # 2. 保存原始文件
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            temp_input_path = tmp.name

        # 3. 模型檢查
        if processor is None or model is None:
            raise HTTPException(status_code=500, detail="ASR 模型未加載")

        # 4. 音頻轉換
        audio = AudioSegment.from_file(temp_input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        temp_wav_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        audio.export(temp_wav_path, format="wav")
 
        # 5. 加載音頻並提取特徵
        waveform, sr = torchaudio.load(temp_wav_path)
        input_features = processor(
            waveform.squeeze(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(DEVICE)

        # 6. 轉錄
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                language="zh",
                task="transcribe",
                max_length=448,
                no_repeat_ngram_size=3,
                temperature=0.0
            )
        transcription_simplified = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        logger.info(f"📝【除錯】Whisper 轉錄結果 (簡體): '{transcription_simplified}'")

        # 7. 繁簡轉換
        cc = OpenCC('s2t')
        transcription_traditional = cc.convert(transcription_simplified)
        logger.info(f"📝【除錯】轉換為繁體: '{transcription_traditional}'")
                                                          
        # 8. 生成拼音
        pinyin_list = lazy_pinyin(transcription_simplified, style=Style.TONE3)
        asr_ipa = " ".join(pinyin_list).strip()

        if not asr_ipa or asr_ipa == "unknown":
            asr_ipa = "unknown"
            logger.warning("⚠️ 拼音為空，使用 'unknown' 作為備用")
        else:
            logger.info(f"🔤 生成拼音: '{asr_ipa}'")

        
        user_asr_input = asr_ipa

        # 9. 預設值
        target_ipa = "unknown"
        suggest_word = "備用字"

        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        session.mount("http://", HTTPAdapter(max_retries=retries))

        # 10. 呼叫 LLM 分析
        if target_word:
            logger.info(f"🔍 發現目標詞 '{target_word}'，呼叫 LLM 分析...")
            try:
                response = requests.post(
                    LLM_ANALYZE_URL,
                    data=[
                        "target_word", target_word,
                        "user_asr_input", asr_ipa,
                        "difficulty_level", difficulty_level
                    ],
                    timeout=15,
                    headers={"ngrok-skip-browser-warning": "true"}
                )

                logger.info(f"📡 LLM 回應狀態碼: {json.dumps(data, ensure_ascii=False, indent=2)}")
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"📥 LLM 原始回應: {json.dumps(data, ensure_ascii=False, indent=2)}")
                    analysis = data.get("analysis", {})
                    raw_suggest = analysis.get("suggest_word", "").strip()
                    if raw_suggest and "錯誤" not in raw_suggest and "找不到" not in raw_suggest:
                            suggest_word = raw_suggest
                    target_ipa = analysis.get("target_ipa", "unknown").strip()
                else:
                    logger.error(f"❌ LLM 返回錯誤狀態: {response.status_code}, body: {response.text[:200]}")
            except requests.exceptions.ConnectionError:
                logger.error("❌ 無法連接到 LLM 伺服器")
            except requests.exceptions.Timeout:
                logger.error("⏰ LLM 請求超時")
            except Exception as e:
                logger.error(f"❌ 呼叫 LLM 時發生未預期錯誤: {e}", exc_info=True)

        # 11. 回傳結果
        result = {
            "target_word": target_word,
            "asr_ipa": asr_ipa,
            "target_ipa": f"/{target_ipa}/" if target_ipa != "unknown" else "unknown",
            "suggest_word": suggest_word
        }

        logger.info(f"""
🎯 最終結果 (供前端使用):
{'='*50}
目標詞:       {target_word}
轉錄拼音:    {asr_ipa}
正確拼音:    {target_ipa}
建議詞:      {suggest_word}
{'='*50}
""")
        
        return result

    except Exception as e:
        logger.error(f"❌ 轉錄失敗: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="音頻處理失敗")

    finally:
        # 刪除臨時文件
        for path in [temp_input_path, temp_wav_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                    logger.info(f"🗑️ 已刪除臨時文件: {path}")
                except Exception as e:
                    logger.warning(f"⚠️ 無法刪除文件: {path}, 錯誤: {e}")

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "asr_model_loaded": processor is not None and model is not None,
        "device": DEVICE,
        "llm_analyze_url": LLM_ANALYZE_URL,
        "version": "2.0"
    }

# -------------------------------
# 6. 啟動伺服器
# -------------------------------
if __name__ == "__main__":
    logger.info("🚀 啟動 ASR 伺服器...")
    import uvicorn
    uvicorn.run("asr3:app", host="0.0.0.0", port=8000, reload=False)