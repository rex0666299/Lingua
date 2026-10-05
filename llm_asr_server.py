from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
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
from fastapi.middleware.cors import CORSMiddleware
import json
import re
import random
from typing import Dict, List, Tuple
from llama_cpp import Llama
from contextlib import redirect_stderr
import io
import noisereduce as nr 


#設置日誌

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Whisper ASR 與 LLM 整合服務", version="2.2")

# 啟用 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 模型配置

ASR_MODEL_PATH = r"model\checkpoint-11000"
LLM_MODEL_PATH = r"model\Tiger-Gemma-9B-v3-Q3_K_M.gguf"
PINYIN_DICT_PATH = r"model\pinyin_dict_ch_v3.json"
KNOWLEDGE_BASE_PATH = r"model\knowledge_base_ch_v3.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# 加載 Whisper 模型

processor = None
asr_model = None

try:
    logger.info(f"🔍 加載 ASR 模型: {ASR_MODEL_PATH}")
    if not os.path.exists(ASR_MODEL_PATH):
        raise FileNotFoundError(f"❌ 模型路徑不存在: {ASR_MODEL_PATH}")

    processor = WhisperProcessor.from_pretrained(ASR_MODEL_PATH)
    asr_model = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL_PATH).to(DEVICE)
    asr_model.eval()
    logger.info(f"✅ ASR 模型加載成功，運行於 {DEVICE}")

except Exception as e:
    logger.critical(f"🛑 ASR 模型加載失敗: {e}")
    raise


# LLM 與 發音診斷模組

try:
    s2t = OpenCC('s2t')
    t2s = OpenCC('t2s')
    logger.info("✅ OpenCC 繁簡轉換器已創建")
except Exception as e:
    logger.warning(f"⚠️ OpenCC 初始化失敗: {e}")
    s2t = lambda x: x
    t2s = lambda x: x

# 初始化 LLM 
llm = None
pinyin_dict = None
rag_knowledge_base = None

try:
    logger.info("⚙️ 加載 LLM 模型...")
    if not os.path.exists(LLM_MODEL_PATH):
        raise FileNotFoundError(f"LLM 模型不存在: {LLM_MODEL_PATH}")

    llm = Llama(
        model_path=LLM_MODEL_PATH,
        n_gpu_layers=35,
        main_gpu=0,
        n_ctx=4096,
        verbose=False,
        chat_format='chatml'
    )
    logger.info("✅ LLM 模型加載成功")

    with open(PINYIN_DICT_PATH, 'r', encoding='utf-8') as f:
        pinyin_dict = json.load(f)
    logger.info("✅ 拼音詞典加載成功")

    with open(KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as f:
        rag_knowledge_base = json.load(f)
    logger.info("✅ 知識庫加載成功")

except Exception as e:
    logger.critical(f"❌ 資源加載失敗: {e}")
    raise

# 核心功能定義 
PINYIN_INITIALS = {
    'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x',
    'zh', 'ch', 'sh', 'r', 'z', 'c', 's'
}
PINYIN_FINALS = {
    'a', 'o', 'e', 'i', 'u', 'v', 'ai', 'ei', 'ui', 'ao', 'ou', 'iu', 'ie',
    've', 'er', 'an', 'en', 'in', 'un', 'vn', 'ang', 'eng', 'ing', 'ong'
}
WHOLE_SYLLABLES = {
    'a', 'o', 'e', 'er', 'ai', 'ei', 'ao', 'ou', 'an', 'en', 'ang', 'eng',
    'yi', 'ya', 'yo', 'ye', 'yao', 'you', 'yan', 'yin', 'yang', 'ying', 'yong',
    'wu', 'wa', 'wo', 'wai', 'wei', 'wan', 'wen', 'wang', 'weng',
    'yu', 'yue', 'yuan', 'yun'
}
ALL_PINYIN_PARTS = PINYIN_INITIALS.union(PINYIN_FINALS).union(WHOLE_SYLLABLES)


def parse_asr_output(asr_string: str) -> str:
    parts = [re.sub(r'\d+$', '', part) for part in asr_string.split()]
    syllables = []
    i = 0
    while i < len(parts):
        if parts[i] in PINYIN_INITIALS and (i + 1) < len(parts) and parts[i+1] in PINYIN_FINALS:
            syllables.append(parts[i] + parts[i+1])
            i += 2
        else:
            syllables.append(parts[i])
            i += 1
    return " ".join(syllables)


def segment_pinyin_syllable(syllable: str) -> list:
    if syllable in ALL_PINYIN_PARTS: return [syllable]
    initial, final = "", syllable
    if syllable.startswith(('zh', 'ch', 'sh')):
        initial, final = syllable[:2], syllable[2:]
    elif syllable.startswith(tuple(PINYIN_INITIALS)):
        initial, final = syllable[0], syllable[1:]
    return [initial, final] if initial and final else [final]


def align_and_find_errors(target_syllables: list, actual_syllables: list) -> tuple:
    substitution_cost, indel_cost = 1, 1
    n, m = len(target_syllables), len(actual_syllables)
    dp = [[(0, '') for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1): dp[i][0] = (i * indel_cost, 'up')
    for j in range(1, m + 1): dp[0][j] = (j * indel_cost, 'left')
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if target_syllables[i - 1] == actual_syllables[j - 1] else substitution_cost
            scores = [(dp[i - 1][j - 1][0] + cost, 'diag'), (dp[i][j - 1][0] + indel_cost, 'left'), (dp[i - 1][j][0] + indel_cost, 'up')]
            dp[i][j] = min(scores, key=lambda x: x[0])
    errors, i, j = [], n, m
    while i > 0 or j > 0:
        direction = dp[i][j][1]
        if direction == 'diag':
            if target_syllables[i - 1] != actual_syllables[j - 1]:
                errors.append({'type': 'Substitution', 'target': target_syllables[i - 1], 'actual': actual_syllables[j - 1]})
            i -= 1; j -= 1
        elif direction == 'up':
            errors.append({'type': 'Deletion', 'target': target_syllables[i - 1], 'actual': None}); i -= 1
        else:
            errors.append({'type': 'Insertion', 'target': None, 'actual': actual_syllables[j - 1]}); j -= 1
    errors.reverse()
    error_count = dp[n][m][0]
    return errors, error_count


def diagnose_pronunciation(target_pinyin_str: str, user_asr_input: str) -> tuple:
    user_pinyin_str = parse_asr_output(user_asr_input)
    user_syllables = user_pinyin_str.split()
    target_pinyins = [p.strip() for p in target_pinyin_str.split(',')]
    best_target_pinyin = ""
    min_error_count = float('inf')
    if len(target_pinyins) == 1:
        best_target_pinyin = target_pinyins[0]
    else:
        for p_option in target_pinyins:
            _, error_count = align_and_find_errors(p_option.split(), user_syllables)
            if error_count < min_error_count:
                min_error_count = error_count
                best_target_pinyin = p_option
    if not best_target_pinyin and target_pinyins:
        best_target_pinyin = target_pinyins[0]
    target_syllables = best_target_pinyin.split()
    all_errors, final_error_count = align_and_find_errors(target_syllables, user_syllables)
    total_error_rate = final_error_count / len(target_syllables) if target_syllables else 0
    error_summary = []
    for error in all_errors:
        if error['type'] == 'Substitution':
            target_parts, actual_parts = segment_pinyin_syllable(error['target']), segment_pinyin_syllable(error['actual'])
            if len(target_parts) > 1 and len(actual_parts) > 1 and target_parts[0] != actual_parts[0]: error_summary.append(target_parts[0])
            if target_parts[-1] != actual_parts[-1]: error_summary.append(target_parts[-1])
        elif error['type'] == 'Deletion': error_summary.extend(segment_pinyin_syllable(error['target']))
        elif error['type'] == 'Insertion': error_summary.append('__INSERTION__')
    unique_errors = sorted(list(dict.fromkeys(error_summary)))
    return unique_errors, total_error_rate, user_pinyin_str, best_target_pinyin


def clean_llm_output(generated_text: str) -> list:
    cleaned = re.sub(r'[^\u4e00-\u9fa5,、]', '', generated_text)
    cleaned = cleaned.replace('、', ',')
    words = [word.strip() for word in cleaned.split(',') if word.strip()]
    return list(dict.fromkeys(words))


# LLM 交互與詞彙生成 
DIFFICULTY_FEATURES_CH = {
    "幼兒園": "詞語特徵：通常是1-2個字。必須是日常生活中極其常見的、具體的名詞（如動物、食物、身體部位）或簡單的動詞（如跑、跳、吃）。絕對不能是任何抽象概念、書面語、成語、人名或地名。",
    "小學": "詞語特徵：通常是2-4個字。必須是常用詞，可以描述簡單的情感、品質或校園活動。可以包含家喻戶曉的簡單成語。不能是專業術語或生僻詞。",
    "中學": "詞語特徵：通常是2個字以上。可以是書面語、課本中出現的術語（如科學、歷史類）、或有一定典故的成語。詞語允許有一定深度。",
    "成人": "詞語特徵：長度不限。可以是專業領域的術語、低頻詞、複雜的成語或具有文學色彩的書面語。詞語允許複雜和抽象。"
}
DIFFICULTY_TEMPERATURES = {"幼兒園": 0.0, "小學": 0.1, "中學": 0.2, "成人": 0.3}

NEGATIVE_KEYWORDS = [
    '菌', '症', '战役', '大学', '学院', '区', '县', '市', '省', '国', '洲', '洋',
    '条约', '协定', '法令', '研究所', '委员会', '有限公司', '集团', '主义',
    '沙门氏菌', '共和国', '自治区', '经济区', '高新区', '开发区', '风景区',
    '戰役', '大學', '學院', '區', '縣', '市', '省', '國', '洲', '洋',
    '條約', '協定', '法令', '研究所', '委員會', '有限公司', '集團', '主義',
    '共和國', '自治區', '經濟區', '高新區', '開發區', '風景區'
]


def get_candidate_pool(difficulty_level: str, all_words_pool: list, count: int = 60) -> list:
    filtered_pool = [w for w in all_words_pool if not any(neg_word in w for neg_word in NEGATIVE_KEYWORDS)]
    final_pool = []
    if difficulty_level == "幼兒園":
        one_char_words = [w for w in filtered_pool if len(w) == 1]
        two_char_words = [w for w in filtered_pool if len(w) == 2]
        final_pool.extend(one_char_words)
        final_pool.extend(two_char_words)
    elif difficulty_level == "小學":
        final_pool = [w for w in filtered_pool if 2 <= len(w) <= 4]
    else:
        final_pool = filtered_pool
    random.shuffle(final_pool)
    return final_pool[:count]


def find_comprehensive_words(error_keys: list) -> list:
    if not error_keys: return []
    base_words = set(rag_knowledge_base.get(error_keys[0], []))
    if not base_words: return []
    for key in error_keys[1:]:
        words_with_key = set(rag_knowledge_base.get(key, []))
        base_words.intersection_update(words_with_key)
    return list(base_words)


def llm_select_one_word(error_keys: list, difficulty_level: str) -> str:
    if len(error_keys) > 1:
        sound_req = f"必須同時包含 /{'/、/'.join(error_keys)}/"
        all_words = find_comprehensive_words(error_keys)
    else:
        sound_req = f"必須包含 /{error_keys[0]}/"
        all_words = rag_knowledge_base.get(error_keys[0], [])

    valid_words = [w for w in all_words if w in pinyin_dict]
    if not valid_words:
        return "知識庫無匹配詞"

    candidate_pool = get_candidate_pool(difficulty_level, valid_words, 50)
    if not candidate_pool:
        return "候選池為空"

    prompt = f"""<start_of_turn>user
Select ONE word from this list that best matches the sound '{sound_req}' and is suitable for {difficulty_level} level.

Candidate Pool:
{', '.join(candidate_pool)}

Rules:
- Output only the Chinese word.
- No punctuation, no explanation.

Now respond:
<end_of_turn>
<start_of_turn>model
"""

    try:
        with io.StringIO() as f, redirect_stderr(f):
            resp = llm(prompt, max_tokens=8, temperature=0.1, stop=["\n", "<"])
        text = resp['choices'][0]['text'].strip()
        logger.info(f"🟢 LLM 輸出: {text}")

        match = re.search(r'[\u4e00-\u9fa5]+', text)
        word = match.group(0) if match else None

        if word and word in candidate_pool:
            return word
        elif candidate_pool:
            return candidate_pool[0]
        else:
            return "無建議詞"
    except Exception as e:
        logger.error(f"LLM 錯誤: {e}")
        return "LLM 錯誤"


def process_pronunciation_task(target_word: str, user_asr_input: str, difficulty_level: str) -> dict:
    if pinyin_dict is None:
        return {"status": "error", "message": "拼音詞典未加載，請檢查檔案。"}
    if rag_knowledge_base is None:
        return {"status": "error", "message": "知識庫未加載，請檢查檔案。"}

    simplified_target_word = t2s.convert(target_word)
    target_pinyin = pinyin_dict.get(simplified_target_word)
    if not target_pinyin:
        return {"status": "error", "message": f"在拼音字典中找不到 '{target_word}' 的拼音。"}

    detected_errors, error_rate, parsed_pinyin, matched_target = diagnose_pronunciation(target_pinyin, user_asr_input)

    result = {
        "status": "success",
        "target_word": target_word,
        "target_pinyin": matched_target,
        "user_asr_input": user_asr_input,
        "parsed_pinyin": parsed_pinyin,
        "error_rate": error_rate,
        "detected_errors": [s2t.convert(e) for e in detected_errors],
        "decision": "",
        "practice_word": "",
        "practice_type": ""
    }

    target_len = len(matched_target.split())
    absolute_error_count = round(error_rate * target_len)
    is_too_high = (error_rate > 0.75 and absolute_error_count > 2) or (error_rate >= 1.0 and target_len > 1)

    if is_too_high:
        result["decision"] = f"錯誤率過高 ({error_rate:.2%})，終止生成。"
        return result
    elif not detected_errors:
        result["decision"] = "未發現錯誤，發音準確！"
        return result

    trainable_errors = [e for e in detected_errors if e != '__INSERTION__']
    if not trainable_errors and '__INSERTION__' in detected_errors:
        result["decision"] = "僅檢測到插入錯誤，不生成單詞。"
        return result
    elif not trainable_errors:
        result["decision"] = "未檢測到可訓練的發音錯誤。"
        return result

    if len(trainable_errors) > 1:
        comprehensive_pool = find_comprehensive_words(trainable_errors)
        if comprehensive_pool:
            result["decision"] = f"檢測到多個錯誤，生成綜合練習。"
            result["practice_type"] = "comprehensive"
            generated_word = llm_select_one_word(trainable_errors, difficulty_level)
        else:
            result["decision"] = f"檢測到多個錯誤，但找不到綜合練習詞，降級為單一練習。"
            result["practice_type"] = "focused"
            generated_word = llm_select_one_word([trainable_errors[0]], difficulty_level)
    else:
        result["decision"] = f"檢測到單一錯誤，生成聚焦練習。"
        result["practice_type"] = "focused"
        generated_word = llm_select_one_word(trainable_errors, difficulty_level)

    result["practice_word"] = s2t.convert(generated_word)
    logger.info(f"🎯 最終 practice_word: {result['practice_word']}")
    return result



# FastAPI 端點


@app.post("/transcribe", response_class=JSONResponse)
async def transcribe(
    file: UploadFile = File(...),
    target_word: str = Form(None),
    difficulty_level: str = Form("小學"),
    phoneme: str = Form(None), 
    enable_noise_reduction: bool = Form(False)  # 是否啟用降噪
):
    temp_input_path = None
    temp_wav_path = None

    try:

        if phoneme and difficulty_level:
            logger.info(f"🎯 收到 phoneme='{phoneme}' 和 difficulty='{difficulty_level}'，跳過音檔，直接生成詞...")
            try:
                generated_word = llm_select_one_word([phoneme], difficulty_level)
                suggest_word = s2t.convert(generated_word) if generated_word else " "

                result = {
                    "target_word": target_word or "N/A",
                    "asr_ipa": "skipped",
                    "target_ipa": f"/{phoneme}/",
                    "suggest_word": suggest_word,
                    "decision": f"直接生成包含 /{phoneme}/ 的 {difficulty_level} 級詞"
                }
                logger.info(f"🟢 直接生成結果: {result}")
                return result
            except Exception as e:
                logger.error(f"❌ LLM 直接生成失敗，繼續處理音檔: {e}", exc_info=True)

      
        # 原有音檔處理流程

        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="音檔為空")

        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            temp_input_path = tmp.name

        if processor is None or asr_model is None:
            raise HTTPException(status_code=500, detail="ASR 模型未加載")

        audio = AudioSegment.from_file(temp_input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        temp_wav_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        audio.export(temp_wav_path, format="wav")

        waveform, sr = torchaudio.load(temp_wav_path)

        # 多通道合併為單通道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 降噪處理
        if enable_noise_reduction:
            logger.info("🔊 執行降噪處理...")
            audio_np = waveform.squeeze().numpy()
            reduced_noise = nr.reduce_noise(
                y=audio_np,
                sr=sr,
                thresh_n_mult_nonstationary=2.0,
                stationary=False,
                n_fft=512,
                win_length=512,
                hop_length=256
            )
            reduced_waveform = torch.from_numpy(reduced_noise).float().unsqueeze(0)
        else:
            reduced_waveform = waveform

        # 提取 Whisper 特徵
        input_features = processor(
            reduced_waveform.squeeze(),
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(DEVICE)

        # 生成轉錄
        with torch.no_grad():
            predicted_ids = asr_model.generate(
                input_features,
                language="zh",
                task="transcribe",
                max_length=448,
                no_repeat_ngram_size=3,
                temperature=0.0
            )
        transcription_simplified = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        logger.info(f"📝【除錯】Whisper 轉錄結果 (簡體): '{transcription_simplified}'")

        cc = OpenCC('s2t')
        transcription_traditional = cc.convert(transcription_simplified)
        logger.info(f"📝【除錯】轉換為繁體: '{transcription_traditional}'")

        # 轉為拼音
        pinyin_list = lazy_pinyin(transcription_simplified, style=Style.TONE3)
        syllable_count = len(pinyin_list)
        asr_ipa = " ".join(pinyin_list).strip()

        # 新增：若音節超過 10，視為無效
        if syllable_count > 10:
            asr_ipa = "unknown"
            logger.warning(f"⚠️ 轉錄音節過長 ({syllable_count} > 10)，視為無效輸入")
        elif not asr_ipa:
            asr_ipa = "unknown"
            logger.warning("⚠️ 拼音為空，使用 'unknown' 作為備用")
        else:
            logger.info(f"🔤 生成拼音: '{asr_ipa}' (共 {syllable_count} 音節)")

        user_asr_input = asr_ipa

        target_ipa = "unknown"
        suggest_word = " "

        if target_word:
            logger.info(f"🔍 發現目標詞 '{target_word}'，進行發音診斷...")
            try:
                analysis_result = process_pronunciation_task(target_word, user_asr_input, difficulty_level)
                raw_suggest = analysis_result.get("practice_word", "").strip()
                if raw_suggest and "錯誤" not in raw_suggest and "找不到" not in raw_suggest:
                    suggest_word = raw_suggest
                target_ipa = analysis_result.get("target_pinyin", "unknown").strip()
            except Exception as e:
                logger.error(f"❌ LLM 分析失敗: {e}", exc_info=True)

        result = {
            "target_word": target_word,
            "asr_ipa": asr_ipa,
            "target_ipa": f"/{target_ipa}/" if target_ipa != "unknown" else "unknown",
            "suggest_word": suggest_word
        }

        logger.info(f"""
            最終結果:{'='*50}
            目標詞:       {target_word}
            轉錄拼音:    {asr_ipa}
            正確拼音:    {target_ipa}
            建議詞:      {suggest_word}
                        {'='*50}""")

        return result

    except Exception as e:
        logger.error(f"❌ 處理失敗: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="音頻處理失敗")

    finally:
        for path in [temp_input_path, temp_wav_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                    logger.info(f"🗑️ 已刪除臨時文件: {path}")
                except Exception as e:
                    logger.warning(f"⚠️ 無法刪除文件: {path}, 錯誤: {e}")


# 診斷端點
@app.post("/diagnose-pronunciation")
async def diagnose_endpoint(request: Request):
    try:
        form = await request.form()
        target_word = (form.get("target_word") or "").strip()
        user_asr_input = (form.get("user_asr_input") or "").strip()
        difficulty_level = (form.get("difficulty_level") or "小學").strip()

        if not target_word:
            raise HTTPException(status_code=400, detail="缺少 target_word")
        if not user_asr_input:
            raise HTTPException(status_code=400, detail="缺少 user_asr_input")

        result = process_pronunciation_task(target_word, user_asr_input, difficulty_level)
        final_suggest = result.get("practice_word", " ")
        if not final_suggest or "錯誤" in final_suggest or "找不到" in final_suggest:
            final_suggest = " "

        return JSONResponse({
            "analysis": {
                "target_ipa": result.get("target_pinyin", "unknown"),
                "suggest_word": final_suggest,
                "decision": result.get("decision", ""),
                "error_rate": round(result.get("error_rate", 0), 3),
                "detected_errors": result.get("detected_errors", [])
            }
        })
    except Exception as e:
        logger.error(f"❌ 伺服器內部錯誤: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="內部錯誤")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "asr_model_loaded": processor is not None and asr_model is not None,
        "llm_loaded": llm is not None,
        "device": DEVICE,
        "version": "2.2"
    }


#啟動伺服器


if __name__ == "__main__":
    logger.info("🚀 啟動整合式 ASR + LLM 服務...")
    import uvicorn
    uvicorn.run("llm_asr_server:app", host="0.0.0.0", port=8000, reload=True)