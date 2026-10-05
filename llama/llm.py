# llm.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
from llama_cpp import Llama
from opencc import OpenCC
import random
import io
import re
from contextlib import redirect_stderr
import requests
import tempfile
import uvicorn

# -------------------------------
# 1. 初始化 FastAPI 應用
# -------------------------------
app = FastAPI(title="Gemma3 + Whisper 整合式 LLM 伺服器", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 2. 載入資源
# -------------------------------
try:
    s2t = OpenCC('s2t')  # 簡體 → 繁體
    t2s = OpenCC('t2s')  # 繁體 → 簡體
    print("✅ OpenCC 初始化成功")
except Exception as e:
    print(f"❌ OpenCC 初始化失敗: {e}")
    s2t = t2s = lambda x: x

# --- 路徑設定 ---
MODEL_PATH = r"C:\Users\r3x06\Desktop\theralingua-app\model\Tiger-Gemma-9B-v3-Q3_K_M.gguf"
PINYIN_DICT_PATH = r"C:\Users\r3x06\Desktop\theralingua-app\model\pinyin_dict_ch_v3.json"
KNOWLEDGE_BASE_PATH = r"C:\Users\r3x06\Desktop\theralingua-app\model\knowledge_base_final_complete.json"

# --- 加載模型與資料 ---
llm = None
pinyin_dict = None
rag_knowledge_base = None

try:
    if os.path.exists(MODEL_PATH):
        print(f"⚙️ 加載 LLM 模型: {os.path.basename(MODEL_PATH)}")
        llm = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=35,
            main_gpu=0,
            n_ctx=4096,
            verbose=False,
            chat_format='chatml'
        )
        print("✅ LLM 模型加載成功")
    else:
        print(f"⚠️ 模型檔案不存在: {MODEL_PATH}")

    with open(PINYIN_DICT_PATH, 'r', encoding='utf-8') as f:
        pinyin_dict = json.load(f)
    print(f"✅ 加載拼音字典: {PINYIN_DICT_PATH}")

    with open(KNOWLEDGE_BASE_PATH, 'r', encoding='utf-8') as f:
        rag_knowledge_base = json.load(f)
    print(f"✅ 加載知識庫: {KNOWLEDGE_BASE_PATH}")

    print("✅ 所有資源加載完成")

except Exception as e:
    print(f"❌ 資源加載失敗: {e}")

# ==============================================================================
# 3. 核心功能定義
# ==============================================================================

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
            scores = [
                (dp[i-1][j-1][0] + cost, 'diag'),
                (dp[i][j-1][0] + indel_cost, 'left'),
                (dp[i-1][j][0] + indel_cost, 'up')
            ]
            dp[i][j] = min(scores, key=lambda x: x[0])
    errors, i, j = [], n, m
    while i > 0 or j > 0:
        direction = dp[i][j][1]
        if direction == 'diag':
            if target_syllables[i-1] != actual_syllables[j-1]:
                errors.append({'type': 'Substitution', 'target': target_syllables[i-1], 'actual': actual_syllables[j-1]})
            i -= 1; j -= 1
        elif direction == 'up':
            errors.append({'type': 'Deletion', 'target': target_syllables[i-1], 'actual': None}); i -= 1
        else:
            errors.append({'type': 'Insertion', 'target': None, 'actual': actual_syllables[j-1]}); j -= 1
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
            if len(target_parts) > 1 and len(actual_parts) > 1 and target_parts[0] != actual_parts[0]:
                error_summary.append(target_parts[0])
            if target_parts[-1] != actual_parts[-1]:
                error_summary.append(target_parts[-1])
        elif error['type'] == 'Deletion':
            error_summary.extend(segment_pinyin_syllable(error['target']))
        elif error['type'] == 'Insertion':
            error_summary.append('__INSERTION__')
    unique_errors = sorted(list(dict.fromkeys(error_summary)))
    return unique_errors, total_error_rate, user_pinyin_str, best_target_pinyin


def clean_llm_output(generated_text: str) -> list:
    cleaned = re.sub(r'[^\u4e00-\u9fa5,、]', '', generated_text)
    cleaned = cleaned.replace('、', ',')
    words = [word.strip() for word in cleaned.split(',') if word.strip()]
    return list(dict.fromkeys(words))


DIFFICULTY_FEATURES_CH = {
    "幼兒園": "詞語特徵：1-2個字，極常見具體名詞或動詞。",
    "小學": "詞語特徵：2-4個字，常用詞，可含簡單成語。",
    "中學": "詞語特徵：2字以上，可為書面語或課本術語。",
    "成人": "詞語特徵：長度不限，可為專業術語或抽象詞。"
}
DIFFICULTY_TEMPERATURES = {"幼兒園": 0.0, "小學": 0.1, "中學": 0.2, "成人": 0.3}

NEGATIVE_KEYWORDS = ['菌', '症', '战役', '大学', '学院', '区', '县', '市', '省', '国', '洲', '洋', '条约', '协定', '法令', '研究所', '委员会', '有限公司', '集团', '主义', '共和國', '自治區', '經濟區', '高新區', '開發區', '風景區']


def get_candidate_pool(difficulty_level: str, all_words_pool: list, count: int = 60) -> list:
    filtered_pool = [w for w in all_words_pool if not any(neg_word in w for neg_word in NEGATIVE_KEYWORDS)]
    final_pool = []
    if difficulty_level == "幼兒園":
        final_pool = [w for w in filtered_pool if len(w) <= 2]
    elif difficulty_level == "小學":
        final_pool = [w for w in filtered_pool if 2 <= len(w) <= 4]
    else:
        final_pool = filtered_pool
    random.shuffle(final_pool)
    return final_pool[:count]


def find_comprehensive_words(error_keys: list) -> list:
    if not error_keys: return []
    base_words = set(rag_knowledge_base.get(error_keys[0], []))
    for key in error_keys[1:]:
        base_words.intersection_update(set(rag_knowledge_base.get(key, [])))
    return list(base_words)


def llm_select_one_word(error_keys: list, difficulty_level: str) -> str:
    is_comprehensive = len(error_keys) > 1
    if is_comprehensive:
        all_words_with_sounds = find_comprehensive_words(error_keys)
        sound_requirement_desc = f"必須同時包含 /{'/、/'.join(error_keys)}/ 發音。"
    else:
        all_words_with_sounds = rag_knowledge_base.get(error_keys[0], [])
        sound_requirement_desc = f"必須包含 /{error_keys[0]}/ 發音。"

    valid_words = [word for word in all_words_with_sounds if word in pinyin_dict]
    if not valid_words: return "知識庫中找不到有效詞語。"

    candidate_pool = get_candidate_pool(difficulty_level, valid_words)
    if not candidate_pool: return "無法建立候選詞庫。"

    difficulty_instruction = DIFFICULTY_FEATURES_CH.get(difficulty_level, "詞語應該多樣化。")
    temperature_setting = DIFFICULTY_TEMPERATURES.get(difficulty_level, 0.5)

    prompt = f"""<start_of_turn>user
請從以下候選詞中選出一個最符合要求的單一詞語：
- 必須是候選詞之一
- 必須符合發音要求
- 必須符合難度描述
- 只輸出一個詞，不要解釋

發音要求：{sound_requirement_desc}
難度描述（{difficulty_level}）：{difficulty_instruction}

候選詞：{", ".join(candidate_pool)}

請輸出最合適的詞：
<end_of_turn>
<start_of_turn>model
"""
    try:
        with io.StringIO() as f, redirect_stderr(f):
            response = llm(prompt, max_tokens=10, temperature=temperature_setting, stop=["<end_of_turn>", "<eos>"])
        text = response['choices'][0]['text'].strip() if response['choices'] else ""
        words = clean_llm_output(text)
        return words[0] if words else "未選出有效詞語。"
    except Exception as e:
        return f"LLM 錯誤: {e}"


def process_pronunciation_task(target_word: str, user_asr_input: str, difficulty_level: str) -> dict:
    simplified_target_word = t2s.convert(target_word)
    target_pinyin = pinyin_dict.get(simplified_target_word)
    if not target_pinyin:
        return {"status": "error", "message": f"找不到 '{target_word}' 的拼音。"}

    detected_errors, error_rate, parsed_pinyin, matched_target = diagnose_pronunciation(target_pinyin, user_asr_input)

    result = {
        "status": "success",
        "target_word": target_word,
        "target_pinyin": matched_target,
        "user_asr_input": user_asr_input,
        "parsed_pinyin": parsed_pinyin,
        "error_rate": round(error_rate, 3),
        "detected_errors": [s2t.convert(e) for e in detected_errors],
        "decision": "",
        "practice_word": "",
        "practice_type": ""
    }

    target_len = len(matched_target.split())
    absolute_error_count = round(error_rate * target_len)
    is_too_high = (error_rate > 0.75 and absolute_error_count > 2) or (error_rate >= 1.0 and target_len > 1)

    if is_too_high:
        result["decision"] = f"錯誤率過高 ({error_rate:.1%})，建議重新練習。"
    elif not detected_errors:
        result["decision"] = "發音準確！"
    else:
        trainable_errors = [e for e in detected_errors if e != '__INSERTION__']
        if not trainable_errors:
            result["decision"] = "未檢測到可訓練錯誤。"
        else:
            if len(trainable_errors) > 1:
                comprehensive_pool = find_comprehensive_words(trainable_errors)
                if comprehensive_pool:
                    result["decision"] = "生成綜合練習。"
                    result["practice_type"] = "comprehensive"
                    generated_word = llm_select_one_word(trainable_errors, difficulty_level)
                else:
                    result["decision"] = "降級為聚焦練習。"
                    result["practice_type"] = "focused"
                    generated_word = llm_select_one_word([trainable_errors[0]], difficulty_level)
            else:
                result["decision"] = "生成聚焦練習。"
                result["practice_type"] = "focused"
                generated_word = llm_select_one_word(trainable_errors, difficulty_level)
            result["practice_word"] = s2t.convert(generated_word)

    return result


# ==============================================================================
# 4. API 端點
# ==============================================================================

@app.post("/analyze-pronunciation")
async def analyze_pronunciation(
    target_word: str = Form(...),
    user_pinyin: str = Form(...),
    difficulty_level: str = Form("小學")
):
    """接收拼音進行發音分析"""
    try:
        result = process_pronunciation_task(target_word, user_pinyin, difficulty_level)
        return {"status": "success", "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "llm_loaded": llm is not None,
        "pinyin_dict_loaded": pinyin_dict is not None,
        "rag_knowledge_base_loaded": rag_knowledge_base is not None,
        "version": "2.0"
    }


if __name__ == "__main__":
    uvicorn.run("llm:app", host="0.0.0.0", port=8080, reload=True)