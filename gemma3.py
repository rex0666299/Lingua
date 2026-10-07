import json
from llama_cpp import Llama
import random
import io
import re
from contextlib import redirect_stderr
import os
from opencc import OpenCC
from colorama import Fore, Style
import colorama

# 初始化 colorama
colorama.init()

print("--- 步驟 2: 定義核心 AI 資源並載入所有模型與數據 (V23.1 - 完整修正版) ---")

# 【V22.3 核心修正】: 使用正確的配置名稱
try:
    s2t = OpenCC('s2t')
    t2s = OpenCC('t2s')
    print("✅ OpenCC 繁簡轉換器已成功創建。")
except Exception as e:
    print(f"❌ 創建 OpenCC 轉換器時發生錯誤: {e}")
    s2t = lambda x: x
    t2s = lambda x: x

# --- 2.2. 定義所有檔案路徑 ---
model_file_path = r"C:\Users\r3x06\Desktop\mywhisper\model\Tiger-Gemma-9B-v3-Q3_K_M.gguf"
pinyin_dict_path_ch = r"C:\Users\r3x06\Desktop\mywhisper\model\pinyin_dict_ch_v3.json"
knowledge_base_path_ch = r"C:\Users\r3x06\Desktop\mywhisper\model\knowledge_base_ch_v3.json"

# --- 2.3. 初始化核心資源 ---
llm = None
pinyin_dict = None
rag_knowledge_base = None

try:
    # 載入 LLM 模型
    if os.path.exists(model_file_path):
        print(f"⚙️ 正在加載 Gemma 模型: '{os.path.basename(model_file_path)}'...")
        llm = Llama(
            model_path=model_file_path,
            n_gpu_layers=35,
            main_gpu=0,
            n_ctx=4096,
            verbose=False,
            chat_format='chatml'
        )
        print(f"✅ Gemma 模型 '{os.path.basename(model_file_path)}' 已成功加載。")
    else:
        print(f"❌ 模型檔案不存在: '{model_file_path}'")
        exit(1)

    # 加載拼音詞典
    if not os.path.exists(pinyin_dict_path_ch):
        raise FileNotFoundError(f"拼音詞典檔案不存在: {pinyin_dict_path_ch}")
    with open(pinyin_dict_path_ch, 'r', encoding='utf-8') as f:
        pinyin_dict = json.load(f)
    print(f"✅ 成功加載中文拼音詞典: '{pinyin_dict_path_ch}'")

    # 加載知識庫
    if not os.path.exists(knowledge_base_path_ch):
        raise FileNotFoundError(f"知識庫檔案不存在: {knowledge_base_path_ch}")
    with open(knowledge_base_path_ch, 'r', encoding='utf-8') as f:
        rag_knowledge_base = json.load(f)
    print(f"✅ 成功加載中文知識庫: '{knowledge_base_path_ch}'")

    print("\n🎉 所有核心資源準備就緒！")

except FileNotFoundError as e:
    print(f"❌ 檔案未找到: {e}")
    exit(1)
except json.JSONDecodeError as e:
    print(f"❌ JSON 解析錯誤（請檢查檔案格式）: {e}")
    exit(1)
except Exception as e:
    print(f"❌ 載入資源時發生未預期錯誤: {e}")
    exit(1)


# ==============================================================================
# 步驟 3: 定義核心功能 (V23 - 生產級封裝)
# ==============================================================================

print("--- 步驟 3: 定義所有核心功能 (V23 - 生產級封裝) ---")

# --- 3.1. 語言學定義與診斷功能 ---

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


# --- 3.2. LLM 交互與詞彙生成 ---

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
    is_comprehensive = len(error_keys) > 1
    if is_comprehensive:
        all_words_with_sounds = find_comprehensive_words(error_keys)
        sound_requirement_desc = f"這個詞必須同時包含 /{'/、/'.join(error_keys)}/ 這些發音。"
    else:
        all_words_with_sounds = rag_knowledge_base.get(error_keys[0], [])
        sound_requirement_desc = f"這個詞必須包含 /{error_keys[0]}/ 這個發音。"

    valid_words = [word for word in all_words_with_sounds if word in pinyin_dict]
    if not valid_words: return f"知識庫中找不到滿足所有發音要求的有效詞語。"

    candidate_pool = get_candidate_pool(difficulty_level, valid_words)
    if not candidate_pool: return f"未能為指定要求創建候選詞庫。"

    difficulty_instruction = DIFFICULTY_FEATURES_CH.get(difficulty_level, "詞語應該多樣化。")
    temperature_setting = DIFFICULTY_TEMPERATURES.get(difficulty_level, 0.5)

    prompt_template = """<start_of_turn>user
You are a linguistic analysis robot. Your task is to strictly follow rules to select the **single best word** from a given "Candidate Pool".

### Rules ###
1.  **Source**: Your output MUST ONLY be a single word from the "Candidate Pool" below.
2.  **Selection Criteria**: Analyze each word in the "Candidate Pool" against the "Difficulty Feature Description" and the "Sound Requirement" to select the **single best match**.
    - **Sound Requirement**: {sound_requirement_desc}
    - **Difficulty Feature Description ({difficulty_level})**: "{difficulty_instruction}"
3.  **Honesty Clause**: If you cannot find any word that truly matches all requirements, return an empty string.
4.  **Format**: Your output must be a single Chinese word. Nothing else.

### Candidate Pool ###
{candidate_list}

### Task ###
Strictly follow all rules. Select the single best word and output it now.
<end_of_turn>
<start_of_turn>model
"""

    final_prompt = prompt_template.format(
        difficulty_level=difficulty_level,
        difficulty_instruction=difficulty_instruction,
        sound_requirement_desc=sound_requirement_desc,
        candidate_list=", ".join(candidate_pool)
    )

    try:
        with io.StringIO() as f, redirect_stderr(f):
            response = llm(
                final_prompt,
                max_tokens=10,
                temperature=temperature_setting,
                stop=["<end_of_turn>", "<eos>", ","]
            )

        if response and response['choices'] and response['choices'][0]['text']:
            words = clean_llm_output(response['choices'][0]['text'].strip())
            return words[0] if words else "LLM 返回了空的內容。"
        else:
            return "LLM 返回了空的內容。"

    except Exception as e:
        return f"LLM 調用時發生錯誤: {e}"


# --- 3.3. 生產級主函數 ---

def process_pronunciation_task(target_word: str, user_asr_input: str, difficulty_level: str) -> dict:
    """
    (V23) 處理單次發音任務的生產級函數，返回一個包含診斷和練習詞的字典。
    """
    # 防禦：檢查必要資源
    if pinyin_dict is None:
        return {"status": "error", "message": "拼音詞典未加載，請檢查檔案。"}
    if rag_knowledge_base is None:
        return {"status": "error", "message": "知識庫未加載，請檢查檔案。"}

    # 步驟 1: 輸入預處理 (繁轉簡)
    simplified_target_word = t2s.convert(target_word)

    # 步驟 2: 查找拼音
    target_pinyin = pinyin_dict.get(simplified_target_word)
    if not target_pinyin:
        return {"status": "error", "message": f"在拼音字典中找不到 '{target_word}' 的拼音。"}

    # 步驟 3: 執行發音診斷
    detected_errors, error_rate, parsed_pinyin, matched_target = diagnose_pronunciation(target_pinyin, user_asr_input)

    # 步驟 4: 初始化返回結果字典
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

    # 步驟 5: 根據診斷結果進行決策
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

    # 步驟 6: 決定練習模式並生成練習詞
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

    # 步驟 7: 輸出後處理 (簡轉繁)
    result["practice_word"] = s2t.convert(generated_word)

    return result

print("✅ 所有核心功能已成功定義 (V23 - 生產級封裝)。\n")


# ==============================================================================
# 步驟 4: 測試環境 (V23 - 模擬後端調用)
# ==============================================================================

print("--- 步驟 4: 測試環境 (V23 - 模擬後端調用) ---")

pressure_tests = [
    {"target_word": "哪裡", "user_input": "l a3 l i3", "comment": "壓力測試: n/l 混淆"},
    {"target_word": "知道", "user_input": "z i1 d ao4", "comment": "壓力測試: zh/z 混淆"},
    {"target_word": "上班", "user_input": "s an4 b an1", "comment": "壓力測試: sh/s & ang/an 綜合練習"},
    {"target_word": "狀況", "user_input": "z uang4 k uang4", "comment": "壓力測試: 綜合練習降級"},
    {"target_word": "你好", "user_input": "n i3 h ao3", "comment": "壓力測試: 完美發音"},
]

user_selected_level = "小學"
print(f"--- 模擬真實場景：用戶選擇了【{user_selected_level}】難度 ---")

for i, case in enumerate(pressure_tests):
    print(f"\n\n========================= 測試案例 {i+1}: '{case['target_word']}' =========================\n")

    task_result = process_pronunciation_task(
        target_word=case["target_word"],
        user_asr_input=case["user_input"],
        difficulty_level=user_selected_level
    )

    print("【診斷層】")
    print(f"  - 目標: '{task_result['target_word']}' (正確拼音: {task_result['target_pinyin']})")
    print(f"  - 用戶 ASR 原始輸入: '{task_result['user_asr_input']}'")
    print(f"  - ASR 輸入解析結果: '{task_result['parsed_pinyin']}'")
    print(f"\n  - 診斷完成: 錯誤率約為 {task_result['error_rate']:.2%}")
    print(f"  - 檢測到的發音錯誤: {Fore.YELLOW}{task_result['detected_errors']}{Style.RESET_ALL}")

    print("\n【決策與生成層】")
    print(f"  - 決策: {task_result['decision']}")

    if task_result.get("practice_word"):
        practice_word = task_result["practice_word"]
        if practice_word and "錯誤" not in practice_word and "沒有" not in practice_word and "找不到" not in practice_word:
             print(f"      ➡️  練習詞: {Fore.GREEN}\"{practice_word}\"{Style.RESET_ALL}")
        else:
             print(f"      ➡️  練習詞: {Fore.YELLOW}[{practice_word}]{Style.RESET_ALL}")
        print("  -----------------------------------")

print("\n\n--- 所有中文版 V23.1 測試已完成 ---")