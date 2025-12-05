import streamlit as st
import json
import random
import os
import difflib
import re
import tempfile
import numpy as np
import time
import speech_recognition as sr
from gtts import gTTS
import ssl
import pandas as pd
import matplotlib.pyplot as plt

# [核心] 使用 Google Generative AI
import google.generativeai as genai

# 1. 設定頁面
try:
    st.set_page_config(page_title="AI 英文教練 Pro (手機完美版)", layout="wide", page_icon="🎓")
except:
    pass

# 2. 忽略 SSL 錯誤
ssl._create_default_https_context = ssl._create_unverified_context

# 3. 安全匯入離線套件
HAS_OFFLINE_TTS = False
try:
    import pyttsx3
    HAS_OFFLINE_TTS = True
except ImportError:
    HAS_OFFLINE_TTS = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

# ==========================================
# 0. 資料存取與輔助邏輯
# ==========================================
VOCAB_FILE = "vocab_book.json"
GRAMMAR_FILE = "grammar_stats.json"
KEY_FILE = "api_key.txt"

def load_vocab():
    if not os.path.exists(VOCAB_FILE): return []
    try:
        with open(VOCAB_FILE, "r", encoding="utf-8") as f: 
            data = json.load(f)
            for item in data:
                if "error_count" not in item:
                    item["error_count"] = 0
            return data
    except: return []

def save_vocab_to_disk(vocab_list):
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(vocab_list, f, ensure_ascii=False, indent=4)

def add_word_to_vocab(word, info):
    if not word or "查詢失敗" in info or "請輸入 API Key" in info or "Exception" in info: return False
    vocab_list = load_vocab()
    for v in vocab_list:
        if v["word"].lower() == word.lower():
            v["info"] = info
            save_vocab_to_disk(vocab_list)
            return True
    vocab_list.append({"word": word, "info": info, "error_count": 0})
    save_vocab_to_disk(vocab_list)
    return True

def increment_error_count(target_word):
    vocab_list = load_vocab()
    updated = False
    for v in vocab_list:
        if v["word"] == target_word:
            if "error_count" not in v: v["error_count"] = 0
            v["error_count"] += 1
            updated = True
            break
    if updated:
        save_vocab_to_disk(vocab_list)

# 載入文法統計
def load_grammar_stats():
    if not os.path.exists(GRAMMAR_FILE): return {}
    try:
        with open(GRAMMAR_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except: return {}

# 更新文法統計
def update_grammar_stats(topic, is_correct, question_text, user_answer, correct_answer, ai_feedback):
    stats = load_grammar_stats()
    if topic not in stats:
        stats[topic] = {"total": 0, "correct": 0, "errors": []}
    
    stats[topic]["total"] += 1
    if is_correct:
        stats[topic]["correct"] += 1
    else:
        new_error = {
            "time": time.strftime("%Y-%m-%d %H:%M"),
            "q": question_text,
            "user": user_answer,
            "ans": correct_answer,
            "feedback": ai_feedback
        }
        if "errors" not in stats[topic]: stats[topic]["errors"] = []
        stats[topic]["errors"].append(new_error)
        stats[topic]["errors"] = stats[topic]["errors"][-50:]
        
    with open(GRAMMAR_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)

def process_imported_text(text_content):
    words = re.findall(r'\b[a-zA-Z]+\b', text_content)
    valid_words = [w for w in words if len(w) >= 2]
    seen = set()
    unique_words = []
    for w in valid_words:
        w_lower = w.lower()
        if w_lower not in seen:
            seen.add(w_lower)
            unique_words.append(w)
    return unique_words

# 產生 AI 檢討報告
def generate_review_report(api_key, model_name, stats_data):
    if not api_key: return "⚠️ 請先輸入 API Key。"
    error_logs = []
    for topic, data in stats_data.items():
        if "errors" in data and data["errors"]:
            examples = data["errors"][-3:]
            for ex in examples:
                error_logs.append(f"題型: {topic} | 學生寫: {ex['user']} | 正解: {ex['ans']} | AI評語: {ex['feedback']}")

    if not error_logs:
        return "🎉 太棒了！目前的記錄中沒有發現錯誤，請繼續保持！"

    prompt = f"""
    你是一位專業的英文家教。以下是學生最近的文法練習錯誤紀錄：
    {json.dumps(error_logs, ensure_ascii=False, indent=2)}
    請根據這些錯誤，生成一份「學習診斷報告」：
    1. **錯誤模式分析**：學生是否有特定的盲點？
    2. **重點複習建議**：針對上述盲點，給出 3 個具體的文法複習重點。
    3. **鼓勵的話**：給學生正向的鼓勵。
    請用繁體中文回答，語氣溫柔專業。
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt, stream=False)
        return response.text
    except Exception as e:
        return f"報告生成失敗: {str(e)}"

# ==========================================
# 1. UI 美化 (重點修正：手機版強制配色)
# ==========================================
def inject_custom_css():
    st.markdown("""
        <style>
        /* --- 強制全域背景為淺色 (覆蓋手機深色模式) --- */
        .stApp { 
            background: linear-gradient(135deg, #fdfbf7 0%, #ebedee 100%) !important; 
            font-family: 'Microsoft JhengHei', sans-serif; 
        }

        /* ============================================================
           【主畫面修正】 強制所有文字為黑色 (#000000)
           ============================================================ */
        /* 包含標題、內文、列表、表格文字、Markdown */
        .main h1, .main h2, .main h3, .main h4, .main p, .main li, .main span, .main div, .main label, .main td, .main th {
            color: #000000 !important;
        }
        /* 修正主畫面輸入框的 Label 顏色 */
        .main .stTextInput label, .main .stSelectbox label, .main .stRadio label {
            color: #000000 !important;
        }
        /* 修正主畫面 Markdown 區塊 */
        .main .stMarkdown {
            color: #000000 !important;
        }

        /* ============================================================
           【側邊欄修正】 強制背景深色，文字白色 (#FFFFFF)
           ============================================================ */
        [data-testid="stSidebar"] {
            background-color: #263238 !important; /* 深藍灰色背景 */
        }
        /* 側邊欄所有標題、段落、Label 強制變白 */
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] span, 
        [data-testid="stSidebar"] div, 
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown {
            color: #ffffff !important;
        }
        
        /* 例外：側邊欄的輸入框 (Input) 內部文字必須是黑色 (因為輸入框背景通常是白) */
        [data-testid="stSidebar"] input {
            color: #000000 !important;
        }
        
        /* --- 特殊元件樣式 --- */
        /* 閱讀區塊 */
        .reading-box { 
            font-size: 26px !important; 
            font-weight: bold; 
            color: #000000 !important; 
            line-height: 1.6; 
            padding: 20px; 
            background-color: #ffffff !important; 
            border-left: 8px solid #4285F4; 
            border-radius: 10px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.15); 
            margin-bottom: 20px; 
            white-space: pre-wrap; 
            font-family: 'Courier New', Courier, monospace; 
        }
        
        /* 單字卡片 */
        .definition-card { 
            background-color: #fff9c4 !important; 
            border: 2px solid #fbc02d; 
            color: #3e2723 !important; 
            padding: 15px; 
            border-radius: 12px; 
            margin-top: 15px; 
            font-size: 18px; 
        }
        
        /* 提示卡 */
        .mobile-hint-card { 
            background-color: #e3f2fd !important; 
            border-left: 5px solid #2196f3; 
            padding: 10px; 
            border-radius: 8px; 
            margin-bottom: 10px; 
            font-size: 16px; 
            font-weight: 600; 
            color: #0d47a1 !important; 
        }
        
        /* 測驗區塊 */
        .quiz-box { 
            background-color: #ffffff !important; 
            border: 2px solid #4caf50; 
            padding: 25px; 
            border-radius: 15px; 
            margin-top: 10px; 
            box-shadow: 0 4px 10px rgba(0,0,0,0.1); 
            text-align: center;
        }
        .quiz-question { 
            font-size: 24px; 
            font-weight: bold; 
            color: #1b5e20 !important; 
            margin-bottom: 20px; 
            line-height: 1.6; 
        }
        
        /* 提示與排行榜 */
        .hint-box { 
            background-color: #ffebee !important; 
            color: #c62828 !important; 
            padding: 10px; 
            border-radius: 5px; 
            font-weight: bold; 
            margin-top: 10px; 
            border: 1px dashed #ef9a9a;
        }
        .leaderboard-box { 
            background-color: #fff3e0 !important; 
            padding: 10px; 
            border-radius: 8px; 
            border: 1px solid #ffcc80; 
            margin-bottom: 15px; 
            color: #e65100 !important; 
        }
        
        /* AI 回饋 */
        .ai-feedback-box { 
            background-color: #f1f8e9 !important; 
            border-left: 5px solid #8bc34a; 
            padding: 15px; 
            border-radius: 10px; 
            color: #33691e !important; 
            margin-top: 20px;
        }
        
        /* 按鈕 */
        div.stButton > button { 
            width: 100%; 
            border-radius: 8px; 
            height: 3em; 
            font-weight: bold; 
        }
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 核心功能 (修改為接收 model_name)
# ==========================================

def split_text_smartly(text):
    text = text.strip()
    is_numbered = re.search(r'(?m)^\d+[\.\s]', text)
    segments = []
    if is_numbered:
        raw_segments = re.split(r'(?m)^(?=\d+[\.\s])', text)
        for s in raw_segments:
            if s.strip():
                cleaned_segment = re.sub(r'\n{3,}', '\n\n', s.strip())
                segments.append(cleaned_segment)
    else:
        clean_text = text.replace('\n', ' ')
        raw_sentences = re.split(r'(?<=[.!?])\s+', clean_text)
        segments = [s.strip() for s in raw_sentences if len(s.strip()) > 0]
        if len(segments) > 0:
            segments.append("🌟 Full Text Review: " + clean_text)
    return segments

def transcribe_audio(audio_path):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = r.record(source)
            return r.recognize_google(audio_data, language="en-US")
    except: return ""

def check_similarity_visual(target, user_text):
    if not user_text: return 0, "無語音輸入"
    target_clean = target.replace("🌟 Full Text Review: ", "")
    t_words = re.findall(r"\w+", target_clean.lower())
    u_words = re.findall(r"\w+", user_text.lower())
    matcher = difflib.SequenceMatcher(None, t_words, u_words)
    score = matcher.ratio() * 100
    html_parts = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        t_segment = " ".join(t_words[i1:i2])
        u_segment = " ".join(u_words[j1:j2])
        if tag == 'equal': html_parts.append(f'<span style="color:green;font-weight:bold;">{t_segment}</span>')
        elif tag == 'replace': html_parts.append(f'<span style="color:red;text-decoration:line-through;">{t_segment}</span> <span style="color:gray;">({u_segment})</span>')
        elif tag == 'delete': html_parts.append(f'<span style="background-color:#ffcccc;color:red;">{t_segment}</span>')
        elif tag == 'insert': html_parts.append(f'<span style="color:gray;font-style:italic;">{u_segment}</span>')
    return score, " ".join(html_parts)

def plot_and_get_trend(teacher_path, student_path):
    if not HAS_LIBROSA: return None, 0, 0
    try:
        y_t, sr_t = librosa.load(teacher_path, sr=22050)
        f0_t, _, _ = librosa.pyin(y_t, fmin=50, fmax=400, frame_length=2048)
        y_s, sr_s = librosa.load(student_path, sr=22050)
        f0_s, _, _ = librosa.pyin(y_s, fmin=50, fmax=400, frame_length=2048)
        if f0_t is None or f0_s is None: return None, 0, 0
        def normalize(f0):
            valid = f0[~np.isnan(f0)]
            if len(valid) == 0: return np.array([])
            return (valid - np.mean(valid)) / (np.std(valid) + 1e-6)
        norm_t = normalize(f0_t)
        norm_s = normalize(f0_s)
        if len(norm_t) == 0 or len(norm_s) == 0: return None, 0, 0
        from scipy.signal import resample
        norm_s_res = resample(norm_s, len(norm_t))
        raw_pitch_score = max(0, np.corrcoef(norm_t, norm_s_res)[0, 1]) * 100
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(norm_t, label='Teacher', color='#42a5f5', linewidth=2)
        ax.plot(norm_s_res, label='You', color='#ffa726', linestyle='--', linewidth=2)
        ax.axis('off')
        plt.close(fig)
        return fig, raw_pitch_score, 0
    except: return None, 0, 0

def handle_ai_error(e, model_name):
    err_str = str(e)
    if "429" in err_str: return f"⚠️ {model_name} 額度已滿 (429)。請切換模型。"
    elif "404" in err_str: return f"❌ 找不到模型 {model_name} (404)。請嘗試使用自動偵測的模型。"
    else: return f"❌ AI 發生錯誤: {err_str}"

# 接收 model_name 參數
def get_ai_coach_feedback(api_key, model_name, target_text, user_text, score):
    if not api_key: return "⚠️ 請在側邊欄輸入 Google API Key"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        prompt = f"""
        你是一位溫暖的英文老師。
        目標句子："{target_text}"
        學生唸出："{user_text}"
        分數：{score:.0f}
        
        請給予繁體中文回饋：
        1. 🌟 亮點讚賞
        2. 🔧 具體發音糾正。
           **重要規則**：針對字尾的 'd' 或 't'，若因連讀(linking)或弱化(stop sound)而不清楚，視為正確。若學生將字尾 d/t 發得太重、太分離，請提醒：「字尾 d/t 試著輕一點或連讀，不要太用力」。
        3. 💪 暖心鼓勵
        """
        responses = model.generate_content(prompt, stream=False)
        return responses.text
    except Exception as e:
        return handle_ai_error(e, model_name)

# 接收 model_name 參數
@st.cache_data(show_spinner=False)
def get_word_info(_api_key, model_name, word, sentence):
    if not _api_key: return "⚠️ 請輸入 Google API Key"
    try:
        genai.configure(api_key=_api_key)
        model = genai.GenerativeModel(model_name)
        prompt = f"解釋單字 '{word}' 在句子 '{sentence}' 中的意思。格式：🔊[{word}] KK音標\\n🏷️[詞性]\\n💡[繁中意思](簡潔)"
        responses = model.generate_content(prompt, stream=False)
        return responses.text
    except Exception as e:
        return handle_ai_error(e, model_name)

def generate_quiz(api_key, model_name, word):
    if not api_key: return "錯誤：未檢測到 API Key"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        prompt = f"""
        請針對英文單字 "{word}" 設計一個「拼字填空題」。
        Q: [英文句子，將 "{word}" 這個字挖空，用 `______ (該單字的繁體中文意思)` 提示。]
        A: [整句英文句子的繁體中文翻譯]
        """
        responses = model.generate_content(prompt, stream=False)
        raw_text = responses.text.strip()
        if "Q:" in raw_text: return raw_text[raw_text.find("Q:"):]
        else: return raw_text
    except Exception as e:
        return handle_ai_error(e, model_name)

# 批次產生文法改寫題目
def generate_grammar_batch(api_key, model_name, count=10):
    if not api_key: return None, "錯誤：未輸入 API Key"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        # 完整的題型列表
        topics = [
            "現在式 Be動詞肯定句 -> 改否定句", "現在式 Be動詞肯定句 -> 改Yes/No疑問句",
            "過去式 Be動詞肯定句 -> 改否定句", "過去式 Be動詞肯定句 -> 改Yes/No疑問句",
            "現在簡單式 一般動詞肯定句 -> 改否定句 (do/does)", "現在簡單式 一般動詞肯定句 -> 改Yes/No疑問句 (do/does)",
            "過去簡單式 一般動詞肯定句 -> 改否定句 (did)", "過去簡單式 一般動詞肯定句 -> 改Yes/No疑問句 (did)",
            "現在進行式 肯定句 -> 改否定句", "現在進行式 肯定句 -> 改Yes/No疑問句",
            "過去進行式 肯定句 -> 改否定句", "過去進行式 肯定句 -> 改Yes/No疑問句",
            "There is/are 肯定句 -> 改否定句", "There is/are 肯定句 -> 改Yes/No疑問句",
            "There was/were 肯定句 -> 改否定句", "There was/were 肯定句 -> 改Yes/No疑問句",
            "情態動詞 (can/may/must) 肯定句 -> 改否定句", "情態動詞 (can/may/must) 肯定句 -> 改Yes/No疑問句",
            "現在簡單式 Yes/No疑問句 -> 改Wh-疑問句", "過去簡單式 Yes/No疑問句 -> 改Wh-疑問句",
            "Will 未來式肯定句 -> 改否定句", "Will 未來式肯定句 -> 改Yes/No疑問句",
            "Be going to 未來式肯定句 -> 改否定句", "Be going to 未來式肯定句 -> 改Yes/No疑問句",
            "現在簡單式肯定句 -> 改過去簡單式", "過去簡單式肯定句 -> 改現在簡單式",
            "形容詞比較級句子 -> 改最高級", "形容詞最高級句子 -> 改比較級",
            "祈使句 -> 改禮貌請求 (please/could you)",
            "現在簡單式 主動語態 -> 改被動語態", "現在簡單式 被動語態 -> 改主動語態",
            "過去簡單式 主動語態 -> 改被動語態", "過去簡單式 被動語態 -> 改主動語態",
            "現在完成式 肯定句 -> 改否定句", "現在完成式 肯定句 -> 改Yes/No疑問句", "現在完成式 肯定句 -> 改Wh-疑問句",
            "過去完成式 肯定句 -> 改否定句", "過去完成式 肯定句 -> 改Yes/No疑問句",
            "第一條件句 (未來可能) -> 改否定句", "第二條件句 (假設) -> 改否定句",
            "關係子句 (who/which/that) -> 改成兩個簡單句",
            "Because 因果句 -> 改成 So 結果句",
            "連接詞句子 (and/but/or) -> 改用其他連接詞",
            "Some 的句子 -> 改成 Any (否定/疑問)",
            "Much/Many 的句子 -> 改成 A lot of/Lots of",
            "Few/Little 的句子 -> 改成 Not many/Not much",
            "Have to 義務句 -> 改成 Must", "Can 能力句 -> 改成 Could (過去式)",
            "Will 預測句 -> 改成 Be going to 計劃句",
            "Too/Enough 句子 -> 改寫", "感嘆句 (How/What) -> 改成陳述句",
            "介系詞句子 -> 改換介系詞", "冠詞句子 -> 改無冠詞",
            "所有格句子 -> 改 Of 結構", "反身代名詞句子 -> 改一般代名詞",
            "現在進行式 (未來計劃) -> 改 Be going to",
            "頻率副詞句子 -> 改變位置", "副詞比較級句子 -> 改 as...as",
            "附加疑問句 (Tag Question) -> 改完整疑問句",
            "間接引語 (Reported Speech) -> 改直接引語 (Direct Speech)"
        ]
        
        prompt = f"""
        請產生 {count} 個英文句型改寫練習題。
        請從以下範圍中隨機挑選不同的題型 (不要重複)：
        {json.dumps(topics, ensure_ascii=False)}
        
        請嚴格使用 JSON 格式回傳一個 List (列表)，物件欄位必須包含 'topic' 以便統計。格式如下：
        [
            {{"topic": "所選的題型名稱", "source": "原始句子", "task": "改寫要求", "answer": "正確答案"}},
            {{"topic": "...", "source": "...", "task": "...", "answer": "..."}}
        ]
        不需要 Markdown 標記，直接回傳純 JSON 文字。
        """
        
        responses = model.generate_content(prompt, stream=False)
        raw_text = responses.text.strip()
        
        # 嘗試清理 Markdown
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_text:
            raw_text = raw_text.split("```")[1].strip()
            
        questions = json.loads(raw_text)
        return questions, None
        
    except Exception as e:
        return None, handle_ai_error(e, model_name)

# 檢查文法答案 (增加拼字檢查與 JSON 輸出)
def check_grammar_answer(api_key, model_name, question, user_answer, correct_answer):
    if not api_key: return False, "無法評分"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        prompt = f"""
        題目："{question}"
        要求目標："{correct_answer}"
        學生回答："{user_answer}"
        
        請判斷學生的回答是否正確。
        1. **嚴格檢查拼字**：如果有任何單字拼錯 (Typo)，請直接視為錯誤，並明確指出哪個字拼錯。
        2. 文法結構必須正確。
        
        請以 JSON 格式回傳：
        {{
            "is_correct": true 或 false,
            "feedback": "這裡寫繁體中文的評語、讚美或糾正內容"
        }}
        不需要 Markdown。
        """
        responses = model.generate_content(prompt, stream=False)
        text = responses.text.strip()
        
        if "```json" in text: text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text: text = text.split("```")[1].strip()
            
        result = json.loads(text)
        return result.get("is_correct", False), result.get("feedback", "解析錯誤")
        
    except Exception as e:
        return False, f"評分失敗: {str(e)}"

def get_spelling_hint(word, attempts):
    length = len(word)
    if length <= 3:
        if attempts == 1: return f"_ " * length + f"({length}個字母)"
        else: return f"{word[0]} " + "_ " * (length - 1)
    else:
        if attempts == 1: return f"_ " * length + f"({length}個字母)"
        elif attempts == 2: return f"{word[0]} " + "_ " * (length - 1)
        elif attempts == 3: return f"{word[0]} " + "_ " * (length - 2) + f" {word[-1]}"
        else:
            reveal = min(attempts, length - 1)
            hint_str = ""
            for i in range(length):
                if i < reveal: hint_str += f"{word[i]} "
                elif i == length - 1: hint_str += f"{word[-1]}"
                else: hint_str += "_ "
            return hint_str

def speak_google(text, speed=1.0):
    try:
        clean_text = text.replace("🌟 Full Text Review: ", "")
        is_slow = speed < 1.0
        tts = gTTS(text=clean_text, lang='en', slow=is_slow)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except: return None

def speak_offline(text, speed=1.0):
    if not HAS_OFFLINE_TTS: return None
    try:
        clean_text = text.replace("🌟 Full Text Review: ", "")
        engine = pyttsx3.init()
        engine.setProperty('rate', int(175 * speed))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            engine.save_to_file(clean_text, fp.name)
            engine.runAndWait()
            return fp.name
    except: return None

def get_offline_voices():
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        return {v.name: v.id for v in voices}
    except: return {}

# ==========================================
# 3. 主程式介面
# ==========================================
inject_custom_css()

# 初始化 State
if 'available_models' not in st.session_state: st.session_state.available_models = []
if 'game_active' not in st.session_state: st.session_state.game_active = False
if 'sentences' not in st.session_state: st.session_state.sentences = []
if 'current_index' not in st.session_state: st.session_state.current_index = 0
if 'current_word_info' not in st.session_state: st.session_state.current_word_info = None
if 'current_word_target' not in st.session_state: st.session_state.current_word_target = None
if 'current_word_audio' not in st.session_state: st.session_state.current_word_audio = None
if 'current_audio_path' not in st.session_state: st.session_state.current_audio_path = None
if 'quiz_data' not in st.session_state: st.session_state.quiz_data = None
if 'quiz_state' not in st.session_state: st.session_state.quiz_state = "QUESTION"
if 'is_finished' not in st.session_state: st.session_state.is_finished = False
if 'segment_times' not in st.session_state: st.session_state.segment_times = {}
if 'start_time' not in st.session_state: st.session_state.start_time = None
if 'quiz_attempts' not in st.session_state: st.session_state.quiz_attempts = 0
if 'quiz_last_msg' not in st.session_state: st.session_state.quiz_last_msg = ""
if 'quiz_error_counted' not in st.session_state: st.session_state.quiz_error_counted = False
if 'last_app_mode' not in st.session_state: st.session_state.last_app_mode = None
# [修改] 寫作練習的 state (支援佇列)
if 'grammar_queue' not in st.session_state: st.session_state.grammar_queue = []
if 'grammar_index' not in st.session_state: st.session_state.grammar_index = 0
if 'grammar_feedback' not in st.session_state: st.session_state.grammar_feedback = ""
if 'review_report' not in st.session_state: st.session_state.review_report = None # 儲存檢討報告

if 'saved_api_key' not in st.session_state:
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "r") as f: st.session_state.saved_api_key = f.read().strip()
    else: st.session_state.saved_api_key = ""

# --- 側邊欄 ---
with st.sidebar:
    st.title("⚙️ 設定")
    google_api_key = st.text_input("🔑 Google API Key", value=st.session_state.saved_api_key, type="password")
    if google_api_key != st.session_state.saved_api_key:
        with open(KEY_FILE, "w") as f: f.write(google_api_key)
        st.session_state.saved_api_key = google_api_key
        st.session_state.available_models = []

    # 自動偵測模型
    selected_model = "gemini-1.5-flash"
    if google_api_key:
        if not st.session_state.available_models:
            try:
                genai.configure(api_key=google_api_key)
                all_models = list(genai.list_models())
                st.session_state.available_models = [m.name.replace("models/", "") for m in all_models if "generateContent" in m.supported_generation_methods]
            except: pass
        
        if st.session_state.available_models:
            default_idx = 0
            for i, name in enumerate(st.session_state.available_models):
                if "1.5-flash" in name: 
                    default_idx = i
                    if "latest" in name: break
            st.success(f"✅ 已偵測到可用模型")
            selected_model = st.selectbox("🤖 選擇 AI 模型", st.session_state.available_models, index=default_idx)
        else:
            st.warning("無法自動偵測，請確認 Key")
            selected_model = st.text_input("手動輸入模型", "gemini-1.5-flash-latest")
    else:
        st.warning("👉 請輸入 API Key 才能使用 AI 功能。")
    
    st.markdown("---")
    # [修改] 加入新的模式選項
    app_mode = st.radio("選擇模式", ["📖 跟讀練習", "📝 拼字測驗 (AI出題)", "👂 英聽拼字測驗", "✍️ 句型改寫練習", "📚 單字庫檢視"], index=0)
    
    if st.session_state.last_app_mode != app_mode:
        st.session_state.quiz_data = None
        st.session_state.quiz_state = "QUESTION"
        st.session_state.quiz_attempts = 0
        st.session_state.quiz_last_msg = ""
        st.session_state.grammar_queue = [] # 重置寫作
        st.session_state.grammar_index = 0
        st.session_state.grammar_feedback = ""
        st.session_state.review_report = None # 重置報告
        st.session_state.last_app_mode = app_mode
        st.rerun()

    st.markdown("---")
    if HAS_OFFLINE_TTS:
        tts_mode = st.radio("發音引擎", ["☁️ 線上 (Google)", "💻 離線 (Windows)"], index=0)
    else:
        tts_mode = "☁️ 線上 (Google)"
    voice_speed = st.slider("語速", 0.5, 1.5, 1.0, 0.1)
    
    if st.session_state.segment_times:
        st.markdown("---")
        st.markdown("### ⏱️ 練習時間統計")
        for idx, duration in st.session_state.segment_times.items():
            label = "全文複習" if idx == len(st.session_state.sentences)-1 and "Full Text" in st.session_state.sentences[idx] else f"第 {idx+1} 段"
            st.caption(f"{label}: {duration:.1f} 秒")

    st.markdown("---")
    with st.expander("🔥 易錯單字排行榜", expanded=True):
        vocab_list = load_vocab()
        error_list = [v for v in vocab_list if v.get("error_count", 0) > 0]
        error_list.sort(key=lambda x: x["error_count"], reverse=True)
        if error_list:
            for i, v in enumerate(error_list[:5]): 
                st.write(f"**{i+1}. {v['word']}** (錯 {v['error_count']} 次)")
        else:
            st.caption("目前沒有拼錯紀錄，繼續保持！")

    with st.expander("💾 單字庫管理", expanded=False):
        st.write(f"目前單字：**{len(vocab_list)}** 個")
        if vocab_list:
            json_str = json.dumps(vocab_list, ensure_ascii=False, indent=4)
            st.download_button("📥 下載備份 (JSON)", json_str, "my_vocab.json", "application/json")
        uploaded_file = st.file_uploader("📤 上傳還原", type=["json"])
        if uploaded_file:
            try:
                data = json.load(uploaded_file)
                save_vocab_to_disk(data)
                st.success(f"已還原 {len(data)} 個單字！")
                st.rerun()
            except:
                 st.error("還原失敗，格式錯誤。")

    # [新增] 文法紀錄管理 (包含錯誤、統計與細節)
    with st.expander("💾 文法練習紀錄備份", expanded=False):
        stats = load_grammar_stats()
        # 計算總錯誤數 (方便顯示)
        total_errors = sum(len(item.get("errors", [])) for item in stats.values())
        st.write(f"目前紀錄：**{len(stats)}** 種題型")
        st.write(f"累計錯誤：**{total_errors}** 筆")
        
        if stats:
            stats_json = json.dumps(stats, ensure_ascii=False, indent=4)
            st.download_button("📥 下載紀錄 (JSON)", stats_json, "grammar_stats_backup.json", "application/json")
        
        uploaded_stats = st.file_uploader("📤 上傳還原紀錄", type=["json"], key="grammar_restore")
        if uploaded_stats:
            try:
                data = json.load(uploaded_stats)
                # 存回硬碟
                with open(GRAMMAR_FILE, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                st.success(f"✅ 已還原文法紀錄！")
                st.rerun()
            except:
                st.error("還原失敗，格式錯誤。")

st.title("🎤 AI 英文教練 Pro (最終UI版)")

# ==========================================
# 模式 A: 跟讀練習
# ==========================================
if app_mode == "📖 跟讀練習":
    if not st.session_state.game_active:
        st.markdown('<div class="reading-box">歡迎！請輸入文章開始練習。</div>', unsafe_allow_html=True)
        default_text = "1 Drug Store\nA: Excuse me, Is there a drug store in this neighborhood?\n\nB: Yes, There's a drug store on Main Street, across from the church.\n\n2 Clinic\nA: Excuse me, Is there a clinic?\n\nB: Yes, next to the bank."
        input_text = st.text_area("文章內容：", value=default_text, height=200)
        
        if st.button("🚀 開始練習", type="primary", use_container_width=True):
            s = split_text_smartly(input_text)
            if s: 
                st.session_state.sentences = s
                st.session_state.current_index = 0
                st.session_state.game_active = True
                st.session_state.is_finished = False
                st.session_state.start_time = time.time()
                st.session_state.segment_times = {}
                st.rerun()
    else:
        if st.session_state.is_finished:
            st.balloons()
            st.markdown("""
            <div class="backup-alert">
                <h2>🎉 練習結束！</h2>
                <p>別忘了去側邊欄下載您的單字庫備份喔！</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.segment_times:
                max_time_idx = max(st.session_state.segment_times, key=st.session_state.segment_times.get)
                max_time_val = st.session_state.segment_times[max_time_idx]
                st.info(f"💡 分析：您在第 {max_time_idx+1} 段花了最多時間 ({max_time_val:.1f}秒)。")

            if st.button("🔄 再練一次 / 回到首頁"):
                st.session_state.game_active = False
                st.session_state.is_finished = False
                st.session_state.segment_times = {}
                st.rerun()
            st.stop()

        idx = st.session_state.current_index
        sentences = st.session_state.sentences
        target_sentence = sentences[idx]

        def switch_page(increment):
            end_time = time.time()
            duration = end_time - st.session_state.start_time
            if idx in st.session_state.segment_times:
                st.session_state.segment_times[idx] += duration
            else:
                st.session_state.segment_times[idx] = duration
            st.session_state.current_index += increment
            st.session_state.current_audio_path = None
            st.session_state.start_time = time.time()
            st.rerun()

        c1, c2, c3 = st.columns([1, 4, 1])
        with c1: 
            if st.button("⬅️ 上句", disabled=(idx==0), use_container_width=True):
                switch_page(-1)
        with c2: st.progress((idx+1)/len(sentences), text=f"進度：{idx+1} / {len(sentences)}")
        with c3:
            is_last = (idx == len(sentences) - 1)
            btn_text = "完成 🎉" if is_last else "下句 ➡️"
            if st.button(btn_text, use_container_width=True):
                if is_last:
                    switch_page(0) 
                    st.session_state.is_finished = True
                    st.rerun()
                else:
                    switch_page(1)
        
        if st.button("🏁 中途結束", type="secondary", use_container_width=True):
             st.session_state.is_finished = True
             st.rerun()

        col_L, col_R = st.columns([1.5, 1], gap="large")

        with col_L:
            st.subheader("📖 閱讀")
            if "Full Text Review" in target_sentence:
                st.info("🌟 挑戰時間：全文連讀！")
            
            display_text = target_sentence.replace("🌟 Full Text Review: ", "")
            st.markdown(f'<div class="reading-box">{display_text}</div>', unsafe_allow_html=True)
            
            st.caption("👇 點擊查單字 (需輸入 API Key)：")
            words = re.findall(r"\b\w+\b", display_text)
            cols = st.columns(5)
            for i, word in enumerate(words):
                if cols[i % 5].button(word, key=f"w_{idx}_{i}", disabled=not google_api_key):
                    st.session_state.current_word_target = word
                    with st.spinner("🔍 AI 查詢中..."):
                        # 使用選擇的模型
                        info = get_word_info(google_api_key, selected_model, word, display_text)
                        st.session_state.current_word_info = info
                        if "查詢失敗" not in info and "請輸入 API Key" not in info:
                            w_path = speak_google(word, 1.0)
                            if not w_path: w_path = speak_offline(word, 1.0)
                            st.session_state.current_word_audio = w_path
                        else:
                            st.session_state.current_word_audio = None
            
            if not google_api_key:
                 st.warning("👉 請先在側邊欄輸入 API Key，才能使用單字查詢功能。")

            if st.session_state.current_word_info:
                info_html = st.session_state.current_word_info.replace('\n', '<br>')
                st.markdown(f'<div class="definition-card">{info_html}</div>', unsafe_allow_html=True)
                
                c_p, c_s = st.columns([4, 1])
                with c_p:
                    if st.session_state.current_word_audio:
                        st.audio(st.session_state.current_word_audio, format='audio/mp3')
                with c_s:
                    if "查詢失敗" not in st.session_state.current_word_info and "請輸入 API Key" not in st.session_state.current_word_info:
                        if st.button("⭐ 收藏加入單字庫", use_container_width=True, type="primary"):
                            saved = add_word_to_vocab(st.session_state.current_word_target, st.session_state.current_word_info)
                            if saved: st.toast("✅ 已成功收藏！")
                            else: st.toast("⚠️ 單字庫裡已經有囉！")

            st.markdown("---")
            st.subheader("🗣️ 示範")
            if st.session_state.current_audio_path is None:
                path = None
                if "線上" in tts_mode: path = speak_google(display_text, voice_speed)
                if not path: path = speak_offline(display_text, voice_speed)
                st.session_state.current_audio_path = path

            if st.session_state.current_audio_path:
                st.audio(st.session_state.current_audio_path, format="audio/mp3")
            else:
                st.warning("無法生成語音")

        with col_R:
            st.subheader("🎙️ 口說")
            st.markdown(f'<div class="mobile-hint-card" style="white-space: pre-wrap;">📖 跟讀：<br>{display_text}</div>', unsafe_allow_html=True)
            
            user_audio = st.audio_input("錄音", key=f"rec_{idx}", disabled=not google_api_key)
            if not google_api_key:
                 st.warning("👉 請先輸入 API Key，才能使用口說評分功能。")
            
            if user_audio and st.session_state.current_audio_path and google_api_key:
                with st.spinner("🤖 AI 分析中..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(user_audio.read()); user_path = tmp.name
                    
                    u_text = transcribe_audio(user_path)
                    score_text, diff_html = check_similarity_visual(display_text, u_text)
                    fig, raw_pitch_score, _ = plot_and_get_trend(st.session_state.current_audio_path, user_path)
                    
                    adj_pitch = max(60, raw_pitch_score)
                    final_score = (score_text * 0.8) + (adj_pitch * 0.2)
                    # 使用選擇的模型
                    feedback = get_ai_coach_feedback(google_api_key, selected_model, display_text, u_text, final_score)

                if final_score >= 80: st.success(f"🎉 分數：{final_score:.0f}")
                else: st.info(f"💪 分數：{final_score:.0f}")
                
                st.write("🎧 回放自己：")
                st.audio(user_path, format="audio/wav")
                st.markdown(f'<div class="ai-feedback-box">{feedback}</div>', unsafe_allow_html=True)
                
                tab1, tab2 = st.tabs(["🔤 糾錯", "📈 語調"])
                with tab1: st.markdown(f'<div class="diff-box">{diff_html}</div>', unsafe_allow_html=True)
                with tab2: 
                    if fig: st.pyplot(fig)
                    else: st.info("無法分析語調")

# ==========================================
# 模式 B: 拼字測驗 (AI出題)
# ==========================================
elif app_mode == "📝 拼字測驗 (AI出題)":
    vocab_list = load_vocab()
    st.subheader("📝 單字本拼字測驗")
    
    if not vocab_list:
        st.info("📭 目前單字庫是空的。請先去「跟讀練習」查詢單字並按「⭐ 收藏」。")
    else:
        st.write(f"📚 目前累積單字：**{len(vocab_list)}** 個")
        st.caption("點擊下方按鈕，AI 會出題讓您練習「拼字」！")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            if st.button("🎲 AI 隨機出一題", type="primary", use_container_width=True, disabled=not google_api_key):
                target = random.choice(vocab_list)
                word = target["word"]
                info = target["info"]

                with st.spinner(f"正在為 '{word}' 出題中..."):
                    # 使用選擇的模型
                    q_text = generate_quiz(google_api_key, selected_model, word)
                    if q_text and "Q:" in q_text and "A:" in q_text:
                        st.session_state.quiz_data = {"word": word, "content": q_text, "original_info": info}
                        st.session_state.quiz_state = "QUESTION"
                        st.session_state.quiz_attempts = 0
                        st.session_state.quiz_last_msg = ""
                        st.session_state.quiz_error_counted = False
                        st.rerun()
                    else:
                        st.error(f"出題失敗：{q_text}")
        
        if not google_api_key:
             st.warning("👉 請先輸入 API Key，才能使用 AI 出題功能。")

        if st.session_state.quiz_data:
            data = st.session_state.quiz_data
            
            # [雙重防呆]
            if 'content' not in data:
                st.warning("⚠️ 偵測到模式切換，請重新點擊上方紅色按鈕出題。")
                st.session_state.quiz_data = None
                st.stop()
            
            content = data["content"]
            try:
                q_part = content.split("A:")[0].replace("Q:", "").strip()
            except:
                q_part = content
            
            st.markdown(f"""
            <div class="quiz-box">
                <h3>❓ 填空拼字：</h3>
                <p class="quiz-question">{q_part}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.quiz_state == "RESULT":
                st.success(f"🎉 答對了！答案就是 **{data['word']}**")
                
                # 自動修復單字卡
                if "待查詢" in data['original_info'] and google_api_key:
                    with st.spinner("🤖 正在為您自動補上單字定義..."):
                        # 使用選擇的模型
                        new_info = get_word_info(google_api_key, selected_model, data['word'], f"The word is {data['word']}")
                        if "查詢失敗" not in new_info:
                            data['original_info'] = new_info
                            add_word_to_vocab(data['word'], new_info)
                            st.toast("✨ 單字卡已自動修復！")

                try:
                    a_part = content.split("A:")[1].strip() if "A:" in content else "無翻譯"
                except:
                    a_part = "解析錯誤"
                st.info(f"💡 翻譯：{a_part}")

                st.markdown("---")
                st.caption("📜 原始單字卡：")
                original_html = data['original_info'].replace('\n', '<br>')
                st.markdown(f'<div style="background-color:#fff9c4; padding:10px; border-radius:8px;">{original_html}</div>', unsafe_allow_html=True)

                w_path = speak_google(data['word'])
                if w_path: st.audio(w_path, format='audio/mp3')
                
                if st.button("下一題", use_container_width=True):
                    target = random.choice(vocab_list)
                    word = target["word"]
                    info = target["info"]
                    with st.spinner(f"正在為 '{word}' 出題中..."):
                        # 使用選擇的模型
                        q_text = generate_quiz(google_api_key, selected_model, word)
                        if q_text and "Q:" in q_text and "A:" in q_text:
                            st.session_state.quiz_data = {"word": word, "content": q_text, "original_info": info}
                            st.session_state.quiz_state = "QUESTION"
                            st.session_state.quiz_attempts = 0
                            st.session_state.quiz_last_msg = ""
                            st.session_state.quiz_error_counted = False
                            st.rerun()

            else:
                user_spelling = st.text_input("✍️ 請輸入您的答案：", key="spelling_input")
                
                c_sub, c_giveup = st.columns([2, 1])
                with c_sub:
                    if st.button("送出檢查", use_container_width=True):
                        correct_word = data['word'].strip().lower()
                        user_word = user_spelling.strip().lower()
                        
                        if correct_word == user_word:
                            st.balloons()
                            st.session_state.quiz_state = "RESULT"
                            st.rerun()
                        else:
                            st.session_state.quiz_attempts += 1
                            if not st.session_state.quiz_error_counted:
                                increment_error_count(data['word'])
                                st.session_state.quiz_error_counted = True
                            
                            hint = get_spelling_hint(data['word'], st.session_state.quiz_attempts)
                            st.session_state.quiz_last_msg = f"❌ 拼錯了 (嘗試 {st.session_state.quiz_attempts} 次)<br>💡 提示：{hint}"
                            st.rerun()
                
                with c_giveup:
                    if st.button("🏳️ 放棄，看答案", use_container_width=True):
                        if not st.session_state.quiz_error_counted:
                            increment_error_count(data['word'])
                            st.session_state.quiz_error_counted = True
                        st.session_state.quiz_state = "RESULT"
                        st.rerun()

                if st.session_state.quiz_last_msg:
                    st.markdown(f'<div class="hint-box">{st.session_state.quiz_last_msg}</div>', unsafe_allow_html=True)

# ==========================================
# 模式 C: 英聽拼字測驗 (英聽修復版)
# ==========================================
elif app_mode == "👂 英聽拼字測驗":
    vocab_list = load_vocab()
    st.subheader("👂 單字本英聽測驗")
    
    if not vocab_list:
        st.info("📭 目前單字庫是空的。請先去「跟讀練習」查詢單字並按「⭐ 收藏」。")
    else:
        st.write(f"📚 目前累積單字：**{len(vocab_list)}** 個")
        st.caption("點擊下方按鈕，系統會播放發音，請您拼出單字！")
        
        # [功能] 隨機選字並產生音檔
        if st.button("🎧 播放題目 (隨機單字)", type="primary", use_container_width=True):
            target = random.choice(vocab_list)
            word = target["word"]
            info = target["info"]
            
            w_path = speak_google(word)
            if not w_path: w_path = speak_offline(word)
            
            st.session_state.quiz_data = {"word": word, "audio": w_path, "original_info": info}
            st.session_state.quiz_state = "QUESTION"
            st.session_state.quiz_attempts = 0
            st.session_state.quiz_last_msg = ""
            st.session_state.quiz_error_counted = False
            st.rerun()

        if st.session_state.quiz_data:
            data = st.session_state.quiz_data
            
            # [雙重防呆]
            if 'audio' not in data:
                st.warning("⚠️ 偵測到模式切換，請重新點擊上方紅色按鈕播放題目。")
                st.session_state.quiz_data = None
                st.stop()

            st.markdown("""
            <div class="quiz-box">
                <h3>🎧 請聽音拼字：</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if 'audio' in data and data['audio']:
                st.audio(data['audio'], format='audio/mp3')
            else:
                st.error("無法生成語音")

            if st.session_state.quiz_state == "RESULT":
                st.success(f"🎉 答對了！答案就是 **{data['word']}**")
                
                # [自動修復] 檢查原始單字卡是否為 "待查詢"
                if "待查詢" in data['original_info'] and google_api_key:
                    with st.spinner("🤖 正在為您自動補上單字定義..."):
                        # 使用選擇的模型
                        new_info = get_word_info(google_api_key, selected_model, data['word'], f"The word is {data['word']}")
                        if "查詢失敗" not in new_info:
                            data['original_info'] = new_info
                            add_word_to_vocab(data['word'], new_info)
                            st.toast("✨ 單字卡已自動修復！")

                st.markdown("---")
                st.caption("📜 原始單字卡：")
                original_html = data['original_info'].replace('\n', '<br>')
                st.markdown(f'<div style="background-color:#fff9c4; padding:10px; border-radius:8px;">{original_html}</div>', unsafe_allow_html=True)
                
                if st.button("下一題", use_container_width=True):
                    target = random.choice(vocab_list)
                    word = target["word"]
                    info = target["info"]
                    
                    w_path = speak_google(word)
                    if not w_path: w_path = speak_offline(word)
                    
                    st.session_state.quiz_data = {"word": word, "audio": w_path, "original_info": info}
                    st.session_state.quiz_state = "QUESTION"
                    st.session_state.quiz_attempts = 0
                    st.session_state.quiz_last_msg = ""
                    st.session_state.quiz_error_counted = False
                    st.rerun()
            else:
                user_spelling = st.text_input("✍️ 請輸入您的答案：", key="listening_input")
                
                c_sub, c_giveup = st.columns([2, 1])
                with c_sub:
                    if st.button("送出檢查", use_container_width=True):
                        correct_word = data['word'].strip().lower()
                        user_word = user_spelling.strip().lower()
                        
                        if correct_word == user_word:
                            st.balloons()
                            st.session_state.quiz_state = "RESULT"
                            st.rerun()
                        else:
                            st.session_state.quiz_attempts += 1
                            if not st.session_state.quiz_error_counted:
                                increment_error_count(data['word'])
                                st.session_state.quiz_error_counted = True
                            
                            hint = get_spelling_hint(data['word'], st.session_state.quiz_attempts)
                            st.session_state.quiz_last_msg = f"❌ 拼錯了 (嘗試 {st.session_state.quiz_attempts} 次)<br>💡 提示：{hint}"
                            st.rerun()
                
                with c_giveup:
                    if st.button("🏳️ 放棄，看答案", use_container_width=True):
                        if not st.session_state.quiz_error_counted:
                            increment_error_count(data['word'])
                            st.session_state.quiz_error_counted = True
                        st.session_state.quiz_state = "RESULT"
                        st.rerun()

                if st.session_state.quiz_last_msg:
                    st.markdown(f'<div class="hint-box">{st.session_state.quiz_last_msg}</div>', unsafe_allow_html=True)

# ==========================================
# [新增模式] ✍️ 句型改寫練習 (批次10題極速版 + 嚴格拼字檢查 + 弱點分析)
# ==========================================
elif app_mode == "✍️ 句型改寫練習":
    st.subheader("✍️ 句型改寫練習 (嚴格拼字版)")
    st.info("AI 會隨機出題，請依指示改寫句子（例如：肯定句改否定句）。")
    
    # 載入統計資料
    stats = load_grammar_stats()

    if not google_api_key:
        st.warning("👉 請先輸入 API Key 才能使用 AI 出題。")
    else:
        # 出題區
        if not st.session_state.grammar_queue:
            if st.button("🎲 AI 隨機出題 (一次生成10題)", type="primary", use_container_width=True):
                with st.spinner("🤖 正在設計 10 道題目... (請稍等約 3~5 秒)"):
                    data_list, err = generate_grammar_batch(google_api_key, selected_model, count=10)
                    if data_list:
                        st.session_state.grammar_queue = data_list
                        st.session_state.grammar_index = 0
                        st.session_state.grammar_feedback = ""
                        st.session_state.review_report = None # 清空舊報告
                        st.rerun()
                    else:
                        st.error(err)

        # 答題區 (如果有題目)
        if st.session_state.grammar_queue:
            # 進度條
            current_q = st.session_state.grammar_index + 1
            total_q = len(st.session_state.grammar_queue)
            st.progress(current_q / total_q, text=f"進度：{current_q} / {total_q}")
            
            # 取出當前題目
            q = st.session_state.grammar_queue[st.session_state.grammar_index]

            st.markdown(f"""
            <div class="quiz-box">
                <p style="font-size: 20px; color: #555;">題目句子：</p>
                <h3 style="color: #1b5e20;">{q['source']}</h3>
                <hr>
                <p style="font-size: 18px; font-weight: bold; color: #d84315;">👉 任務：{q['task']}</p>
            </div>
            """, unsafe_allow_html=True)

            user_input = st.text_input("✍️ 請輸入您的答案：", key=f"grammar_input_{st.session_state.grammar_index}")

            # 檢查按鈕
            if st.button("送出檢查", use_container_width=True, key=f"check_btn_{st.session_state.grammar_index}"):
                if user_input.strip():
                    with st.spinner("🤖 AI 老師正在批改 (嚴格拼字模式)..."):
                        is_correct, feedback = check_grammar_answer(
                            google_api_key, 
                            selected_model, 
                            f"將 '{q['source']}' 改寫為 {q['task']}", 
                            user_input, 
                            q['answer']
                        )
                        st.session_state.grammar_feedback = (is_correct, feedback)
                        
                        # [更新統計] (包含詳細錯誤日誌)
                        topic_name = q.get('topic', q.get('task', 'Unknown'))
                        update_grammar_stats(topic_name, is_correct, q['source'], user_input, q['answer'], feedback)
                else:
                    st.warning("請先輸入答案喔！")

            # 顯示回饋與下一題按鈕
            if st.session_state.grammar_feedback:
                is_correct, feedback_text = st.session_state.grammar_feedback
                
                if is_correct:
                    st.markdown(f'<div class="ai-feedback-box" style="border-left: 5px solid #4caf50; background-color: #e8f5e9;">🎉 正確！<br>{feedback_text}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="ai-feedback-box" style="border-left: 5px solid #f44336; background-color: #ffebee;">❌ 錯誤<br>{feedback_text}</div>', unsafe_allow_html=True)
                
                with st.expander("👀 查看參考答案"):
                    st.info(f"參考答案：{q['answer']}")
                
                st.markdown("---")
                # 判斷是否還有下一題
                if current_q < total_q:
                    if st.button("下一題 ➡️", type="primary", use_container_width=True):
                        st.session_state.grammar_index += 1
                        st.session_state.grammar_feedback = ""
                        st.rerun()
                else:
                    if st.button("🏁 完成！再來一組 (10題)", type="primary", use_container_width=True):
                        st.session_state.grammar_queue = [] # 清空以重新生成
                        st.session_state.grammar_index = 0
                        st.session_state.grammar_feedback = ""
                        st.session_state.review_report = None # 重置報告
                        st.rerun()

    # [新增] 弱點分析報表 (含詳細日誌)
    st.markdown("---")
    with st.expander("📊 您的文法弱點分析", expanded=True):
        if stats:
            # 轉換為 DataFrame
            data = []
            for topic, s in stats.items():
                accuracy = (s['correct'] / s['total']) * 100 if s['total'] > 0 else 0
                data.append({"題型": topic, "練習題數": s['total'], "正確數": s['correct'], "正確率": f"{accuracy:.1f}%", "raw_acc": accuracy})
            
            df_stats = pd.DataFrame(data)
            # 依照正確率由低到高排序 (找出弱點)
            df_stats = df_stats.sort_values(by="raw_acc", ascending=True)
            
            st.dataframe(
                df_stats[["題型", "正確率", "練習題數", "正確數"]], 
                use_container_width=True, 
                hide_index=True
            )
            
            # [新增] AI 綜合檢討報告按鈕
            if st.button("📑 生成 AI 綜合檢討報告 (分析歷史錯誤)", type="secondary"):
                with st.spinner("🧠 AI 顧問正在分析所有歷史錯誤，請稍候..."):
                    report_text = generate_review_report(google_api_key, selected_model, stats)
                    st.session_state.review_report = report_text
            
            if st.session_state.review_report:
                st.markdown("### 📝 AI 檢討報告")
                st.markdown(st.session_state.review_report)
                
            # [新增] 詳細錯誤追蹤日誌 (可展開)
            st.markdown("### 🕵️‍♀️ 詳細錯誤追蹤日誌")
            for topic, s in stats.items():
                if "errors" in s and s["errors"]:
                    with st.expander(f"❌ {topic} ({len(s['errors'])} 筆錯誤)"):
                        for err in reversed(s["errors"]): # 最新錯誤在最上面
                            st.markdown(f"""
                            **時間**: {err.get('time', 'N/A')}
                            - **題目**: {err.get('q', 'N/A')}
                            - **您的回答**: `{err.get('user', 'N/A')}`
                            - **正確答案**: `{err.get('ans', 'N/A')}`
                            - **AI 點評**: {err.get('feedback', 'N/A')}
                            ---
                            """)
        else:
            st.info("目前還沒有練習記錄，快開始練習吧！")

# ==========================================
# [新增模式] 📚 單字庫檢視
# ==========================================
elif app_mode == "📚 單字庫檢視":
    st.subheader("📚 完整單字庫列表")
    vocab_list = load_vocab()
    
    if vocab_list:
        df = pd.DataFrame(vocab_list)
        if "error_count" not in df.columns: df["error_count"] = 0
        if "info" not in df.columns: df["info"] = ""
        
        df_display = df[["word", "error_count", "info"]].rename(columns={
            "word": "單字",
            "error_count": "錯誤次數",
            "info": "詳細定義"
        })
        
        # [排序選擇]
        sort_option = st.radio("排序方式：", ["🔥 錯誤次數 (由多到少)", "🔤 字母順序 (A-Z)"], horizontal=True)
        
        if sort_option == "🔥 錯誤次數 (由多到少)":
            df_display = df_display.sort_values(by="錯誤次數", ascending=False)
        else:
            df_display = df_display.sort_values(by="單字", ascending=True)
        
        col1, col2 = st.columns(2)
        col1.metric("總單字數", len(vocab_list))
        col2.metric("曾拼錯單字數", len(df[df["error_count"] > 0]))
        
        st.dataframe(df_display, use_container_width=True, height=600, hide_index=True)
    else:
        st.info("📭 目前單字庫是空的，請先去「跟讀練習」加入單字！")