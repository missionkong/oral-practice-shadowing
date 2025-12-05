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
import pandas as pd # 新增 pandas 用於處理 CSV

# [核心] 使用 Google Generative AI
import google.generativeai as genai

# 1. 設定頁面
try:
    st.set_page_config(page_title="AI 英文教練 Pro (匯入增強版)", layout="wide", page_icon="🎓")
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
    # 檢查是否已存在 (不分大小寫)
    for v in vocab_list:
        if v["word"].lower() == word.lower(): return False
    
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

# [新功能] 處理匯入的檔案內容
def process_imported_text(text_content):
    # 1. 使用 Regex 只保留英文字母和空格/換行
    # [a-zA-Z]+ 匹配一個或多個英文字母
    words = re.findall(r'\b[a-zA-Z]+\b', text_content)
    
    # 2. 過濾掉過短的字 (例如 a, I 這種單字以外的雜訊) 或保留
    # 這裡假設保留所有長度 >= 2 的單字
    valid_words = [w for w in words if len(w) >= 2]
    
    # 3. 去重 (保留順序)
    seen = set()
    unique_words = []
    for w in valid_words:
        w_lower = w.lower()
        if w_lower not in seen:
            seen.add(w_lower)
            unique_words.append(w) # 這裡保留原始大小寫
            
    return unique_words

# ==========================================
# 1. UI 美化 (包含手機優化)
# ==========================================
def inject_custom_css():
    st.markdown("""
        <style>
        /* 全局設定 */
        .stApp { 
            background: linear-gradient(135deg, #fdfbf7 0%, #ebedee 100%); 
            font-family: 'Microsoft JhengHei', sans-serif; 
        }
        
        /* 強制主區域文字深色 */
        .main .block-container h1, 
        .main .block-container h2, 
        .main .block-container h3, 
        .main .block-container p, 
        .main .block-container div,
        .main .block-container span,
        .main .block-container label {
            color: #333333 !important;
        }

        /* 側邊欄樣式鎖定 (深色背景，淺色文字) */
        [data-testid="stSidebar"] {
            background-color: #263238 !important; 
        }
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] span, 
        [data-testid="stSidebar"] div, 
        [data-testid="stSidebar"] label {
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] input {
             color: #333333 !important;
        }

        /* 閱讀區塊 */
        .reading-box { 
            font-size: 26px !important; 
            font-weight: bold; 
            color: #2c3e50 !important; 
            line-height: 1.6; 
            padding: 20px; 
            background-color: #ffffff !important; 
            border-left: 8px solid #4285F4; 
            border-radius: 10px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
            margin-bottom: 20px; 
            white-space: pre-wrap; 
            font-family: 'Courier New', Courier, monospace; 
        }
        
        /* 單字卡片 */
        .definition-card { 
            background-color: #fff9c4 !important; 
            border: 2px solid #fbc02d; 
            color: #5d4037 !important; 
            padding: 15px; 
            border-radius: 12px; 
            margin-top: 15px; 
            font-size: 18px; 
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
            color: #1565c0 !important; 
            margin-bottom: 20px; 
            line-height: 1.6; 
        }
        
        /* 錯誤提示框 */
        .hint-box { 
            background-color: #ffebee !important; 
            color: #c62828 !important; 
            padding: 10px; 
            border-radius: 5px; 
            font-weight: bold; 
            margin-top: 10px; 
            border: 1px dashed #ef9a9a;
        }
        
        /* 按鈕 */
        div.stButton > button { 
            width: 100%; 
            border-radius: 8px; 
            height: 3em; 
            font-weight: bold; 
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
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 核心功能
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

def get_ai_coach_feedback(api_key, target_text, user_text, score):
    if not api_key: return "⚠️ 請在側邊欄輸入 Google API Key"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
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
        return f"AI 錯誤: {str(e)}"

@st.cache_data(show_spinner=False)
def get_word_info(_api_key, word, sentence):
    if not _api_key: return "⚠️ 請輸入 Google API Key"
    try:
        genai.configure(api_key=_api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"解釋單字 '{word}' 在句子 '{sentence}' 中的意思。格式：🔊[{word}] KK音標\\n🏷️[詞性]\\n💡[繁中意思](簡潔)"
        responses = model.generate_content(prompt, stream=False)
        return responses.text
    except Exception as e:
        return f"❌ 查詢失敗: {str(e)}"

def generate_quiz(api_key, word):
    if not api_key: return "錯誤：未檢測到 API Key"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""
        請針對英文單字 "{word}" 設計一個「拼字填空題」。
        
        嚴格遵守以下格式規則：
        Q: [英文句子，將 "{word}" 這個字挖空，並在挖空處用 `______ (該單字的繁體中文意思)` 來提示。例如：I walk on the ______ (街道).]
        A: [整句英文句子的繁體中文翻譯]
        """
        responses = model.generate_content(prompt, stream=False)
        
        raw_text = responses.text.strip()
        if "Q:" in raw_text:
            cleaned_text = raw_text[raw_text.find("Q:"):]
            return cleaned_text
        else:
            return raw_text
            
    except Exception as e:
        return f"Google API 報錯: {str(e)}"

def get_spelling_hint(word, attempts):
    length = len(word)
    if length <= 3:
        if attempts == 1:
            return f"_ " * length + f"({length}個字母)"
        else:
            return f"{word[0]} " + "_ " * (length - 1)
    else:
        if attempts == 1:
            return f"_ " * length + f"({length}個字母)"
        elif attempts == 2:
            return f"{word[0]} " + "_ " * (length - 1)
        elif attempts == 3:
            return f"{word[0]} " + "_ " * (length - 2) + f" {word[-1]}"
        else:
            reveal = min(attempts, length - 1)
            hint_str = ""
            for i in range(length):
                if i < reveal:
                    hint_str += f"{word[i]} "
                elif i == length - 1:
                    hint_str += f"{word[-1]}"
                else:
                    hint_str += "_ "
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

    if not google_api_key:
        st.warning("👉 請輸入 API Key 才能使用 AI 功能。")
    else:
        st.success("✅ API Key 已載入！")
    
    st.markdown("---")
    app_mode = st.radio("選擇模式", ["📖 跟讀練習", "📝 拼字測驗 (AI出題)", "👂 英聽拼字測驗"], index=0)
    
    # [看門狗邏輯] 偵測模式切換，強制重置 quiz_data
    if st.session_state.last_app_mode != app_mode:
        st.session_state.quiz_data = None
        st.session_state.quiz_state = "QUESTION"
        st.session_state.quiz_attempts = 0
        st.session_state.quiz_last_msg = ""
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

    st.markdown("---")
    
    # [新增] 外來單字庫匯入區
    with st.expander("📤 匯入外部單字檔", expanded=False):
        uploaded_txt = st.file_uploader("上傳純文字或CSV檔", type=["txt", "csv"])
        if uploaded_txt:
            if st.button("開始匯入分析"):
                # 讀取檔案內容
                stringio = uploaded_txt.getvalue().decode("utf-8")
                
                # 呼叫資料清洗邏輯
                new_words = process_imported_text(stringio)
                
                if not new_words:
                    st.warning("⚠️ 檔案中找不到有效的英文單字。")
                else:
                    added_count = 0
                    for w in new_words:
                        # 預設資訊先填 "待查詢"，讓使用者在練習時自己點擊查單字
                        # 這樣可以避免一次消耗大量 API 配額，也不會讓匯入卡太久
                        success = add_word_to_vocab(w, "💡 待查詢... (請在練習模式點擊查詢)")
                        if success:
                            added_count += 1
                    
                    if added_count > 0:
                        st.success(f"🎉 成功匯入 {added_count} 個新單字！")
                        time.sleep(1) # 讓使用者看到訊息後再重整
                        st.rerun()
                    else:
                        st.info("這些單字都已經在單字庫裡囉！")

    with st.expander("💾 單字庫備份與還原", expanded=False):
        st.write(f"目前單字：**{len(vocab_list)}** 個")
        if vocab_list:
            json_str = json.dumps(vocab_list, ensure_ascii=False, indent=4)
            st.download_button("📥 下載備份 (JSON)", json_str, "my_vocab.json", "application/json")
        uploaded_file = st.file_uploader("📤 上傳備份檔", type=["json"])
        if uploaded_file:
            try:
                data = json.load(uploaded_file)
                save_vocab_to_disk(data)
                st.success(f"已還原 {len(data)} 個單字！")
                st.rerun()
            except:
                 st.error("還原失敗，格式錯誤。")

st.title("🎤 AI 英文教練 Pro (匯入增強版)")

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
                        info = get_word_info(google_api_key, word, display_text)
                        st.session_state.current_word_info = info
                        if "查詢失敗" not in info and "請輸入 API Key" not in info:
                            w_path = speak_google(word, 1.0)
                            if not w_path: w_path = speak_offline(word, 1.0)
                            st.session_state.current_word_audio = w_path
                        else:
                            st.session_state.current_word_audio = None
            
            if not google_api_key:
                 st.warning