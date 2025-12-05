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

# [回歸原始] 使用 Google Generative AI (API Key)
import google.generativeai as genai

# 1. 設定頁面
try:
    st.set_page_config(page_title="AI 英文教練 Pro (排版修正版)", layout="wide", page_icon="🎓")
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
        with open(VOCAB_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except: return []

def save_vocab_to_disk(vocab_list):
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(vocab_list, f, ensure_ascii=False, indent=4)

def add_word_to_vocab(word, info):
    if not word or "查詢失敗" in info or "請輸入 API Key" in info or "Exception" in info: return False
    vocab_list = load_vocab()
    for v in vocab_list:
        if v["word"] == word: return False
    vocab_list.append({"word": word, "info": info})
    save_vocab_to_disk(vocab_list)
    return True

# ==========================================
# 1. UI 美化 (關鍵：保留排版)
# ==========================================
def inject_custom_css():
    st.markdown("""
        <style>
        .stApp { background: linear-gradient(135deg, #fdfbf7 0%, #ebedee 100%); font-family: 'Microsoft JhengHei', sans-serif; }
        
        /* [關鍵修改] white-space: pre-wrap; 確保換行被保留，不會擠在一起 */
        .reading-box { 
            font-size: 26px !important; 
            font-weight: bold; 
            color: #2c3e50; 
            line-height: 1.6; 
            padding: 20px; 
            background-color: #ffffff; 
            border-left: 8px solid #4285F4; 
            border-radius: 10px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
            margin-bottom: 20px; 
            white-space: pre-wrap; 
            font-family: 'Courier New', Courier, monospace; /* 使用等寬字體讓對齊更整齊 */
        }
        
        .definition-card { background-color: #fff9c4; border: 2px solid #fbc02d; color: #5d4037; padding: 15px; border-radius: 12px; margin-top: 15px; font-size: 18px; }
        .mobile-hint-card { background-color: #e3f2fd; border-left: 5px solid #2196f3; padding: 10px; border-radius: 8px; margin-bottom: 10px; font-size: 16px; font-weight: 600; color: #0d47a1; }
        .quiz-box { background-color: #ffffff; border: 2px solid #4caf50; padding: 25px; border-radius: 15px; margin-top: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); text-align: center;}
        .quiz-question { font-size: 28px; font-weight: bold; color: #1565c0; margin-bottom: 20px; }
        .backup-alert { background-color: #e8f5e9; border: 2px solid #66bb6a; padding: 20px; border-radius: 15px; text-align: center; margin-top: 20px; margin-bottom: 20px; }
        div.stButton > button { width: 100%; border-radius: 8px; height: 3em; font-weight: bold; }
        .ai-feedback-box { background-color: #f1f8e9; border-left: 5px solid #8bc34a; padding: 15px; border-radius: 10px; color: #33691e; margin-top: 20px;}
        .diff-box { background-color: #fff; border: 2px dashed #bdc3c7; padding: 15px; border-radius: 10px; font-size: 18px; }
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 核心功能
# ==========================================

def split_text_smartly(text):
    text = text.strip()
    
    # regex: 偵測行首是否為「數字+點」或「數字+空白」
    is_numbered = re.search(r'(?m)^\d+[\.\s]', text)
    
    segments = []
    
    if is_numbered:
        # [模式 A] 編號式：以數字為基準切割
        # raw_segments 包含原本的換行格式
        raw_segments = re.split(r'(?m)^(?=\d+[\.\s])', text)
        
        for s in raw_segments:
            if s.strip():
                # [關鍵] 這裡不做 replace('\n', ' ')，保留原文換行
                # 只把連續3個以上的換行，縮減為2個，避免過多空白
                cleaned_segment = re.sub(r'\n{3,}', '\n\n', s.strip())
                segments.append(cleaned_segment)
        
    else:
        # [模式 B] 連貫式：以標點符號切割 (這裡還是要變成單句)
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
        model = genai.GenerativeModel('gemini-pro')
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
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"解釋單字 '{word}' 在句子 '{sentence}' 中的意思。格式：🔊[{word}] KK音標\\n🏷️[詞性]\\n💡[繁中意思](簡潔)"
        responses = model.generate_content(prompt, stream=False)
        return responses.text
    except Exception as e:
        return f"❌ 查詢失敗: {str(e)}"

def generate_quiz(api_key, word):
    if not api_key: return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        請針對單字 "{word}" 出一個「句子填空題」。
        格式要求：
        Q: [英文句子，將 {word} 挖空變成 ______ ]
        A: [繁體中文翻譯]
        """
        responses = model.generate_content(prompt, stream=False)
        return responses.text
    except: return None

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
if 'quiz_answer_show' not in st.session_state: st.session_state.quiz_answer_show = False
if 'is_finished' not in st.session_state: st.session_state.is_finished = False
if 'segment_times' not in st.session_state: st.session_state.segment_times = {}
if 'start_time' not in st.session_state: st.session_state.start_time = None

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
    app_mode = st.radio("選擇模式", ["📖 跟讀練習", "📝 單字測驗 (AI出題)"], index=0)
    
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
    with st.expander("💾 單字庫管理", expanded=False):
        vocab_list = load_vocab()
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

st.title("🎤 AI 英文教練 Pro (排版修正版)")

# ==========================================
# 模式 A: 跟讀練習
# ==========================================
if app_mode == "📖 跟讀練習":
    if not st.session_state.game_active:
        st.markdown('<div class="reading-box">歡迎！請輸入文章開始練習。</div>', unsafe_allow_html=True)
        # 預設文字包含換行範例
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
            # [顯示區域] 會依照 CSS white-space: pre-wrap 保留換行
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
            # 這裡也用 pre-wrap 讓手機版提示卡保留換行
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
                    feedback = get_ai_coach_feedback(google_api_key, display_text, u_text, final_score)

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
# 模式 B: 單字測驗 (使用 API Key 出題)
# ==========================================
elif app_mode == "📝 單字測驗 (AI出題)":
    vocab_list = load_vocab()
    st.subheader("📝 單字本隨堂考")
    
    if not vocab_list:
        st.info("📭 目前單字庫是空的。請先去「跟讀練習」查詢單字並按「⭐ 收藏」。")
    else:
        st.write(f"📚 目前累積單字：**{len(vocab_list)}** 個")
        st.caption("點擊下方按鈕，AI 會從您的單字庫中隨機挑選一個字，並出一題填空題考考您！")
        
        if st.button("🎲 AI 隨機出一題", type="primary", use_container_width=True, disabled=not google_api_key):
            target = random.choice(vocab_list)
            word = target["word"]
            info = target["info"]

            with st.spinner(f"正在為 '{word}' 出題中..."):
                q_text = generate_quiz(google_api_key, word)
                if q_text and "失敗" not in q_text:
                    st.session_state.quiz_data = {"word": word, "content": q_text, "original_info": info}
                    st.session_state.quiz_answer_show = False
                else:
                    st.error("出題失敗，請檢查 API Key 或網路連線。")
        
        if not google_api_key:
             st.warning("👉 請先輸入 API Key，才能使用 AI 出題功能。")

        if st.session_state.quiz_data:
            data = st.session_state.quiz_data
            content = data["content"]
            try:
                q_part = content.split("A:")[0].replace("Q:", "").strip()
            except:
                q_part = content
            st.markdown(f"""
            <div class="quiz-box">
                <h3>❓ 填空題：</h3>
                <p style="font-size:22px; font-weight:bold; color:#1565c0;">{q_part}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("👀 看答案", use_container_width=True):
                st.session_state.quiz_answer_show = True
            
            if st.session_state.quiz_answer_show:
                st.success(f"✅ 正確單字：**{data['word']}**")
                try:
                    a_part = content.split("A:")[1].strip() if "A:" in content else "無翻譯"
                except:
                    a_part = "解析錯誤"
                st.info(f"💡 翻譯：{a_part}")

                st.markdown("---")
                st.caption("📜 您收藏的原始單字卡：")
                original_html = data['original_info'].replace('\n', '<br>')
                st.markdown(f'<div style="background-color:#fff9c4; padding:10px; border-radius:8px;">{original_html}</div>', unsafe_allow_html=True)

                w_path = speak_google(data['word'])
                if w_path: st.audio(w_path, format='audio/mp3')