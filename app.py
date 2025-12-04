import streamlit as st
# 1. 設定頁面 (保持在第一行)
st.set_page_config(page_title="AI 英文教練 Pro (Google版)", layout="wide", page_icon="🎤")

import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import tempfile
import os
import difflib
import re
import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai
import ssl

# 2. 忽略 SSL 錯誤
ssl._create_default_https_context = ssl._create_unverified_context

# 3. 嘗試匯入 librosa
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

# ==========================================
# 0. UI 美化
# ==========================================
def inject_custom_css():
    st.markdown("""
        <style>
        .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); font-family: 'Microsoft JhengHei', sans-serif; }
        
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
            margin-bottom: 25px; 
        }

        .definition-card { 
            background-color: #fff9c4; 
            border: 2px solid #fbc02d; 
            color: #5d4037; 
            padding: 20px; 
            border-radius: 12px; 
            margin-top: 20px; 
            font-size: 18px; 
            text-align: left; 
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        
        div.stButton > button {
            width: 100%;
            border-radius: 8px;
            border: 1px solid #ced4da;
            background-color: white;
            color: #495057;
            font-size: 18px !important;
            font-weight: 600;
            transition: all 0.2s;
        }
        div.stButton > button:hover {
            border-color: #4285F4; color: #4285F4; background-color: #e8f0fe; transform: translateY(-2px);
        }

        .ai-feedback-box { background-color: #e8f0fe; border-left: 5px solid #4285F4; padding: 15px; border-radius: 5px; color: #174ea6; margin-top: 20px;}
        .diff-box { background-color: #fff; border: 2px dashed #bdc3c7; padding: 15px; border-radius: 10px; font-size: 18px; }
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. 核心邏輯
# ==========================================
def split_text_into_sentences(text):
    text = text.replace('\n', ' ')
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 0]
    return sentences

def transcribe_audio(audio_path):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = r.record(source)
            try:
                text = r.recognize_google(audio_data, language="en-US")
                return text
            except: return ""
    except: return ""

def check_similarity_visual(target, user_text):
    if not user_text: return 0, "無語音輸入"
    t_words = re.findall(r"\w+", target.lower())
    u_words = re.findall(r"\w+", user_text.lower())
    matcher = difflib.SequenceMatcher(None, t_words, u_words)
    score = matcher.ratio() * 100
    html_parts = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        t_segment = " ".join(t_words[i1:i2])
        u_segment = " ".join(u_words[j1:j2])
        if tag == 'equal': html_parts.append(f'<span style="color:#198754; font-weight:bold;">{t_segment}</span>')
        elif tag == 'replace': html_parts.append(f'<span style="color:#dc3545; text-decoration:line-through;">{t_segment}</span> <span style="color:#6c757d;">({u_segment})</span>')
        elif tag == 'delete': html_parts.append(f'<span style="background-color:#f8d7da; color:#dc3545;">{t_segment}</span>')
        elif tag == 'insert': html_parts.append(f'<span style="color:#6c757d; font-style:italic;">{u_segment}</span>')
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
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(norm_t, label='Teacher', color='#4285F4', linewidth=2)
        ax.plot(norm_s_res, label='You', color='#fd7e14', linestyle='--', linewidth=2)
        ax.axis('off')
        plt.close(fig)
        score = max(0, np.corrcoef(norm_t, norm_s_res)[0, 1]) * 100
        
        seg = norm_s_res[int(len(norm_s_res)*0.7):]
        trend = 0
        if len(seg) > 1:
            diff = np.mean(seg[len(seg)//2:]) - np.mean(seg[:len(seg)//2])
            trend = 1 if diff > 0.2 else -1 if diff < -0.2 else 0
        return fig, score, trend
    except: return None, 0, 0

def get_ai_coach_feedback(api_key, target_text, user_text, pitch_trend, score):
    if not api_key: return "⚠️ 請輸入 API Key"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        trend_str = "上揚" if pitch_trend == 1 else "下降" if pitch_trend == -1 else "平淡"
        prompt = f"你是英文教練。目標：{target_text}。學生：{user_text}。語調：{trend_str}。分數：{score}。請給予簡短、幽默的繁體中文建議。"
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        if "429" in str(e): return "⚠️ AI 忙碌中 (流量限制)，請稍後再試。"
        return f"AI 錯誤: {str(e)}"

@st.cache_data(show_spinner=False)
def get_word_info(api_key, word, sentence):
    if not api_key: return "⚠️ 請輸入 API Key"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"""
        解釋單字 "{word}" 在句子 "{sentence}" 中的意思。
        格式：
        🔊 [{word}] KK音標
        🏷️ [詞性]
        💡 [繁體中文意思]
        (只要內容，不要廢話)
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg: return "⏳ 查詢太快了，請等 10 秒後再試。"
        return f"❌ 查詢失敗: {error_msg}"

# ==========================================
# 2. 發音引擎
# ==========================================

def speak_google_tts(text, speed=1.0, lang='en'):
    try:
        is_slow = speed < 1.0
        tts = gTTS(text=text, lang=lang, slow=is_slow)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name, "Google 線上語音"
    except Exception as e:
        print(f"Google TTS 失敗: {e}") 
        return None, None

def speak_offline_fallback(text, speed=1.0):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', int(175 * speed))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            engine.save_to_file(text, fp.name)
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

# Session 初始化
if 'game_active' not in st.session_state: st.session_state.game_active = False
if 'sentences' not in st.session_state: st.session_state.sentences = []
if 'current_index' not in st.session_state: st.session_state.current_index = 0
if 'current_word_definition' not in st.session_state: st.session_state.current_word_definition = None
if 'current_word_audio' not in st.session_state: st.session_state.current_word_audio = None # 新增單字聲音

KEY_FILE = "secret_key.txt"
if 'saved_api_key' not in st.session_state:
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "r") as f: st.session_state.saved_api_key = f.read().strip()
    else: st.session_state.saved_api_key = ""

with st.sidebar:
    st.header("⚙️ 設定")
    gemini_api_key = st.text_input("🔑 Google API Key", value=st.session_state.saved_api_key, type="password")
    if gemini_api_key != st.session_state.saved_api_key:
        with open(KEY_FILE, "w") as f: f.write(gemini_api_key)
        st.session_state.saved_api_key = gemini_api_key
    
    st.markdown("---")
    tts_mode = st.radio("發音模式", ["☁️ 線上 (AI)", "💻 離線 (Windows)"], index=0)
    
    selected_voice_id = None
    if "線上" in tts_mode:
        online_voices = {"🇺🇸 Jenny": "en-US-JennyNeural", "🇺🇸 Christopher": "en-US-ChristopherNeural", "🇬🇧 Sonia": "en-GB-SoniaNeural"}
        voice_choice = st.selectbox("選擇角色", list(online_voices.keys()))
        selected_voice_id = online_voices[voice_choice]
    else:
        offline_voices = get_offline_voices()
        if offline_voices:
            voice_choice = st.selectbox("選擇語音", list(offline_voices.keys()))
            selected_voice_id = offline_voices[voice_choice]
            
    voice_speed = st.slider("語速", 0.5, 1.5, 1.0, 0.1)

st.title("🎤 Voice Lab 英文跟讀教練")

# --- 輸入介面 ---
if not st.session_state.game_active:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    input_text = st.text_area("📝 請貼上文章：", value="Technology is changing how we live and work every single day.", height=150)
    if st.button("🚀 開始練習", type="primary", use_container_width=True):
        s = split_text_into_sentences(input_text)
        if s: 
            st.session_state.sentences = s
            st.session_state.current_index = 0
            st.session_state.game_active = True
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- 練習介面 ---
else:
    idx = st.session_state.current_index
    sentences = st.session_state.sentences
    target_sentence = sentences[idx]

    # === [新增功能] 導航列 (上一句/下一句) ===
    col_prev, col_prog, col_next = st.columns([1, 4, 1])
    
    with col_prev:
        if st.button("⬅️ 上一句", disabled=(idx == 0), use_container_width=True):
            st.session_state.current_index -= 1
            st.session_state.current_word_definition = None # 清除單字查詢
            st.session_state.current_word_audio = None
            st.rerun()
            
    with col_prog:
        st.progress((idx+1)/len(sentences), text=f"進度：{idx+1} / {len(sentences)}")
        
    with col_next:
        if st.button("下一句 ➡️", disabled=(idx == len(sentences)-1), use_container_width=True):
            st.session_state.current_index += 1
            st.session_state.current_word_definition = None
            st.session_state.current_word_audio = None
            st.rerun()

    col1, col2 = st.columns([1.5, 1], gap="large")

    # === 左邊：閱讀與查詢 ===
    with col1:
        st.subheader("📖 閱讀與查詢")
        st.markdown(f'<div class="reading-box">{target_sentence}</div>', unsafe_allow_html=True)
        
        st.caption("👇 點擊單字查看解釋與發音：")
        words = re.findall(r"\b\w+\b", target_sentence)
        cols = st.columns(5)
        for i, word in enumerate(words):
            if cols[i % 5].button(word, key=f"btn_{idx}_{i}"):
                if gemini_api_key:
                    with st.spinner("🔍..."):
                        # 1. 查義
                        info = get_word_info(gemini_api_key, word, target_sentence)
                        st.session_state.current_word_definition = f"**{word}**：\n{info}"
                        
                        # 2. 發音 (帶入語速參數)
                        w_path, _ = speak_google_tts(word, voice_speed)
                        if not w_path: 
                            w_path = speak_offline_fallback(word, voice_speed)
                        
                        st.session_state.current_word_audio = w_path
                else:
                    st.error("請輸入 Key")

        # 顯示查詢結果
        if st.session_state.current_word_definition:
            info_html = st.session_state.current_word_definition.replace('\n', '<br>')
            st.markdown(f"""
            <div class="definition-card">
                <div>{info_html}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.session_state.current_word_audio:
                st.audio(st.session_state.current_word_audio, format='audio/mp3')

        st.markdown("---")
        st.subheader("🗣️ 整句示範")
        
        # 整句發音 (帶入語速參數)
        path, engine_name = speak_google_tts(target_sentence, voice_speed)
        if not path: 
             path = speak_offline_fallback(target_sentence, voice_speed)
             engine_name = "離線語音"

        if path:
            st.audio(path, format='audio/mp3')
            st.caption(f"🔈 播放引擎: {engine_name} | 語速: {voice_speed}x")
        else:
            st.error("無法生成語音")

    # === 右邊：錄音與評分 ===
    with col2:
        st.subheader("🎙️ 口說挑戰")
        user_audio = st.audio_input("請按錄音鈕開始", key=f"rec_{idx}")
        
        if user_audio and path:
            with st.spinner("🤖 AI 分析中..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(user_audio.read())
                    user_path = tmp.name
                
                user_text = transcribe_audio(user_path)
                score_text, diff_html = check_similarity_visual(target_sentence, user_text)
                fig, score_pitch, trend = plot_and_get_trend(path, user_path)
                
                final_score = score_text
                if HAS_LIBROSA and fig:
                    final_score = (score_text * 0.6) + (score_pitch * 0.4)
                
                feedback = get_ai_coach_feedback(gemini_api_key, target_sentence, user_text, trend, final_score)

            if final_score >= 80:
                st.success(f"🎉 太棒了！分數：{final_score:.0f}")
            else:
                st.warning(f"💪 再加油：{final_score:.0f}")

            # === [新增功能] 回放自己的聲音 ===
            st.write("🎧 **回放你的錄音：**")
            st.audio(user_path, format='audio/wav')

            st.markdown(f"""
            <div style="background-color:#e8f0fe; padding:15px; border-radius:10px; border-left:5px solid #4285F4; margin-top:10px;">
                <strong>🤖 AI 教練建議：</strong><br>{feedback}
            </div>
            """, unsafe_allow_html=True)

            tab1, tab2 = st.tabs(["🔤 發音糾錯", "📈 語調分析"])
            with tab1:
                st.markdown(f'<div style="font-size:20px; padding:10px;">{diff_html}</div>', unsafe_allow_html=True)
            with tab2:
                if fig: st.pyplot(fig)
                else: st.info("無法分析語調")

            if final_score >= 80:
                st.markdown("---")
                if st.button("➡️ 下一句", type="primary", use_container_width=True):
                    st.session_state.current_index += 1; st.session_state.current_audio_path = None; st.session_state.current_word_definition = None; st.session_state.current_word_audio = None; st.rerun()