import streamlit as st

# 1. 設定頁面 (絕對第一行)
try:
    st.set_page_config(page_title="AI 英文教練 Pro (最終修復)", layout="wide", page_icon="🎤")
except:
    pass

import speech_recognition as sr
from gtts import gTTS
import tempfile
import os
import re
import google.generativeai as genai
import ssl

# 2. 忽略 SSL 錯誤
ssl._create_default_https_context = ssl._create_unverified_context

# 3. 安全匯入離線發音 (防崩潰)
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

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

# ==========================================
# [關鍵修正] Session State 初始化 (確保變數存在)
# ==========================================
if 'game_active' not in st.session_state: st.session_state.game_active = False
if 'sentences' not in st.session_state: st.session_state.sentences = []
if 'current_index' not in st.session_state: st.session_state.current_index = 0
if 'current_audio_path' not in st.session_state: st.session_state.current_audio_path = None
if 'current_word_info' not in st.session_state: st.session_state.current_word_info = None
if 'current_word_audio' not in st.session_state: st.session_state.current_word_audio = None
if 'current_word_data' not in st.session_state: st.session_state.current_word_data = None 

# Key 管理
KEY_FILE = "secret_key.txt"
if 'saved_api_key' not in st.session_state:
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "r") as f: st.session_state.saved_api_key = f.read().strip()
    else: st.session_state.saved_api_key = ""

# ==========================================
# 0. UI 美化
# ==========================================
def inject_custom_css():
    st.markdown("""
        <style>
        .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); font-family: 'Microsoft JhengHei', sans-serif; }
        .reading-box { 
            font-size: 26px !important; font-weight: bold; color: #2c3e50; 
            line-height: 1.6; padding: 20px; background-color: #ffffff; 
            border-left: 8px solid #4285F4; border-radius: 10px; margin-bottom: 25px; 
        }
        .definition-card { 
            background-color: #fff9c4; border: 2px solid #fbc02d; color: #5d4037; 
            padding: 15px; border-radius: 12px; margin-top: 15px; font-size: 18px; 
        }
        .mobile-hint-card {
            background-color: #e3f2fd; border-left: 5px solid #2196f3;
            padding: 12px; border-radius: 8px; margin-bottom: 10px;
            font-size: 16px; font-weight: 600; color: #1565c0; line-height: 1.4;
        }
        div.stButton > button { width: 100%; border-radius: 8px; height: 3em; font-weight: bold; }
        .ai-feedback-box { 
            background-color: #ffffff; border: 2px solid #e0e0e0; border-left: 8px solid #d32f2f;
            padding: 20px; border-radius: 10px; color: #212121; margin-top: 20px;
            font-size: 18px; line-height: 1.8; box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        .score-card {
            background-color: #ffffff; padding: 15px; border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 15px;
            border: 1px solid #eee; text-align: center;
        }
        .score-title { font-size: 16px; color: #666; font-weight: bold; }
        .score-val { font-size: 24px; font-weight: bold; color: #2e7d32; }
        .diff-box { background-color: #fff; border: 2px dashed #bdc3c7; padding: 15px; border-radius: 10px; font-size: 18px; }
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. 核心邏輯
# ==========================================
def split_text_into_sentences(text):
    text = text.replace('\n', ' ')
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw_sentences if len(s.strip()) > 0]

def transcribe_audio(audio_path):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = r.record(source)
            return r.recognize_google(audio_data, language="en-US")
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

def analyze_audio_with_gemini(api_key, target_sentence, audio_path):
    if not api_key: return None, "請輸入 API Key"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        with open(audio_path, "rb") as f: audio_data = f.read()
        prompt = f"""
        你是一位專業英文口說教練。目標句子："{target_sentence}"
        請仔細聆聽並針對準確度、流暢度、語調評分(0-100)。
        
        回傳格式：
        [SCORE_START]
        ACCURACY: (分數)
        FLUENCY: (分數)
        INTONATION: (分數)
        [SCORE_END]
        
        **🌟 綜合講評 (繁體中文)**：
        給予肯定與具體建議。
        【重要】：唸錯的字或關鍵建議，請用 HTML 標籤 <strong style='color:#d32f2f; text-decoration:underline;'>標示為紅色粗體底線</strong>。
        """
        response = model.generate_content([prompt, {"mime_type": "audio/wav", "data": audio_data}])
        return response.text, None
    except Exception as e: return None, f"AI 分析失敗: {str(e)}"

def parse_scores(text):
    scores = {"ACCURACY": 0, "FLUENCY": 0, "INTONATION": 0}
    comment = text
    try:
        if "[SCORE_START]" in text and "[SCORE_END]" in text:
            parts = text.split("[SCORE_END]")
            block = text.split("[SCORE_START]")[1].split("[SCORE_END]")[0]
            comment = parts[1].strip()
            for line in block.strip().split('\n'):
                if ":" in line:
                    key, val = line.split(":")
                    key = key.strip().upper()
                    if key in scores: scores[key] = int(re.search(r'\d+', val).group())
    except: pass
    return scores, comment

@st.cache_data(show_spinner=False)
def get_word_info(api_key, word, sentence):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"解釋單字 '{word}' 在句子 '{sentence}' 中的意思。格式：🔊[{word}] KK音標\\n🏷️[詞性]\\n💡[繁中意思](簡潔)"
        response = model.generate_content(prompt)
        return response.text
    except: return "查詢失敗"

# ==========================================
# 2. 發音引擎 (修復 SyntaxError)
# ==========================================

def speak_google(text, speed=1.0):
    try:
        is_slow = speed < 1.0
        tts = gTTS(text=text, lang='en', slow=is_slow)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except Exception:
        return None

def speak_offline(text, speed=1.0):
    if not HAS_OFFLINE_TTS:
        return None
    
    # [關鍵修正] 這裡加上了完整的 try-except 結構，解決 SyntaxError
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', int(175 * speed))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            engine.save_to_file(text, fp.name)
            engine.runAndWait()
            return fp.name
    except Exception:
        return None

def get_offline_voices():
    if not HAS_OFFLINE_TTS: return {}
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        return {v.name: v.id for v in voices}
    except: return {}

# ==========================================
# 3. 主程式介面
# ==========================================
inject_custom_css()

# Sidebar
with st.sidebar:
    st.header("⚙️ 設定")
    gemini_api_key = st.text_input("🔑 Google API Key", value=st.session_state.saved_api_key, type="password")
    if gemini_api_key != st.session_state.saved_api_key:
        with open(KEY_FILE, "w") as f: f.write(gemini_api_key)
        st.session_state.saved_api_key = gemini_api_key
    
    st.markdown("---")
    if HAS_OFFLINE_TTS:
        tts_mode = st.radio("發音模式", ["☁️ 線上 (Google)", "💻 離線 (Windows)"], index=0)
    else:
        st.info("☁️ 雲端模式 (Google 發音)")
        tts_mode = "☁️ 線上 (Google)"
    
    voice_speed = st.slider("語速 (Google僅支援1.0/慢速)", 0.5, 1.5, 1.0, 0.1)

st.title("🎤 AI 英文教練 (Pro)")

# Input Area
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

# Practice Area
else:
    idx = st.session_state.current_index
    sentences = st.session_state.sentences
    target_sentence = sentences[idx]

    # Nav
    c1, c2, c3 = st.columns([1, 4, 1])
    with c1: 
        if st.button("⬅️ 上一句", disabled=(idx==0), use_container_width=True):
            st.session_state.current_index -= 1
            st.session_state.current_word_data = None
            st.session_state.current_audio_path = None
            st.rerun()
    with c2: st.progress((idx+1)/len(sentences), text=f"進度：{idx+1} / {len(sentences)}")
    with c3:
        if st.button("下一句 ➡️", disabled=(idx==len(sentences)-1), use_container_width=True):
            st.session_state.current_index += 1
            st.session_state.current_word_data = None
            st.session_state.current_audio_path = None
            st.rerun()

    col_L, col_R = st.columns([1.5, 1], gap="large")

    # Left: Text & Words
    with col_L:
        st.subheader("📖 閱讀與查詢")
        st.markdown(f'<div class="reading-box">{target_sentence}</div>', unsafe_allow_html=True)
        
        words = re.findall(r"\b\w+\b", target_sentence)
        cols = st.columns(5)
        for i, word in enumerate(words):
            if cols[i % 5].button(word, key=f"w_{idx}_{i}"):
                if gemini_api_key:
                    with st.spinner("🔍..."):
                        # 1. 查義
                        info = get_word_info(gemini_api_key, word, target_sentence)
                        info_html = info.replace('\n', '<br>')
                        
                        # 2. 發音
                        w_path = speak_google(word, 1.0)
                        if not w_path: w_path = speak_offline(word, 1.0)
                        
                        st.session_state.current_word_data = (info_html, w_path)
                else:
                    st.error("請輸入 Key")

        # 顯示單字查詢結果
        if st.session_state.current_word_data:
            info_html, w_path = st.session_state.current_word_data
            st.markdown(f'<div class="definition-card">{info_html}</div>', unsafe_allow_html=True)
            if w_path: st.audio(w_path, format='audio/mp3')

        st.markdown("---")
        st.subheader("🗣️ 整句示範")
        
        # 整句發音
        if st.session_state.current_audio_path is None:
            path = None
            if "線上" in tts_mode: path = speak_google(target_sentence, voice_speed)
            if not path: path = speak_offline(target_sentence, voice_speed)
            st.session_state.current_audio_path = path

        if st.session_state.current_audio_path:
            st.audio(st.session_state.current_audio_path, format="audio/mp3")
        else:
            st.warning("無法生成語音")

    # Right: Audio Analysis
    with col_R:
        st.subheader("🎙️ 口說挑戰")
        st.markdown(f'<div class="mobile-hint-card">📖 跟讀：<br>{target_sentence}</div>', unsafe_allow_html=True)
        
        user_audio = st.audio_input("開始錄音", key=f"rec_{idx}")
        
        if user_audio:
            with st.spinner("🧠 AI 正在聆聽分析..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(user_audio.read())
                    user_path = tmp.name
                
                # 直聽分析
                raw_response, error = analyze_audio_with_gemini(gemini_api_key, target_sentence, user_path)
                
                if error:
                    st.error(error)
                else:
                    scores, comment = parse_scores(raw_response)
                    
                    st.write("🎧 **回放您的錄音：**")
                    st.audio(user_path, format="audio/wav")
                    
                    s1, s2, s3 = st.columns(3)
                    s1.markdown(f"<div class='score-card'><div class='score-title'>準確度</div><div class='score-val'>{scores['ACCURACY']}</div></div>", unsafe_allow_html=True)
                    s2.markdown(f"<div class='score-card'><div class='score-title'>流暢度</div><div class='score-val'>{scores['FLUENCY']}</div></div>", unsafe_allow_html=True)
                    s3.markdown(f"<div class='score-card'><div class='score-title'>語調</div><div class='score-val'>{scores['INTONATION']}</div></div>", unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="ai-feedback-box">
                        <strong>🤖 AI 總評：</strong><br>{comment}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    avg_score = (scores['ACCURACY'] + scores['FLUENCY'] + scores['INTONATION']) / 3
                    if avg_score >= 80:
                        st.balloons()