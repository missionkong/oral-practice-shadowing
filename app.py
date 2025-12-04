import streamlit as st

# 1. 設定頁面
try:
    st.set_page_config(page_title="AI 英文教練 Pro (AI直聽版)", layout="wide", page_icon="🎧")
except:
    pass

from gtts import gTTS
import tempfile
import os
import re
import time
import google.generativeai as genai
import ssl

# 2. 忽略 SSL 錯誤
ssl._create_default_https_context = ssl._create_unverified_context

# 3. 安全匯入離線發音 (雲端防崩潰)
HAS_OFFLINE_TTS = False
try:
    import pyttsx3
    HAS_OFFLINE_TTS = True
except ImportError:
    HAS_OFFLINE_TTS = False

# ==========================================
# 0. UI 美化
# ==========================================
def inject_custom_css():
    st.markdown("""
        <style>
        .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); font-family: 'Microsoft JhengHei', sans-serif; }
        
        .reading-box { 
            font-size: 24px !important; font-weight: bold; color: #2c3e50; 
            line-height: 1.5; padding: 20px; background-color: #ffffff; 
            border-left: 8px solid #4285F4; border-radius: 10px; margin-bottom: 20px; 
        }
        
        .mobile-hint-card {
            background-color: #e3f2fd; border-left: 5px solid #2196f3;
            padding: 10px; border-radius: 8px; margin-bottom: 10px;
            font-size: 16px; font-weight: bold; color: #0d47a1;
        }

        .definition-card { 
            background-color: #fff9c4; border: 2px solid #fbc02d; color: #5d4037; 
            padding: 15px; border-radius: 12px; margin-top: 15px; font-size: 18px; 
        }
        
        /* 評分卡片樣式 */
        .score-card {
            background-color: #ffffff; padding: 15px; border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 15px;
            border: 1px solid #eee;
        }
        .score-title { font-size: 16px; color: #666; font-weight: bold; }
        .score-val { font-size: 24px; font-weight: bold; color: #2e7d32; }
        
        div.stButton > button { width: 100%; border-radius: 8px; height: 3em; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. 核心邏輯
# ==========================================
def split_text_into_sentences(text):
    text = text.replace('\n', ' ')
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw_sentences if len(s.strip()) > 0]

# [核心] AI 直聽分析
def analyze_audio_with_gemini(api_key, target_sentence, audio_path):
    if not api_key: return None, "請輸入 API Key"
    
    try:
        genai.configure(api_key=api_key)
        # 使用具備聽力能力的 Gemini 2.0 Flash
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # 讀取音訊檔
        with open(audio_path, "rb") as f:
            audio_data = f.read()
            
        prompt = f"""
        你是一位專業的英文口說教練。
        目標句子是："{target_sentence}"
        
        請「仔細聆聽」使用者的錄音，並針對以下三個維度進行評分與分析：
        1. **準確度 (Accuracy)**：發音是否正確？有無唸錯字？
        2. **流暢度 (Fluency)**：是否有不自然的停頓、結巴或遲疑？連音是否自然？
        3. **語調 (Intonation)**：抑揚頓挫是否自然？有沒有像機器人一樣平淡？

        請依照以下格式回傳結果 (請嚴格遵守格式)：
        
        [SCORE_START]
        ACCURACY: (0-100的數字)
        FLUENCY: (0-100的數字)
        INTONATION: (0-100的數字)
        [SCORE_END]
        
        **🌟 綜合講評 (繁體中文)**：
        先給予肯定，再明確指出哪裡不順暢、哪個字發音要修正，以及語調建議。
        """
        
        # 傳送音訊與提示詞 (Multimodal)
        response = model.generate_content([
            prompt,
            {"mime_type": "audio/wav", "data": audio_data}
        ])
        
        return response.text, None
        
    except Exception as e:
        return None, f"AI 分析失敗: {str(e)}"

def parse_scores(text):
    """從 AI 回傳文字中解析分數"""
    scores = {"ACCURACY": 0, "FLUENCY": 0, "INTONATION": 0}
    try:
        if "[SCORE_START]" in text and "[SCORE_END]" in text:
            block = text.split("[SCORE_START]")[1].split("[SCORE_END]")[0]
            for line in block.strip().split('\n'):
                if ":" in line:
                    key, val = line.split(":")
                    key = key.strip().upper()
                    if key in scores:
                        scores[key] = int(re.search(r'\d+', val).group())
            
            # 移除分數區塊，只留講評
            comment = text.split("[SCORE_END]")[1].strip()
            return scores, comment
    except:
        pass
    return scores, text # 解析失敗則回傳原文字

# 單字查詢
@st.cache_data(show_spinner=False)
def get_word_info(api_key, word, sentence):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"解釋單字 '{word}' 在句子 '{sentence}' 中的意思。格式：🔊[{word}] KK音標\\n🏷️[詞性]\\n💡[繁中意思](簡潔)"
        response = model.generate_content(prompt)
        return response.text
    except: return "查詢失敗"

# 發音引擎
def speak_google(text, speed=1.0):
    try:
        is_slow = speed < 1.0
        tts = gTTS(text=text, lang='en', slow=is_slow)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except: return None

def speak_offline(text, speed=1.0):
    if not HAS_OFFLINE_TTS: return None
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
# 2. 主程式
# ==========================================
inject_custom_css()

# Session
if 'game_active' not in st.session_state: st.session_state.game_active = False
if 'sentences' not in st.session_state: st.session_state.sentences = []
if 'current_index' not in st.session_state: st.session_state.current_index = 0
if 'current_word_info' not in st.session_state: st.session_state.current_word_info = None
if 'current_word_audio' not in st.session_state: st.session_state.current_word_audio = None
if 'current_audio_path' not in st.session_state: st.session_state.current_audio_path = None

KEY_FILE = "secret_key.txt"
if 'saved_api_key' not in st.session_state:
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "r") as f: st.session_state.saved_api_key = f.read().strip()
    else: st.session_state.saved_api_key = ""

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
            st.session_state.current_word_info = None
            st.session_state.current_word_audio = None
            st.session_state.current_audio_path = None
            st.rerun()
    with c2: st.progress((idx+1)/len(sentences), text=f"進度：{idx+1} / {len(sentences)}")
    with c3:
        if st.button("下一句 ➡️", disabled=(idx==len(sentences)-1), use_container_width=True):
            st.session_state.current_index += 1
            st.session_state.current_word_info = None
            st.session_state.current_word_audio = None
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
                        info = get_word_info(gemini_api_key, word, target_sentence)
                        st.session_state.current_word_info = f"**{word}**：\n{info}"
                        w_path = speak_google(word, 1.0)
                        if not w_path: w_path = speak_offline(word, 1.0)
                        st.session_state.current_word_audio = w_path
                else:
                    st.error("請輸入 Key")

        if st.session_state.current_word_info:
            info_html = st.session_state.current_word_info.replace('\n', '<br>')
            st.markdown(f'<div class="definition-card">{info_html}</div>', unsafe_allow_html=True)
            if st.session_state.current_word_audio:
                st.audio(st.session_state.current_word_audio, format='audio/mp3')

        st.markdown("---")
        st.subheader("🗣️ 整句示範")
        
        if st.session_state.current_audio_path is None:
            path = None
            if "線上" in tts_mode: path = speak_google(target_sentence, voice_speed)
            if not path: path = speak_offline(target_sentence, voice_speed)
            st.session_state.current_audio_path = path

        if st.session_state.current_audio_path:
            st.audio(st.session_state.current_audio_path, format="audio/mp3")
        else:
            st.warning("無法生成語音")

    # Right: Audio Analysis (The New Core)
    with col_R:
        st.subheader("🎙️ 口說挑戰")
        st.markdown(f'<div class="mobile-hint-card">📖 跟讀：<br>{target_sentence}</div>', unsafe_allow_html=True)
        
        user_audio = st.audio_input("開始錄音", key=f"rec_{idx}")
        
        if user_audio:
            with st.spinner("🧠 AI 正在聆聽分析..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(user_audio.read())
                    user_path = tmp.name
                
                # 直接送給 Gemini 聽！
                raw_response, error = analyze_audio_with_gemini(gemini_api_key, target_sentence, user_path)
                
                if error:
                    st.error(error)
                else:
                    # 解析分數與評語
                    scores, comment = parse_scores(raw_response)
                    
                    # 顯示回放
                    st.write("🎧 **回放您的錄音：**")
                    st.audio(user_path, format="audio/wav")
                    
                    # 顯示三維度評分
                    s1, s2, s3 = st.columns(3)
                    s1.metric("準確度 Accuracy", f"{scores['ACCURACY']}", help="發音是否正確？有無唸錯字？")
                    s2.metric("流暢度 Fluency", f"{scores['FLUENCY']}", help="停頓是否自然？有無結巴？")
                    s3.metric("語調 Intonation", f"{scores['INTONATION']}", help="抑揚頓挫是否像真人？")
                    
                    # 顯示總評
                    st.markdown(f"""
                    <div style="background-color:#e8f0fe; padding:20px; border-radius:10px; border-left:5px solid #4285F4; margin-top:20px;">
                        <strong>🤖 AI 總評：</strong><br>{comment}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 鼓勵機制
                    avg_score = (scores['ACCURACY'] + scores['FLUENCY'] + scores['INTONATION']) / 3
                    if avg_score >= 80:
                        st.balloons()