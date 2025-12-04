import streamlit as st

# 1. 設定頁面
try:
    st.set_page_config(page_title="AI 英文教練 Pro (重點版)", layout="wide", page_icon="🖍️")
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

# 3. 安全匯入離線發音
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
        
        /* 評論區塊 */
        .ai-feedback-box {
            background-color: #ffffff;
            border: 2px solid #e0e0e0;
            border-left: 8px solid #d32f2f; /* 改成紅色系，強調修正 */
            padding: 20px;
            border-radius: 10px;
            color: #212121;
            margin-top: 20px;
            font-size: 18px;
            line-height: 1.8; /* 行高加大，讓重點字更清楚 */
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        
        .score-card {
            background-color: #ffffff; padding: 15px; border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 15px;
            border: 1px solid #eee; text-align: center;
        }
        
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

# [核心] AI 直聽分析 (加入重點標示指令)
def analyze_audio_with_gemini(api_key, target_sentence, audio_path):
    if not api_key: return None, "請輸入 API Key"
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        with open(audio_path, "rb") as f:
            audio_data = f.read()
            
        prompt = f"""
        你是一位專業的英文口說教練。
        目標句子是："{target_sentence}"
        
        請「仔細聆聽」使用者的錄音，針對準確度、流暢度、語調評分 (0-100)。

        回傳格式：
        [SCORE_START]
        ACCURACY: (分數)
        FLUENCY: (分數)
        INTONATION: (分數)
        [SCORE_END]
        
        **🌟 綜合講評 (繁體中文)**：
        先給予肯定，再明確指出建議。
        
        【重要格式要求】：
        **若有唸錯的單字、需要加強的發音、或是關鍵建議，請務必使用 HTML 標籤標示為「紅色粗體+底線」。**
        範例格式： <strong style='color:#d32f2f; text-decoration:underline;'>word</strong>
        請大量使用這個格式來強調重點，讓學生一眼就能看到哪裡要改。
        """
        
        response = model.generate_content([
            prompt,
            {"mime_type": "audio/wav", "data": audio_data}
        ])
        
        return response.text, None
        
    except Exception as e:
        return None, f"AI 分析失敗: {str(e)}"

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
                    if key in scores:
                        scores[key] = int(re.search(r'\d+', val).group())
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
if 'current_audio_path' not in st.session_state: st.session_state.current_audio_path