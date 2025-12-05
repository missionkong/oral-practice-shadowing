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

# [核心] 使用 Google Generative AI
import google.generativeai as genai

# 1. 設定頁面
try:
    st.set_page_config(page_title="AI 英文教練 Pro (最終UI版)", layout="wide", page_icon="🎓")
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

# ==========================================
# 1. UI 美化 (手機版色彩對比極致優化)
# ==========================================
def inject_custom_css():
    st.markdown("""
        <style>
        /* --- 全局背景 --- */
        .stApp { 
            background: linear-gradient(135deg, #fdfbf7 0%, #ebedee 100%); 
            font-family: 'Microsoft JhengHei', sans-serif; 
        }
        
        /* --- 主畫面 (Main Area) 文字顏色設定 --- */
        /* 強制所有標題、段落、文字為深黑色 */
        .main .block-container h1, 
        .main .block-container h2, 
        .main .block-container h3, 
        .main .block-container h4, 
        .main .block-container p, 
        .main .block-container div,
        .main .block-container span,
        .main .block-container label,
        .main .block-container li {
            color: #000000 !important;
        }

        /* --- 側邊欄 (Sidebar) 設定 --- */
        [data-testid="stSidebar"] {
            background-color: #263238 !important; /* 深藍灰色背景 */
        }
        
        /* 強制側邊欄內的所有文字為白色 */
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
        
        /* 側邊欄的輸入框 (Input) 內部文字維持深色 (因為輸入框背景通常是白) */
        [data-testid="stSidebar"] input {
            color: #000000 !important;
        }

        /* --- 閱讀區塊樣式 --- */
        .reading-box { 
            font-size: 26px !important; 
            font-weight: bold; 
            color: #000000 !important; /* 純黑 */
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
        
        /* --- 單字卡片 --- */
        .definition-card { 
            background-color: #fff9c4 !important; 
            border: 2px solid #fbc02d; 
            color: #3e2723 !important; 
            padding: 15px; 
            border-radius: 12px; 
            margin-top: 15px; 
            font-size: 18px; 
        }
        
        /* --- 提示卡 --- */
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
        
        /* --- 測驗區塊 --- */
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
        
        /* --- 提示與排行榜 --- */
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
        
        /* --- AI 回饋 --- */
        .ai-feedback-box { 
            background-color: #f1f8e9 !important; 
            border-left: 5px solid #8bc34a; 
            padding: 15px; 
            border-radius: 10px; 
            color: #33691e !important; 
            margin-top: 20px;
        }
        
        /* --- 按鈕 --- */
        div.stButton > button { 
            width: 100%; 
            border-radius: 8px; 
            height: 3em; 
            font-weight: bold; 
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
        f0_t, _, _ = librosa.pyin(y_t, fmin=50, fmax=400, frame_length=2048