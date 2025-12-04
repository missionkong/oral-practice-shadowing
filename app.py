import streamlit as st

# 1. 設定頁面 (必須在第一行)
try:
    st.set_page_config(page_title="AI 英文教練 Pro (手機版)", layout="wide", page_icon="📱")
except:
    pass

import speech_recognition as sr
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

# 3. 安全匯入 (這裡才是重點！)
# 我們把 pyttsx3 的匯入包在 try...except 裡，這樣雲端沒安裝也不會當機
HAS_OFFLINE_TTS = False
try:
    import pyttsx3
    HAS_OFFLINE_TTS = True
except ImportError:
    HAS_OFFLINE_TTS = False

# 嘗試匯入 librosa
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
            background-color: #fff9c4; border: 2px solid #fbc02d; color: #5d4037; 
            padding: 15px; border-radius: 12px; margin-top: 15px; font-size: 18px; 
        }
        
        .mobile-hint-card {
            background-color: #e3f2fd;
            border: 1px solid #90caf9;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 10px;
            font-size: 16px;
            font-weight: 600;
            color: #1565c0;
            line-height: 1.4;
        }

        div.stButton > button { width: 100%; border-radius: 8px; height: 3em; font-weight: bold; }
        
        .ai-feedback-box { background-color: #f1f8e9; border-left: 5px solid #8bc34a; padding: 15px; border-radius: 10px; color: #33691e; margin-top: 20px;}
        .diff-box { background-color: #fff; border: 2px dashed #bdc3c7; padding: 15px; border-radius: 10px; font-size: 18px; }
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. 核心功能
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