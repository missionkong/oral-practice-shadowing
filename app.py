import streamlit as st
import json
import random
import os
import difflib
import re
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import speech_recognition as sr
from gtts import gTTS
import ssl

# [核心修改] 改用 Vertex AI
import vertexai
from vertexai.preview.generative_models import GenerativeModel

# 1. 設定頁面
try:
    st.set_page_config(page_title="AI 英文教練 Pro (Vertex 本機驗證版)", layout="wide", page_icon="🎓")
except:
    pass

# 2. 忽略 SSL 錯誤
ssl._create_default_https_context = ssl._create_unverified_context

# 3. 安全匯入
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
# 0. 資料存取邏輯
# ==========================================
VOCAB_FILE = "vocab_book.json"
PROJECT_FILE = "google_project_id.txt"

def load_vocab():
    if not os.path.exists(VOCAB_FILE): return []
    try:
        with open(VOCAB_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except: return []

def save_vocab_to_disk(vocab_list):
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(vocab_list, f, ensure_ascii=False, indent=4)

def add_word_to_vocab(word, info):
    if not word or "查詢失敗" in info or "請檢查" in info: return False
    vocab_list = load_vocab()
    for v in vocab_list:
        if v["word"] == word: return False
    vocab_list.append({"word": word, "info": info})
    save_vocab_to_disk(vocab_list)
    return True

# ==========================================
# [核心修改] Vertex AI 初始化 (關鍵：讀取上傳的檔案)
# ==========================================
def init_vertex_ai(project_id, cred_file):
    # 確保有專案 ID 和憑證檔案物件
    if not project_id or cred_file is None: return None
    try:
        # 1. 創造一個暫時的檔案來存 json 內容
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w+', encoding='utf-8') as tmp:
            # 把上傳檔案的內容寫進去
            # cred_file 是一個 BytesIO 物件，要 decode 成字串
            content = cred_file.getvalue().decode("utf-8")
            tmp.write(content)
            tmp_cred_path = tmp.name
        
        # 2. 告訴 Google SDK 暫時的憑證檔案在哪裡
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp_cred_path
        
        # 3. 初始化 Vertex AI
        vertexai.init(project=project_id, location="us-central1")
        return True
    except Exception as e:
        print(f"Vertex AI Init Error: {e}")
        return None

# ==========================================
# 1. UI 美化
# ==========================================
def inject_custom_css():
    st.markdown("""
        <style>
        .stApp { background: linear-gradient(135deg, #fdfbf7 0%, #ebedee 100%); font-family: 'Microsoft JhengHei', sans-serif; }
        .reading-box { font-size: 26px !important; font-weight: bold; color: #2c3e50; line-height: 1.6; padding: 20px; background-color: #ffffff; border-left: 8px solid #4285F4; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .definition-card { background-color: #fff9c4; border: 2px solid #fbc02d; color: #5d4037; padding: 15px; border-radius: 12px; margin-top: 15px; font-size: 18px; }
        .mobile-hint-card { background-color: #e3f2fd; border-left: 5px solid #2196f3; padding: 10px; border-radius: 8px; margin-bottom: 10px; font-size: 16px; font-weight: 600; color: #0d47a1; }
        .quiz-box { background-color: #ffffff; border: 2px solid #4caf50; padding: 20px; border-radius: 15px; margin-top: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
        .backup-alert { background-color: #e8f5e9; border: 2px solid #66bb6a; padding: 20px; border-radius: 15px; text-align: center; margin-top: 20px; margin-bottom: 20px; }
        div.stButton > button { width: 100%; border-radius: 8px; height: 3em; font-weight: bold; }
        .ai-feedback-box { background-color: #f1f8e9; border-left: 5px solid #8bc34a; padding: 15px; border-radius: 10px; color: #33691e; margin-top: 20px;}
        .diff-box { background-color: #fff; border: 2px dashed #bdc3c7; padding: 15px; border-radius: 10px; font-size: 18px; }
        </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. 核心功能
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

# [修改] 使用 Vertex AI 的回饋函式 (傳入 cred_file)
def get_ai_coach_feedback(project_id, cred_file, target_text, user_text, score):
    if not init_vertex_ai(project_id, cred_file): return "⚠️ 請檢查 Project ID 並一定要上傳憑證 JSON 檔案"
    try:
        # 使用最新的 Gemini Pro 模型
        model = GenerativeModel("gemini-1.5-pro-preview-0409")
        prompt = f"""
        你是一位溫暖的英文老師。
        目標句子："{target_text}"
        學生唸出："{user_text}"
        分數：{score:.0f}
        請給予繁體中文回饋：
        1. 🌟 亮點讚賞
        2. 🔧 具體發音糾正 (指出哪個字唸錯)
        3. 💪 暖心鼓勵
        """
        responses = model.generate_content(prompt, stream=False)
        return responses.text
    except Exception as e:
        return f"AI 錯誤: {str(e)}"

# [修改] 使用 Vertex AI 的單字查詢函式 (傳入 cred_file，並加入快取清除邏輯)
def get_word_info(_cred_file, project_id, word, sentence):
    if not init_vertex_ai(project_id, _cred_file): return "⚠️ 請檢查 Project ID 並一定要上傳憑證 JSON 檔案"
    try:
        # 使用最新的 Gemini Pro 模型
        model = GenerativeModel("gemini-1.5-pro-preview-0409")
        prompt = f"解釋單字 '{word}' 在句子 '{sentence}' 中的意思。格式：🔊[{word}] KK音標\\n🏷️[詞性]\\n💡[繁中意思](簡潔)"
        responses = model.generate_content(prompt, stream=False)
        return responses.text
    except Exception as e:
        print(f"Vertex AI Query Failed: {e}")
        return f"❌ 查詢失敗: {str(e)}"

# [修改] 使用 Vertex AI 的出題函式 (傳入 cred_file)
def generate_quiz(project_id, cred_file, word):
    if not init_vertex_ai(project_id, cred_file): return None
    try:
        model = GenerativeModel("gemini-1.5-pro-preview-0409")
        prompt = f"""
        請針對單字 "{word}" 出一個「句子填空題」。
        格式要求：
        Q: [英文句子，將 {word} 挖空變成 ______ ]
        A: [繁體中文翻譯]
        """
        responses = model.generate_content(prompt, stream=False)
        return responses.text
    except: return None

# 發音邏輯
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
# 3. 主程式介面
# ==========================================
inject_custom_css()

# Session 初始化
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

# 讀取 Project ID
if 'saved_project_id' not in st.session_state:
    if os.path.exists(PROJECT_FILE):
        with open(PROJECT_FILE, "r") as f: st.session_state.saved_project_id = f.read().strip()
    else: st.session_state.saved_project_id = ""

# --- 側邊欄 ---
with st.sidebar:
    st.title("⚙️ 設定 (Vertex 本機版)")
    
    # Project ID 輸入框
    google_project_id = st.text_input("Google Project ID", value=st.session_state.saved_project_id)
    if google_project_id != st.session_state.saved_project_id:
        with open(PROJECT_FILE, "w") as f: f.write(google_project_id)
        st.session_state.saved_project_id = google_project_id

    # [關鍵] 憑證檔案上傳 (一定要有這個檔案)
    st.markdown("### 🔐 驗證方式 (必填)")
    st.info("請上傳您的 Google Cloud 服務帳戶 JSON 憑證檔。")
    uploaded_creds = st.file_uploader("📄 上傳 Google 憑證 (JSON)", type=["json"], key="vertex_creds")

    if not google_project_id or not uploaded_creds:
        st.error("⛔ 請輸入 Project ID 並上傳憑證檔案，否則無法查詢！")
    else:
        st.success("✅ 憑證已載入，可以開始使用！")
    
    st.markdown("---")
    app_mode = st.radio("選擇模式", ["📖 跟讀練習", "📝 單字測驗"], index=0)
    st.markdown("---")
    
    with st.expander("💾 資料備份與還原", expanded=True):
        vocab_list = load_vocab()
        if vocab_list:
            json_str = json.dumps(vocab_list, ensure_ascii=False, indent=4)
            st.download_button("📥 下載單字本", json_str, "my_vocab.json", "application/json")
        uploaded_file = st.file_uploader("📤 上傳還原", type=["json"])
        if uploaded_file:
            try:
                data = json.load(uploaded_file)
                save_vocab_to_disk(data)
                st.success("還原成功！")
            except: pass

    st.markdown("---")
    if HAS_OFFLINE_TTS:
        tts_mode = st.radio("發音引擎", ["☁️ 線上 (Google)", "💻 離線 (Windows)"], index=0)
    else:
        tts_mode = "☁️ 線上 (Google)"
    voice_speed = st.slider("語速", 0.5, 1.5, 1.0, 0.1)

st.title("🎤 AI 英文教練 (付費升級版)")

# ==========================================
# 模式 A: 跟讀練習
# ==========================================
if app_mode == "📖 跟讀練習":
    if not st.session_state.game_active:
        st.markdown('<div class="reading-box">歡迎！請輸入文章開始練習。</div>', unsafe_allow_html=True)
        input_text = st.text_area("文章內容：", value="Technology is changing how we live and work every single day.", height=150)
        if st.button("🚀 開始練習", type="primary", use_container_width=True):
            s = split_text_into_sentences(input_text)
            if s: 
                st.session_state.sentences = s
                st.session_state.current_index = 0
                st.session_state.game_active = True
                st.session_state.is_finished = False
                st.rerun()
    else:
        if st.session_state.is_finished:
            st.balloons()
            st.markdown("""
            <div class="backup-alert">
                <h2>🎉 練習結束！</h2>
                <p>請點擊下方按鈕下載您的單字本備份。</p>
            </div>
            """, unsafe_allow_html=True)
            vocab_list = load_vocab()
            if vocab_list:
                json_str = json.dumps(vocab_list, ensure_ascii=False, indent=4)
                st.download_button(
                    label="📥 點我下載單字本 (Backup)",
                    data=json_str,
                    file_name="vocab_book_backup.json",
                    mime="application/json",
                    type="primary",
                    use_container_width=True
                )
            else:
                st.info("這次沒有收藏新單字。")
            if st.button("🔄 再練一次 / 回到首頁"):
                st.session_state.game_active = False
                st.session_state.is_finished = False
                st.rerun()
            st.stop()

        idx = st.session_state.current_index
        sentences = st.session_state.sentences
        target_sentence = sentences[idx]

        c1, c2, c3 = st.columns([1, 4, 1])
        with c1: 
            if st.button("⬅️ 上句", disabled=(idx==0), use_container_width=True):
                st.session_state.current_index -= 1
                st.session_state.current_audio_path = None
                st.rerun()
        with c2: st.progress((idx+1)/len(sentences), text=f"進度：{idx+1} / {len(sentences)}")
        with c3:
            is_last = (idx == len(sentences) - 1)
            btn_text = "完成 🎉" if is_last else "下句 ➡️"
            if st.button(btn_text, use_container_width=True):
                if is_last:
                    st.session_state.is_finished = True
                    st.rerun()
                else:
                    st.session_state.current_index += 1
                    st.session_state.current_audio_path = None
                    st.rerun()

        if st.button("🏁 中途結束並備份", type="secondary", use_container_width=True):
            st.session_state.is_finished = True
            st.rerun()

        col_L, col_R = st.columns([1.5, 1], gap="large")

        with col_L:
            st.subheader("📖 閱讀")
            st.markdown(f'<div class="reading-box">{target_sentence}</div>', unsafe_allow_html=True)
            
            st.caption("👇 點擊查單字 (使用 Google Vertex AI)：")
            words = re.findall(r"\b\w+\b", target_sentence)
            cols = st.columns(5)
            for i, word in enumerate(words):
                # [修改] 只有在有上傳憑證時才允許點擊
                if cols[i % 5].button(word, key=f"w_{idx}_{i}", disabled=not uploaded_creds):
                    st.session_state.current_word_target = word
                    with st.spinner("🔍 Vertex AI 查詢中..."):
                        # [修改] 傳入 uploaded_creds
                        info = get_word_info(uploaded_creds, google_project_id, word, target_sentence)
                        st.session_state.current_word_info = info
                        
                        if "查詢失敗" not in info and "請檢查" not in info:
                            w_path = speak_google(word, 1.0)
                            if not w_path: w_path = speak_offline(word, 1.0)
                            st.session_state.current_word_audio = w_path
                        else:
                            st.session_state.current_word_audio = None
            
            if not uploaded_creds:
                 st.warning("請先在側邊欄上傳憑證檔案，才能使用單字查詢功能。")

            if st.session_state.current_word_info:
                info_html = st.session_state.current_word_info.replace('\n', '<br>')
                st.markdown(f'<div class="definition-card">{info_html}</div>', unsafe_allow_html=True)
                
                c_p, c_s = st.columns([4, 1])
                with c_p:
                    if st.session_state.current_word_audio:
                        st.audio(st.session_state.current_word_audio, format='audio/mp3')
                with c_s:
                    if "查詢失敗" not in st.session_state.current_word_info and "請檢查" not in st.session_state.current_word_info:
                        if st.button("⭐ 收藏", use_container_width=True):
                            saved = add_word_to_vocab(st.session_state.current_word_target, st.session_state.current_word_info)
                            if saved: st.toast("✅ 已收藏")
                            else: st.toast("⚠️ 已存在")

            st.markdown("---")
            st.subheader("🗣️ 示範")
            if st.session_state.current_audio_path is None:
                path = None
                if "線上" in tts_mode: path = speak_google(target_sentence, voice_speed)
                if not path: path = speak_offline(target_sentence, voice_speed)
                st.session_state.current_audio_path = path

            if st.session_state.current_audio_path:
                st.audio(st.session_state.current_audio_path, format="audio/mp3")
            else:
                st.warning("無法生成語音")

        with col_R:
            st.subheader("🎙️ 口說")
            st.markdown(f'<div class="mobile-hint-card">📖 跟讀：<br>{target_sentence}</div>', unsafe_allow_html=True)
            # [修改] 只有在有上傳憑證時才允許錄音
            user_audio = st.audio_input("錄音", key=f"rec_{idx}", disabled=not uploaded_creds)
            if not uploaded_creds:
                 st.warning("請先上傳憑證檔案，才能使用口說評分功能。")
            
            if user_audio and st.session_state.current_audio_path and uploaded_creds:
                with st.spinner("🤖 Vertex AI 分析中..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(user_audio.read()); user_path = tmp.name
                    
                    u_text = transcribe_audio(user_path)
                    score_text, diff_html = check_similarity_visual(target_sentence, u_text)
                    fig, raw_pitch_score, _ = plot_and_get_trend(st.session_state.current_audio_path, user_path)
                    
                    adj_pitch = max(60, raw_pitch_score)
                    final_score = (score_text * 0.8) + (adj_pitch * 0.2)
                    # [修改] 傳入 uploaded_creds
                    feedback = get_ai_coach_feedback(google_project_id, uploaded_creds, target_sentence, u_text, final_score)

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
# 模式 B: 單字測驗
# ==========================================
elif app_mode == "📝 單字測驗":
    vocab_list = load_vocab()
    st.subheader("📝 單字本隨堂考")
    
    if not vocab_list:
        st.info("📭 目前單字本是空的。請先去「跟讀練習」查詢單字並按「⭐ 收藏」。")
    else:
        st.write(f"📚 累積單字：**{len(vocab_list)}** 個")
        # [修改] 只有在有上傳憑證時才允許出題
        if st.button("🎲 隨機出一題 (Vertex AI)", type="primary", use_container_width=True, disabled=not uploaded_creds):
            target = random.choice(vocab_list)
            word = target["word"]
            info = target["info"]

            with st.spinner(f"正在為 '{word}' 出題..."):
                # [修改] 傳入 uploaded_creds
                q_text = generate_quiz(google_project_id, uploaded_creds, word)
                if q_text and "失敗" not in q_text:
                    st.session_state.quiz_data = {"word": word, "content": q_text, "original_info": info}
                    st.session_state.quiz_answer_show = False
                else:
                    st.error("出題失敗 (請檢查 Project ID 和憑證檔案)")
        
        if not uploaded_creds:
             st.warning("請先上傳憑證檔案，才能使用測驗功能。")

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
                st.caption("📜 原始單字卡資料：")
                original_html = data['original_info'].replace('\n', '<br>')
                st.markdown(f'<div style="background-color:#fff9c4; padding:10px; border-radius:8px;">{original_html}</div>', unsafe_allow_html=True)

                w_path = speak_google(data['word'])
                if w_path: st.audio(w_path, format='audio/mp3')