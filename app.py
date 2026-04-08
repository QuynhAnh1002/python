import streamlit as st
import math
import re
import pandas as pd
from collections import defaultdict
from underthesea import word_tokenize

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Vietnamese Spam Detector", page_icon="🛡️")

# --- 1. TẢI STOP WORDS & DỮ LIỆU (Dùng cache để không load lại khi nhấn nút) ---
@st.cache_resource
def load_stopwords(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return {line.strip().lower() for line in f if line.strip()}
    except:
        return set()

STOP_WORDS = load_stopwords('vietnamese-stopwords.txt')

def tokenize(message):
    text = str(message).lower().strip()
    if not text:
        return set()
    try:
        segmented_text = word_tokenize(text, format="chose")
        if isinstance(segmented_text, list):
            segmented_text = " ".join(segmented_text)
    except:
        segmented_text = text
    all_words = re.findall(r"\w+", segmented_text, flags=re.UNICODE)
    return {word for word in all_words if word not in STOP_WORDS}

class VietnameseNaiveBayes:
    def __init__(self, k=0.5):
        self.k = k
        self.word_probs = []

    def train(self, df):
        df = df.dropna(subset=['labels', 'texts_vi'])
        num_spams = len(df[df['labels'] == 'spam'])
        num_non_spams = len(df) - num_spams
        word_counts = defaultdict(lambda: [0, 0])
        for _, row in df.iterrows():
            is_spam = (row['labels'] == 'spam')
            for word in tokenize(row['texts_vi']):
                word_counts[word][0 if is_spam else 1] += 1
        self.word_probs = [
            (w,
             (spam + self.k) / (num_spams + 2 * self.k),
             (non_spam + self.k) / (num_non_spams + 2 * self.k))
            for w, (spam, non_spam) in word_counts.items()
        ]

    def classify(self, message):
        message_words = tokenize(message)
        log_prob_if_spam = log_prob_if_not_spam = 0.0
        for word, p_spam, p_ham in self.word_probs:
            if word in message_words:
                log_prob_if_spam += math.log(p_spam)
                log_prob_if_not_spam += math.log(p_ham)
            else:
                log_prob_if_spam += math.log(1.0 - p_spam)
                log_prob_if_not_spam += math.log(1.0 - p_ham)
        max_log = max(log_prob_if_spam, log_prob_if_not_spam)
        prob_if_spam = math.exp(log_prob_if_spam - max_log)
        prob_if_not_spam = math.exp(log_prob_if_not_spam - max_log)
        return prob_if_spam / (prob_if_spam + prob_if_not_spam)

# --- 2. KHỞI TẠO MÔ HÌNH ---
@st.cache_resource
def get_trained_model():
    classifier = VietnameseNaiveBayes(k=0.5)
    try:
        df = pd.read_csv('datavn.csv', sep=';', encoding='utf-8-sig')
        classifier.train(df)
        return classifier
    except Exception as e:
        st.error(f"Lỗi nạp dữ liệu: {e}")
        return None

classifier = get_trained_model()

# --- 3. GIAO DIỆN NGƯỜI DÙNG (UI) ---
st.title("🛡️ Kiểm tra Tin nhắn Spam Tiếng Việt")
st.markdown("Hệ thống sử dụng thuật toán **Naive Bayes** và thư viện **Underthesea**.")

input_text = st.text_area("Nhập nội dung tin nhắn cần kiểm tra:", height=150, placeholder="Ví dụ: Chúc mừng bạn đã trúng thưởng...")

if st.button("Phân loại ngay"):
    if not input_text.strip():
        st.warning("Vui lòng nhập nội dung!")
    else:
        # 1. Kiểm tra Whitelist
        whitelist = ['.edu.vn', 'daotao', 'phòng qlđt', 'nhà trường', 'thông báo', 'phqldt']
        is_whitelisted = any(key in input_text.lower() for key in whitelist)
        
        if is_whitelisted:
            spam_prob = 0.0
            is_spam = False
        else:
            # 2. Chạy model
            spam_prob = classifier.classify(input_text)
            is_spam = spam_prob > 0.92

        # Hiển thị kết quả
        st.divider()
        if is_spam:
            st.error(f"### 🚩 KẾT QUẢ: TIN NHẮN SPAM")
        else:
            st.success(f"### ✅ KẾT QUẢ: TIN NHẮN AN TOÀN")
        
        st.metric("Xác suất Spam", f"{spam_prob*100:.2f}%")
        
        if is_whitelisted:
            st.info("Tin nhắn này nằm trong danh sách ưu tiên (Whitelist).")
