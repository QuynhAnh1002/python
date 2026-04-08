import streamlit as st
import pandas as pd
import re
import math
from collections import defaultdict
from underthesea import word_tokenize

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(page_title="VN Spam Detector Pro", layout="wide")

# --- TIỀN XỬ LÝ & TOKENIZE ---
def clean_tokenize(text, stop_words):
    text = str(text).lower()
    # Tách từ ghép tiếng Việt [cite: 106-108]
    tokens = word_tokenize(text, format="chose")
    # Lọc nhiễu và từ dừng [cite: 98, 104-105]
    words = [w for w in tokens if re.match(r'^\w+$', w) and w not in stop_words]
    # Tạo Bigrams để bắt ngữ cảnh cụm từ
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
    return words + bigrams

class BetterNaiveBayes:
    def __init__(self, k=1.0):
        self.k = k
        self.vocab = set()
        self.word_counts = {'spam': defaultdict(int), 'ham': defaultdict(int)}
        self.class_counts = {'spam': 0, 'ham': 0}

    def train(self, df, stop_words):
        for _, row in df.iterrows():
            label = row['labels']
            tokens = clean_tokenize(row['texts_vi'], stop_words)
            self.class_counts[label] += 1
            for token in tokens:
                self.word_counts[label][token] += 1
                self.vocab.add(token)

    def classify(self, text, stop_words):
        tokens = clean_tokenize(text, stop_words)
        # Sử dụng Log-space để tránh Underflow [cite: 33-36]
        log_p_s = math.log(self.class_counts['spam'] / sum(self.class_counts.values()))
        log_p_h = math.log(self.class_counts['ham'] / sum(self.class_counts.values()))

        n_s = sum(self.word_counts['spam'].values())
        n_h = sum(self.word_counts['ham'].values())
        v_size = len(self.vocab)

        for token in tokens:
            if token in self.vocab:
                # Laplace Smoothing công thức [cite: 128]
                log_p_s += math.log((self.word_counts['spam'][token] + self.k) / (n_s + self.k * v_size))
                log_p_h += math.log((self.word_counts['ham'][token] + self.k) / (n_h + self.k * v_size))

        max_log = max(log_p_s, log_p_h)
        prob_spam = math.exp(log_p_s - max_log)
        prob_ham = math.exp(log_p_h - max_log)
        return prob_spam / (prob_spam + prob_ham)

# --- LOAD DATA & CACHING ---
@st.cache_resource
def load_all():
    try:
        with open('vietnamese-stopwords.txt', 'r', encoding='utf-8') as f:
            stop_words = {line.strip() for line in f}
        model = BetterNaiveBayes()
        # Nạp dataset (Dataset 1: Mixed hoặc Dataset 3: Diacritics) [cite: 169-171]
        df = pd.read_csv('datavn.csv', sep=';', encoding='utf-8-sig')
        model.train(df, stop_words)
        return model, stop_words, df
    except Exception as e:
        st.error(f"Lỗi hệ thống: {e}")
        return None, None, None

model, stop_words, df_train = load_all()

# --- GIAO DIỆN CHÍNH ---
st.title("🛡️ Phân tích & Phát hiện Tin nhắn Rác")

with st.sidebar:
    st.header("📊 Thông số Model")
    if df_train is not None:
        st.write(f"- Tổng mẫu huấn luyện: **{len(df_train)}**")
        st.write(f"- Kích thước từ vựng: **{len(model.vocab)}**")
        st.info("Độ chính xác tham chiếu (NB): **81.23%** ")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Kiểm tra tin nhắn")
    msg = st.text_area("Nhập nội dung:", height=200, placeholder="Ví dụ: Chuc mung ban da trung thuong...")
    run_btn = st.button("🚀 Phân tích ngay")

if run_btn and msg:
    score = model.classify(msg, stop_words)
    
    with col2:
        st.subheader("📈 Kết quả phân tích")
        if score > 0.8:
            st.error(f"### KẾT QUẢ: SPAM ({score*100:.2f}%)")
        else:
            st.success(f"### KẾT QUẢ: AN TOÀN ({(1-score)*100:.2f}%)")

        # --- PHẦN VISUALIZATION: ĐÓNG GÓP CỦA TỪ KHÓA ---
        tokens = clean_tokenize(msg, stop_words)
        impact_data = []
        n_s = sum(model.word_counts['spam'].values())
        n_h = sum(model.word_counts['ham'].values())
        v_size = len(model.vocab)

        for t in set(tokens): # Dùng set để không lặp lại từ trên biểu đồ
            if t in model.vocab:
                p_s = (model.word_counts['spam'][t] + model.k) / (n_s + model.k * v_size)
                p_h = (model.word_counts['ham'][t] + model.k) / (n_h + model.k * v_size)
                # Tính tỉ lệ đóng góp (Likelihood Ratio)
                impact_data.append({"Từ khóa": t, "Trọng số Spam": p_s / p_h})
        
        if impact_data:
            df_impact = pd.DataFrame(impact_data).sort_values(by="Trọng số Spam", ascending=False).head(10)
            st.write("**Top 10 từ khóa quyết định nhãn Spam:**")
            st.bar_chart(df_impact.set_index("Từ khóa"))
        else:
            st.info("Không tìm thấy từ khóa quen thuộc trong từ điển.")
