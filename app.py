import streamlit as st
import pandas as pd
import re
from collections import defaultdict
import math
from underthesea import word_tokenize

# --- OPTIMIZED TOKENIZER ---
def clean_tokenize(text, stop_words):
    text = str(text).lower()
    # Tách từ bằng underthesea
    tokens = word_tokenize(text, format="chose")
    # Lọc nhiễu và từ dừng
    words = [w for w in tokens if re.match(r'^\w+$', w) and w not in stop_words]
    
    # TẠO BIGRAMS: Ghép cặp 2 từ liên tiếp để bắt được cụm "trúng_thưởng", "cho_vay"
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
    
    return words + bigrams # Kết hợp cả từ đơn và từ ghép

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
        log_prob_spam = math.log(self.class_counts['spam'] / sum(self.class_counts.values()))
        log_prob_ham = math.log(self.class_counts['ham'] / sum(self.class_counts.values()))

        n_spam = sum(self.word_counts['spam'].values())
        n_ham = sum(self.word_counts['ham'].values())
        v_size = len(self.vocab)

        for token in tokens:
            if token in self.vocab:
                # Laplace smoothing
                log_prob_spam += math.log((self.word_counts['spam'][token] + self.k) / (n_spam + self.k * v_size))
                log_prob_ham += math.log((self.word_counts['ham'][token] + self.k) / (n_ham + self.k * v_size))

        # Chuyển đổi sang xác suất %
        max_log = max(log_prob_spam, log_prob_ham)
        prob_spam = math.exp(log_prob_spam - max_log)
        prob_ham = math.exp(log_prob_ham - max_log)
        return prob_spam / (prob_spam + prob_ham)

# --- UI STREAMLIT ---
st.title("🛡️ Spam Detector Pro")

@st.cache_resource
def load_all():
    # Tải stopwords
    try:
        with open('vietnamese-stopwords.txt', 'r', encoding='utf-8') as f:
            stop_words = {line.strip() for line in f}
    except:
        stop_words = set()
    
    # Train model
    model = BetterNaiveBayes()
    df = pd.read_csv('datavn.csv', sep=';', encoding='utf-8-sig')
    model.train(df, stop_words)
    return model, stop_words

model, stop_words = load_all()

msg = st.text_area("Nhập tin nhắn:")
if st.button("Kiểm tra"):
    score = model.classify(msg, stop_words)
    if score > 0.8:
        st.error(f"Cảnh báo Spam: {score*100:.2f}%")
    else:
        st.success(f"Tin nhắn sạch: {score*100:.2f}%")
