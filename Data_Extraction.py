import os
import re
import requests
import pandas as pd
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import cmudict
import streamlit as st

nltk.download('punkt')
nltk.download('cmudict')


pronoun_pattern = r"\b(I|we|my|ours|us)\b"
syllable_dict = cmudict.dict()

# Load stopwords
def load_word_list(files):
    word_set = set()
    for file in files:
        content = file.read().decode("latin1")
        for line in content.splitlines():
            word = line.strip().split('|')[0].strip().lower()
            if word:
                word_set.add(word)
    return word_set

# Extract article content
def extract_article(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.content, "html.parser")
        title = soup.find(['h1', 'title'])
        paragraphs = soup.find_all('p')
        title_text = title.get_text(strip=True) if title else ""
        para_text = " ".join(p.get_text(strip=True) for p in paragraphs)
        return f"{title_text}\n\n{para_text}".strip()
    except Exception as e:
        return None

# Tokenization and cleaning
def clean_and_tokenize(text, stopwords):
    words = word_tokenize(text.lower())
    words = [re.sub(r'[^\w\s]', '', w) for w in words]  # Remove punctuation
    return [w for w in words if w and w not in stopwords]

# Count syllables
def count_syllables(word):
    word = word.lower()
    if word in syllable_dict:
        return max([len([y for y in x if y[-1].isdigit()]) for x in syllable_dict[word]])
    else:
        count = len(re.findall(r'[aeiouy]+', word))
        if word.endswith(("es", "ed")):
            count -= 1
        return max(1, count)

# Analyze text
def analyze_text(text, stopwords, pos_words, neg_words):
    tokens = clean_and_tokenize(text, stopwords)
    total_words = len(tokens)
    total_chars = sum(len(word) for word in tokens)
    positive_score = sum(1 for w in tokens if w in pos_words)
    negative_score = sum(1 for w in tokens if w in neg_words)
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 1e-6)
    subjectivity_score = (positive_score + negative_score) / (total_words + 1e-6)

    sentences = sent_tokenize(text)
    sentence_count = len(sentences)
    avg_sentence_length = total_words / sentence_count if sentence_count else 0

    complex_words = [w for w in tokens if count_syllables(w) > 2]
    complex_word_count = len(complex_words)
    complex_word_pct = complex_word_count / total_words if total_words else 0
    fog_index = 0.4 * (avg_sentence_length + complex_word_pct)

    avg_word_length = total_chars / total_words if total_words else 0
    syllables_per_word = sum(count_syllables(w) for w in tokens) / total_words if total_words else 0
    personal_pronouns = re.findall(pronoun_pattern, text, re.I)
    personal_pronoun_count = len(personal_pronouns)

    return {
        "Positive Score": positive_score,
        "Negative Score": negative_score,
        "Polarity Score": round(polarity_score, 4),
        "Subjectivity Score": round(subjectivity_score, 4),
        "Average Sentence Length": round(avg_sentence_length, 2),
        "Complex Word Count": complex_word_count,
        "Percentage of Complex Words": round(complex_word_pct * 100, 2),
        "Fog Index": round(fog_index, 2),
        "Average Words per Sentence": round(avg_sentence_length, 2),
        "Word Count": total_words,
        "Syllables per Word": round(syllables_per_word, 2),
        "Personal Pronouns": personal_pronoun_count,
        "Average Word Length": round(avg_word_length, 2)
    }

# ---------------- Streamlit UI ----------------

st.title("🧠 Article Sentiment & Readability Analyzer")

input_file = st.file_uploader("📥 Upload Excel File (with URL & URL_ID)", type=["xlsx"])
stopword_files = st.file_uploader("🚫 Upload Stopword File(s)", type=["txt"], accept_multiple_files=True)
pos_file = st.file_uploader("😊 Upload Positive Words File", type=["txt"])
neg_file = st.file_uploader("☹️ Upload Negative Words File", type=["txt"])

if input_file and stopword_files and pos_file and neg_file:
    df = pd.read_excel(input_file)
    stopwords_set = load_word_list(stopword_files)
    positive_words = load_word_list([pos_file])
    negative_words = load_word_list([neg_file])
    st.success(f"✅ Loaded {len(stopwords_set)} stopwords, {len(positive_words)} positive words, {len(negative_words)} negative words.")

    if 'URL' in df.columns and 'URL_ID' in df.columns:
        index = st.number_input("Select article index", min_value=0, max_value=len(df)-1, step=1)
        row = df.iloc[index]
        article_url = row['URL']
        article_id = row['URL_ID']

        with st.spinner(f"Extracting article {article_id}..."):
            content = extract_article(article_url)

        if content:
            st.subheader("📄 Original Article Content")
            st.text_area("Original", content, height=300)

            cleaned = " ".join(clean_and_tokenize(content, stopwords_set))
            st.subheader("🧹 Cleaned Content (No Stopwords)")
            st.text_area("Cleaned", cleaned, height=200)

            st.subheader("📊 Analysis Results")
            results = analyze_text(content, stopwords_set, positive_words, negative_words)
            st.dataframe(pd.DataFrame([results]))

            # Save
            os.makedirs("extracted_articles", exist_ok=True)
            with open(f"extracted_articles/{article_id}.txt", "w", encoding="utf-8") as f:
                f.write(content)
            with open(f"extracted_articles/{article_id}_cleaned.txt", "w", encoding="utf-8") as f:
                f.write(cleaned)
            st.success(f"✅ Saved article as {article_id}.txt and cleaned version.")
        else:
            st.error("❌ Failed to extract article content.")
    else:
        st.warning("Excel file must have 'URL' and 'URL_ID' columns.")
else:
    st.info("📂 Please upload the Excel file, stopwords, and pos/neg word files to begin.")
