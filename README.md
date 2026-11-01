from __future__ import annotations

import re
from collections import Counter
from typing import List, Dict, Tuple

import streamlit as st
from transformers import pipeline

MODEL_NAME = "sshleifer/distilbart-cnn-12-4"


@st.cache_resource
def load_summarizer():
    """Load the summarization pipeline once and reuse it across reruns."""
    return pipeline("summarization", model=MODEL_NAME, tokenizer=MODEL_NAME)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_into_chunks(text: str, max_words: int = 500) -> List[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    for index in range(0, len(words), max_words):
        chunk = " ".join(words[index:index + max_words])
        chunks.append(chunk)
    return chunks


def fallback_summary(text: str, target_words: int = 80) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return ""

    stop_words = {
        "the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "with",
        "without", "from", "to", "of", "in", "on", "at", "by", "as", "is", "it",
        "its", "be", "was", "were", "are", "this", "that", "these", "those", "they",
        "them", "their", "he", "she", "we", "you", "i", "me", "my", "our", "your",
    }

    word_freq = Counter()
    for sentence in sentences:
        for word in re.findall(r"\b[a-zA-Z]+\b", sentence.lower()):
            if word not in stop_words and len(word) > 3:
                word_freq[word] += 1

    if not word_freq:
        return " ".join(sentences[:2])

    ranked = []
    for sentence in sentences:
        score = 0
        for word in re.findall(r"\b[a-zA-Z]+\b", sentence.lower()):
            if word in word_freq:
                score += word_freq[word]
        ranked.append((score, sentence))

    ranked.sort(key=lambda x: x[0], reverse=True)
    selected = []
    total_words = 0
    for _, sentence in ranked:
        sentence_words = sentence.split()
        if total_words + len(sentence_words) <= target_words + 20:
            selected.append(sentence)
            total_words += len(sentence_words)
        if total_words >= target_words:
            break

    if not selected:
        return sentences[0]
    summary = " ".join(selected)
    return re.sub(r"\s+", " ", summary).strip()


def generate_summary(document: str, target_words: int = 80) -> str:
    text = normalize_text(document)
    if not text:
        return "Please enter some text to summarize."

    if len(text.split()) <= 150:
        summarizer = load_summarizer()
        try:
            result = summarizer(
                text,
                max_length=min(220, max(60, target_words + 25)),
                min_length=max(15, min(target_words, 80)),
                do_sample=False,
            )
            if result:
                return result[0]["summary_text"].strip()
        except Exception:
            pass
        return fallback_summary(text, target_words=target_words)

    chunks = split_into_chunks(text)
    summarizer = load_summarizer()
    chunk_summary_parts = []
    for chunk in chunks:
        try:
            result = summarizer(
                chunk,
                max_length=min(200, max(60, target_words // max(1, len(chunks)) + 20)),
                min_length=max(10, min(target_words // max(1, len(chunks)), 50)),
                do_sample=False,
            )
            if result:
                chunk_summary_parts.append(result[0]["summary_text"].strip())
        except Exception:
            chunk_summary_parts.append(fallback_summary(chunk, target_words=max(25, target_words // max(1, len(chunks)))))

    combined = " ".join(chunk_summary_parts)
    combined = normalize_text(combined)
    if len(combined.split()) <= target_words:
        return combined

    return fallback_summary(combined, target_words=target_words)


def tokenize(document: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z0-9'-]+\b", document.lower())


def evaluate_summary(source_text: str, candidate_summary: str) -> Dict[str, float]:
    source_words = tokenize(source_text)
    summary_words = tokenize(candidate_summary)
    if not summary_words:
        return {"readability": 0.0, "relevance": 0.0, "accuracy": 0.0}

    source_set = set(source_words)
    summary_set = set(summary_words)
    overlap = len(summary_set & source_set) / len(summary_set)

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", candidate_summary) if s.strip()]
    if sentences:
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        readability = max(0.0, min(1.0, 1 - abs(avg_sentence_length - 18) / 22))
    else:
        readability = 0.0

    unsupported = 0
    for word in summary_set:
        if word not in source_set and len(word) > 4:
            unsupported += 1
    accuracy = max(0.0, 1 - (unsupported / max(1, len(summary_set))))

    return {
        "readability": round(readability, 3),
        "relevance": round(overlap, 3),
        "accuracy": round(accuracy, 3),
    }


def main() -> None:
    st.set_page_config(page_title="AI Text Summarizer", page_icon="✍️", layout="wide")
    st.title("AI Text Summarizer")
    st.caption("Transform long documents into concise summaries using an NLP summarization model.")

    with st.form("summarizer_form"):
        document = st.text_area(
            "Paste your text or document here",
            height=260,
            placeholder="Paste a long article, report, or meeting notes here..."
        )
        target_words = st.slider("Summary length", min_value=25, max_value=200, value=80, step=5)
        submitted = st.form_submit_button("Generate Summary")

    if submitted:
        if not document.strip():
            st.warning("Please enter text before summarizing.")
            return

        with st.spinner("Generating a focused summary..."):
            summary = generate_summary(document, target_words=target_words)
            quality = evaluate_summary(document, summary)

        st.subheader("Summary")
        st.write(summary)

        st.subheader("Quality Check")
        col1, col2, col3 = st.columns(3)
        col1.metric("Readability", f"{quality['readability'] * 100:.1f}%")
        col2.metric("Relevance", f"{quality['relevance'] * 100:.1f}%")
        col3.metric("Accuracy", f"{quality['accuracy'] * 100:.1f}%")

        st.caption(
            "These scores are heuristic checks to gauge readability, relevance, and factual alignment "
            "with the source material."
        )


if __name__ == "__main__":
    main()


path=README.md
content
"""# AI Text Summarizer

This project is a lightweight web application that turns long text into concise summaries using an NLP summarization model.

## Features
- Paste any text or document
- Adjust summary length with a slider
- Summarize using the Hugging Face transformer model
- View basic quality checks for readability, relevance, and accuracy
- Built as a simple Streamlit app

## Project GitHub
https://github.com/maheshguntipally08-alt/ai-text-summarizer

## Run locally

1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the app:

```bash
streamlit run app.py
```

## Example usage
- Paste meeting notes or a long article
- Choose a summary length like 50 or 100 words
- The app generates a concise summary and quality stats

## Technologies used
- Python
- Streamlit
- Hugging Face Transformers
- NLP summarization model

## Notes
This app uses a transformer-based summarization model, with a fallback rule-based summary for reliability when the model is unavailable or the document is too long.
"""
