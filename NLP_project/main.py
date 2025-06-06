!pip install PyMuPDF
!pip install python-docx
!pip install gradio
!pip install gradio -q
!pip install scikit-learn pandas PyPDF2 nltk
import tempfile
import nltk
nltk.download('punkt_tab', quiet=True) # Download the punkt_tab resource
nltk.download('punkt', quiet=True)
import tempfile
from PIL import Image

import gradio as gr
import PyPDF2
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download NLTK data
nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# ---- Read PDF and preprocess ----
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def preprocess(text):
    tokens = word_tokenize(text.lower())
    return ' '.join([w for w in tokens if w.isalnum() and w not in stop_words])

# ---- Compute overall similarity ----
def compute_doc_similarity(input_text, reference_texts):
    documents = [input_text] + reference_texts
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(documents)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    return cosine_sim[0]

# ---- Compute sentence-level similarity ----
def sentence_level_similarity(input_text, ref_text):
    input_sents = sent_tokenize(input_text)
    ref_sents = sent_tokenize(ref_text)

    all_sents = input_sents + ref_sents
    tfidf = TfidfVectorizer().fit(all_sents)

    input_vecs = tfidf.transform(input_sents)
    ref_vecs = tfidf.transform(ref_sents)

    sim_matrix = cosine_similarity(input_vecs, ref_vecs)

    top_matches = []
    for i, input_sent in enumerate(input_sents):
        best_idx = np.argmax(sim_matrix[i])
        best_score = sim_matrix[i][best_idx]
        if best_score > 0.4:
          top_matches.append((input_sent, ref_sents[best_idx], best_score))


    return top_matches

# ---- Gradio logic ----
def compare_documents(input_file, reference_files):
    if not input_file or not reference_files:
        return "Please upload one input document and at least one reference document."

    input_raw = read_pdf(input_file)
    input_text = preprocess(input_raw)

    reference_texts = []
    reference_raws = []
    reference_names = []

    for ref in reference_files:
        raw_text = read_pdf(ref)
        reference_raws.append(raw_text)
        reference_texts.append(preprocess(raw_text))
        reference_names.append(ref.name)

    # Document-level similarity
    doc_similarities = compute_doc_similarity(input_text, reference_texts)

    result_md = "## 📊 Overall Document Similarity\n"
    for name, score in sorted(zip(reference_names, doc_similarities), key=lambda x: x[1], reverse=True):
        result_md += f"- **{name}**: {score:.4f}\n"

    # Sentence-level matches
    result_md += "\n---\n## 📌 Sentence-Level Similarity (Top Matches)\n"
    for name, raw_text, score in zip(reference_names, reference_raws, doc_similarities):
        result_md += f"\n### 🔍 {name} (score: {score:.4f})\n"
        matches = sentence_level_similarity(input_raw, raw_text)
        for input_sent, ref_sent, sim in sorted(matches, key=lambda x: x[2], reverse=True)[:10]:  #change the limit number of sentence similarity in results

            result_md += f"- 🔹 **Input:** \"{input_sent.strip()}\"\n"
            result_md += f"  \t➡️ **Match:** \"{ref_sent.strip()}\"\n"
            result_md += f"  \t**Similarity:** {sim:.3f}\n\n"
        if not matches:
            result_md += "No strong sentence-level matches found.\n"

    return result_md

# ---- Gradio UI ----
with gr.Blocks() as demo:
    gr.Markdown("# 📄 Document Similarity Checker with Sentence-Level Analysis")
    gr.Markdown("Upload a main document and one or more reference PDFs. Get overall similarity and key matching sentences.")

    with gr.Row():
        input_file = gr.File(label="Upload Input Document", file_types=[".pdf"])
        reference_files = gr.File(label="Upload Reference Documents", file_types=[".pdf"], file_count="multiple")

    compare_btn = gr.Button("Compare Documents")
    output = gr.Markdown()

    compare_btn.click(fn=compare_documents, inputs=[input_file, reference_files], outputs=output)

demo.launch()
