# NLIP Project: Document Similarity Checker

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A powerful Natural Language Processing (NLP) tool for comparing research papers and documents. This application uses TF-IDF vectorization and cosine similarity to analyze document similarity at both document and sentence levels.

## Features

- 📄 **PDF Document Comparison**: Upload and compare multiple PDF documents
- 📊 **Overall Similarity Scores**: Get cosine similarity scores between documents
- 🔍 **Sentence-Level Analysis**: Identify the most similar sentences across documents
- 🌐 **Web-Based Interface**: Easy-to-use Gradio web interface
- ⚡ **Fast Processing**: Efficient TF-IDF based similarity computation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/NLIP_project.git
cd NLIP_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python NLP_project/main.py
```

This will launch a web interface where you can:
1. Upload your main document (PDF)
2. Upload one or more reference documents (PDFs)
3. Click "Compare Documents" to get similarity analysis

## How It Works

The application preprocesses text by:
- Tokenizing and removing stopwords
- Converting to TF-IDF vectors
- Computing cosine similarity between documents
- Finding sentence-level matches with similarity > 0.4

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
