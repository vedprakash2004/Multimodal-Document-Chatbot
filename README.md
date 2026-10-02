# Multimodal RAG Q&A

A Streamlit-based multimodal RAG application that supports:

- Direct text
- PDF
- TXT / CSV / Markdown
- Audio transcription
- Image understanding
- Text answers
- Generated educational images
- Generated audio answers

## Project Structure

```text
multimodal_rag_project/
├── app.py
├── config.py
├── input_processing.py
├── rag.py
├── generation.py
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:

```env
GROQ_API_KEY=your_key
OPENAI_API_KEY=your_key
```

Run:

```bash
streamlit run app.py
```

## Module Responsibilities

### app.py
Streamlit UI, file uploaders, buttons, question input, and output display.

### config.py
Environment variables, API clients, and Streamlit page configuration.

### input_processing.py
PDF extraction, audio transcription, image understanding, and conversion to LangChain Documents.

### rag.py
Text splitting, HuggingFace embeddings, vector database creation, and similarity retrieval.

### generation.py
Groq text generation, OpenAI image generation, and Groq audio generation.
