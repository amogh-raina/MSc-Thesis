# LLM Legal Knowledge Evaluator (RAG Streamlit App)

## Overview
This application provides a production-ready Streamlit interface for evaluating LLMs on legal question-answering tasks. It supports multi-provider LLMs, dynamic embedding selection, RAG (Retrieval-Augmented Generation), and real-time evaluation using RAGAS metrics (BLEU, ROUGE, String & Semantic Similarity). Results can be exported for further analysis.

---

## Features
- **Multi-Provider LLM Support:** NVIDIA, Groq, Mistral, OpenAI, Google, GitHub (via Azure)
- **Dynamic Embedding Selection:** Choose from OpenAI, Nomic, Jina, Google, Mistral, HuggingFace, or SentenceTransformers
- **Flexible Question Bank Loading:** Load legal Q&A datasets (JSON) with or without embeddings
- **Manual, Batch, and RAG Q&A Modes:** Evaluate LLMs manually, in batch, or with retrieval-augmented context
- **Reference Answer Matching:** Uses embedding similarity or fuzzy matching for reference lookup
- **Real-Time RAGAS Evaluation:** BLEU, ROUGE, String, and (optionally) Semantic Similarity
- **Export Results:** Download evaluation results as Excel or JSON
- **Configurable RAG Pipeline:** Upload your own paragraph-to-paragraph dataset for RAG

---

## Installation

1. **Clone/Download the Application Files**
   - Save the main application as `rag_app.py` and the requirements file as `requirements.txt`.

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**
   Create a `.env` file in your project root with your API keys:
   ```env
   # LLM Provider API Keys (at least one required)
   NVIDIA_API_KEY=your_nvidia_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   MISTRAL_API_KEY=your_mistral_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   GITHUB_TOKEN=your_github_token_here

   # Embedding Provider API Keys (at least one required)
   NOMIC_API_KEY=your_nomic_api_key_here
   JINA_API_KEY=your_jina_api_key_here
   # OPENAI_API_KEY, GOOGLE_API_KEY, MISTRAL_API_KEY also work for embeddings
   # HuggingFace and SentenceTransformers do not require API keys for public models
   ```

4. **Prepare Your Data**
   - Place your legal Q&A JSON files in a directory (default: `./JSON Trial 1/`) with the naming pattern `BEUL_EXAM_*.json`.
   - Each JSON file should contain an array of objects:
     ```json
     [
       {
         "id": "unique_id",
         "year": "2023",
         "question_number": "1",
         "question_text": "Your question here",
         "answer_text": "Your answer here"
       }
     ]
     ```

---

## Usage

1. **Launch the Application**
   ```bash
   streamlit run rag_app.py
   ```

2. **Sidebar Configuration**
   - **Data Directory:** Set the path to your Q&A JSON files.
   - **Embedding Configuration:**
     - Enable embeddings for manual evaluation and/or semantic similarity.
     - Select embedding provider/model for each purpose.
   - **RAG Configuration:**
     - Upload a paragraph-to-paragraph dataset (CSV/JSON) for RAG.
     - Set Chroma vector store directory.
     - Build or reset the RAG system as needed.
   - **Question Bank Loading:**
     - Load basic (no embeddings) or enhanced (with embeddings) question bank.
   - **Model Selection:**
     - Choose LLM provider and model.
     - Choose response type (detailed/concise).
   - **Export Results:**
     - Download evaluation results as Excel or JSON.

3. **Main Tabs**
   - **Manual Evaluation:**
     - Enter a legal question.
     - Choose to auto-find a reference answer (embedding/fuzzy match) or provide one manually.
     - Generate and evaluate the LLM's answer with RAGAS metrics.
   - **Batch Evaluation:**
     - Evaluate multiple questions from the loaded database in one go.
     - View aggregate and per-question metrics.
     - Export results.
   - **RAG Q&A:**
     - Ask questions with retrieval-augmented context (if RAG is set up).
     - The system retrieves relevant context and instructs the LLM to cite only from it.
     - Checks for hallucinated citations.

---

## Evaluation Metrics
- **BLEU Score:** n-gram overlap (precision)
- **ROUGE Score:** recall-oriented similarity
- **String Similarity:** non-LLM string similarity
- **Semantic Similarity:** embedding-based (optional, if enabled)

---

## File Structure
```
your_project/
├── rag_app.py              # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables (create this)
├── JSON Trial 1/          # Your JSON data directory
│   ├── BEUL_EXAM_2023.json
│   └── BEUL_EXAM_2024.json
├── chroma_db/             # Vector store (auto-created)
└── RAG_Pipeline/          # RAG pipeline modules
```

---

## Troubleshooting
- **API Key Errors:**
  - Ensure all required API keys are set in your `.env` file and are valid.
- **Document Loading Errors:**
  - Check that JSON files exist and match the expected structure and naming pattern.
- **Embedding Issues:**
  - If using embeddings, ensure the correct provider/model is selected and the package is installed.
- **Vector Store Issues:**
  - Clear the Chroma directory if you encounter persistence issues and rebuild.
- **Performance:**
  - For large datasets, start with a small batch for testing.

---

## Advanced Configuration
- **Customizing Prompts:**
  - Edit the `_rag_prompt()` function in `rag_app.py` to change LLM instructions.
- **Adding New Metrics:**
  - Extend the evaluation logic in `LLMEvaluator` to include additional metrics.
- **Changing Vector Store:**
  - Modify the RAG pipeline to use a different vector store if needed.

---

## Support
- **LangChain:** [LangChain documentation](https://docs.langchain.com)
- **RAGAS:** [RAGAS documentation](https://docs.ragas.io)
- **Streamlit:** [Streamlit documentation](https://docs.streamlit.io)
- **NVIDIA API:** [NVIDIA API documentation](https://docs.nvidia.com)

---

## Next Steps
1. **Deploy:** Deploy to Streamlit Cloud or other platforms
2. **Scale:** Add user authentication and multi-user support
3. **Enhance:** Implement additional RAG techniques like re-ranking
4. **Monitor:** Add logging and performance monitoring