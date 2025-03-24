# Multi-Agent RAG Chatbot

A sophisticated Retrieval-Augmented Generation (RAG) chatbot that uses multiple specialized agents to ensure high-quality, factual responses. The system combines local knowledge retrieval with web search capabilities when needed, and implements a robust evaluation pipeline to detect and correct hallucinations.

## Features

- **Multi-Agent Architecture**: Each component of the RAG pipeline is handled by a specialized agent
- **Local Knowledge Base**: Retrieve information from your own documents
- **Web Search Fallback**: Automatically search the web when local knowledge is insufficient
- **Quality Evaluation**: Multiple evaluation stages ensure high-quality responses
- **Hallucination Detection**: Dedicated agent to identify and correct factual inaccuracies
- **API Interface**: Easy-to-use FastAPI interface for integration
- **Powered by DeepSeek & MxBai**: Uses deepseek-r1 for LLM operations and mxbai-embed-large for embeddings

## System Architecture

The chatbot implements the following workflow:

1. **User Query Analysis**: The Dialogue Manager Agent analyzes the query to understand the information needs
2. **Local Knowledge Retrieval**: The Retriever Agent searches the vector database for relevant documents
3. **Retrieval Quality Evaluation**: The Retrieval Evaluator Agent assesses the quality and relevance of retrieved documents
4. **Web Search (if needed)**: If local knowledge is insufficient, the system performs a web search
5. **Answer Generation**: The Answer Generator Agent creates a response based on the retrieved context
6. **Answer Quality Evaluation**: The Answer Evaluator Agent assesses the quality of the generated answer
7. **Hallucination Detection**: The Hallucination Checker Agent identifies factual inaccuracies
8. **Final Answer Refinement**: The Final Answer Agent refines the response based on evaluation feedback

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd rag-chatbot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

   > **Important**: Make sure you have the correct versions of langchain packages. If you encounter import errors, try:
   > ```
   > pip install langchain-ollama==0.0.3
   > ```

3. Copy the example environment file and modify as needed:
   ```
   cp .env.example .env
   ```

4. Make sure you have Ollama installed and running:
   ```
   # Follow instructions at https://ollama.ai to install Ollama
   # Start the Ollama server
   ollama serve
   ```

5. Pull the required Ollama models:
   ```
   ollama pull deepseek-r1:latest
   ollama pull mxbai-embed-large:latest
   ```

## Usage

### Indexing Documents

Before using the chatbot, you need to index your documents:

```
python index_documents.py --directory /path/to/your/documents
```

### Command Line Interface

Run the chatbot in interactive mode:

```
python rag_chatbot.py
```

### API

Start the API server:

```
python api.py
```

The API will be available at `http://localhost:8000`. You can access the documentation at `http://localhost:8000/docs`.

### API Endpoints

- `POST /chat`: Process a user query
  ```json
  {
    "query": "What is RAG?",
    "debug": false
  }
  ```

- `GET /health`: Check if the API is running

## Configuration

You can configure the behavior of the chatbot by editing the `.env` file. See `.env.example` for available options.

## Models

This project uses the following models from Ollama:

- **deepseek-r1:latest** - A powerful language model used for all LLM operations including dialogue management, retrieval, answer generation, and evaluation
- **mxbai-embed-large:latest** - High-quality embedding model used for vector search operations

## Dependencies

- langchain and langchain-ollama for LLM integration
- Ollama for running LLMs locally
- ChromaDB for vector storage
- FastAPI for the API server
- DuckDuckGo Search for web search

## Troubleshooting

If you encounter an error like `ImportError: cannot import name 'Ollama' from 'langchain_ollama'`, make sure you have the latest version of langchain-ollama installed or try installing a specific version with `pip install langchain-ollama==0.0.3`.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 