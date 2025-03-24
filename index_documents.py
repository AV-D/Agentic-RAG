import argparse
from utils.vector_store import VectorStore
import os

def main():
    parser = argparse.ArgumentParser(description="Index documents into the vector store")
    parser.add_argument(
        "--directory", 
        "-d", 
        type=str, 
        required=True, 
        help="Directory containing text documents to index"
    )
    args = parser.parse_args()
    
    # Validate the directory exists
    if not os.path.isdir(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist")
        return
    
    print(f"Indexing documents from '{args.directory}'...")
    vector_store = VectorStore()
    
    try:
        num_chunks = vector_store.index_documents(args.directory)
        print(f"Successfully indexed documents. Created {num_chunks} chunks in the vector store.")
    except Exception as e:
        print(f"Error indexing documents: {e}")

if __name__ == "__main__":
    main() 