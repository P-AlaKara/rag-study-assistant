import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb import PersistentClient

DATA_DIR = './data'
CHROMA_PERSIST_DIR = './chroma_db'
COLLECTION_NAME = "student_notes_kb"

def get_document_loader(path: str):
    """Returns the correct LangChain loader based on file extension."""
    file_extension = os.path.splitext(path)[1].lower()
    if file_extension == ".pdf":
        return PyPDFLoader(path)
    elif file_extension == ".docx":
        return Docx2txtLoader(path)
    return None

def extract_metadata_from_filename(file_path: str) -> dict:
    """
    Parses a file path based on the convention:
    Category_Unit_Topic_Year_Code.ext
    """
    try:
        file_name_with_ext = os.path.basename(file_path)
        file_name = os.path.splitext(file_name_with_ext)[0]
        
        parts = file_name.split('_')
        
        if len(parts) < 5:
            return {"error": f"Filename does not match convention: {file_name}"}

        metadata = {
            "source_type": parts[0],
            "unit_code": parts[1],
            "topic": parts[2],
            "year": parts[3],
            "unique_code": parts[4],
            "original_file": file_name_with_ext,
        }
        return metadata
    
    except Exception as e:
        print(f"Error parsing metadata for {file_path}: {e}")
        return {"error": str(e)}

def build_knowledge_base():
    """Executes the full RAG indexing pipeline."""
    print("--- 1. Starting Document Loading and Metadata Injection ---")
    
    all_docs = []
    # Walk through the data directory recursively
    for root, _, files in os.walk(DATA_DIR):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            
            # 1. Get the correct loader for the file type
            loader = get_document_loader(file_path)
            if not loader:
                print(f"Skipping unsupported file type: {file_name}")
                continue

            # 2. Extract metadata from the filename
            metadata = extract_metadata_from_filename(file_path)
            if metadata.get("error"):
                print(f"Skipping file due to metadata error: {metadata.get('error')}")
                continue

            # 3. Load the document and inject metadata in one step
            try:
                docs = loader.load()
                for doc in docs:
                    doc.metadata.update(metadata)
                all_docs.extend(docs)
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")

    if not all_docs:
        print("No documents were loaded. Please check the data directory and file names.")
        return None

    print(f"Successfully loaded and annotated {len(all_docs)} documents.")

    print("\n--- 2. Splitting Documents (Chunking) ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""] 
    )
    all_splits = text_splitter.split_documents(all_docs)
    print(f"Split documents into {len(all_splits)} total chunks.")
    
    print("\n--- 3. Embedding and Indexing with Chroma DB (Local Embeddings) ---")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    chroma_client = PersistentClient(path=CHROMA_PERSIST_DIR)

    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        client=chroma_client
    )
    
    print(f"Knowledge Base built and stored persistently in: {CHROMA_PERSIST_DIR}")
    print("\n INDEXING COMPLETE. Ready for the LLM Application Chains.")
    return vectorstore

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data directory '{DATA_DIR}' not found. Please create it and add your study material.")
    else:

        build_knowledge_base()
