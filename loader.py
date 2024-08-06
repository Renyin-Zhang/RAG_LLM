import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document



def load_documents(DATA_PATH):
    
    print("Loading documents")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"The specified path does not exist: {os.path.abspath(DATA_PATH)}")

    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    
    return documents

def split_document(documents: list[Document]):
    
    print("Splitting documents into chunks")
    
    TEXT_SPLITTER = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = TEXT_SPLITTER.split_documents(documents)

    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    return chunks
    

def ingest_documents(embedding_model: str, documents_path: str, db_path: str):
    """
    Loads documents from the specified directory into the FAISS vector database
    after splitting the text into chunks.
    """

    raw_documents = load_documents(documents_path)
    chunks = split_document(raw_documents)
        
    embeddings = OllamaEmbeddings(model=embedding_model)
    
    print("Creating embeddings and loading documents into vectorstore")
    
    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(db_path)
        print("Faiss vector database created successfully")
        
    except Exception as e:
        vectorstore = None
        raise Exception(f"Unable to create embeddings. {e}")
        


def main():    
    raw_docs_path = "./docs"
    model='nomic'
    db_path = "./db/faiss_index"
    ingest_documents(embedding_model=model, documents_path=raw_docs_path, db_path=db_path)

    
if __name__ == "__main__":
    main() 