from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import os
from constants import CHROMA_SETTINGS

persist_directory = "db"

def main():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PyPDFLoader(os.path.join(root, file))
    documents = loader.load()
    print("Splitting into chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # Load your fine-tuned model
    print("Loading sentence transformers model")
    finetuned_model_path = "./test1_model"
    # finetuned_model = SentenceTransformer(finetuned_model_path)
    embeddings = HuggingFaceEmbeddings(model_name="./test1_model")
    # # Create embeddings using your fine-tuned model
    # print("Creating embeddings. May take some minutes...")
    # embeddings = SentenceTransformerEmbeddings(model_name=finetuned_model)

    # Create vector store
    print("Creating vector store...")
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None

    print("Ingestion complete! You can now run privateGPT.py to query your documents")

if __name__ == "__main__":
    main()
