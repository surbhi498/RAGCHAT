import streamlit as st
import os
import base64
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import textwrap
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from constants import CHROMA_SETTINGS
from langchain_huggingface import HuggingFaceEmbeddings
from streamlit_chat import message

st.set_page_config(layout="wide")

device = torch.device('cpu')

checkpoint = "MBZUAI/LaMini-T5-738M"
print(f"Checkpoint path: {checkpoint}")  # Add this line for debugging
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
   # device_map=device,
    torch_dtype=torch.float32
)
# Explicitly move the model to CPU
#base_model.to(device)
persist_directory = "db"


@st.cache_resource
def data_ingestion():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    # create embeddings here
    embeddings = HuggingFaceEmbeddings(model_name="./test1_model")
    # create vector store here
    db = Chroma.from_documents(
        texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None


@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=512,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
        # device=device
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = HuggingFaceEmbeddings(model_name="./test1_model")
    db = Chroma(persist_directory="db", embedding_function=embeddings,
                client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa


def process_answer(instruction):
    response = ''
    if 'query' in instruction:
        user_input = instruction['query']  # Retrieve user input from dictionary
        
        # Check if user input length exceeds model's max length
        max_length = 512
        words = user_input.split()
        
        if len(words) > max_length:
            # Split user_input into chunks of max_length
            segments = [words[i:i + max_length] for i in range(0, len(words), max_length)]
            
            # Process each segment and combine results
            generated_responses = []
            for segment in segments:
                qa = qa_llm()
                # Create a prompt that encourages a clear and logical response
                segment_text = ' '.join(segment)
                prompt = f"Based on the information provided in your car insurance policy booklet, please provide a clear, logical, and well-structured answer to the following query:\n\nQuery: {segment_text}\n\nAnswer:"
                generated_text = qa({'query': prompt})
                generated_responses.append(generated_text['result'])
                # Print and store the context from source documents
                if 'source_documents' in generated_text:
                    context = ' '.join([doc.page_content for doc in generated_text['source_documents']])
                    print(f"Context for segment: {context}")
            
            # Combine generated responses
            response = ' '.join(generated_responses)
        else:
            # Process user_input directly if it's within the max length
            qa = qa_llm()
            
            # Create a prompt that encourages a clear and logical response
            prompt = f"Based on the information provided in your car insurance policy booklet, please provide a clear, logical, and well-structured answer to the following query:\n\nQuery: {user_input}\n\nAnswer:"
            print(f"Single Prompt: {prompt}") 
            # Get the response including source documents
            generated_text = qa({'query': prompt})

            # Print and store the context from source documents
            if 'source_documents' in generated_text:
                context = ' '.join([doc.page_content for doc in generated_text['source_documents']])
                print(f"Context: {context}")

            response = generated_text['result']
           
    
    return response

def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size


@st.cache_data
# function to display the PDF of a given file
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

# Display conversation history using Streamlit messages


def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i], key=str(i))


def main():
    st.markdown("<h1 style='text-align: center; color: blue;'>Chat with your PDF 🦜📄 </h1>",
                unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: grey;'>Built by <a href='https://github.com/surbhi498/RAG_CHATBOT_WITH_PDF1.git'>SURBHI SHARMA ❤️ </a></h3>", unsafe_allow_html=True)

    st.markdown("<h2 style='text-align: center; color:red;'>Upload your PDF 👇</h2>",
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["pdf"])

    if uploaded_file is not None:
        file_details = {
            "Filename": uploaded_file.name,
            "File size": get_file_size(uploaded_file)
        }
        filepath = "docs/"+uploaded_file.name
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("<h4 style color:black;'>File details</h4>",
                        unsafe_allow_html=True)
            st.json(file_details)
            st.markdown("<h4 style color:black;'>File preview</h4>",
                        unsafe_allow_html=True)
            pdf_view = displayPDF(filepath)

        with col2:
            with st.spinner('Embeddings are in process...'):
                ingested_data = data_ingestion()
            st.success('Embeddings are created successfully!')
            st.markdown("<h4 style color:black;'>Chat Here</h4>",
                        unsafe_allow_html=True)

            user_input = st.text_input("", key="input")

            # Initialize session state for generated responses and past messages
            if "generated" not in st.session_state:
                st.session_state["generated"] = ["I am ready to help you"]
            if "past" not in st.session_state:
                st.session_state["past"] = ["Hey there!"]

            # Search the database for a response based on user input and update session state
            if user_input:
                answer = process_answer({'query': user_input})
                st.session_state["past"].append(user_input)
                response = answer
                st.session_state["generated"].append(response)

            # Display conversation history using Streamlit messages
            if st.session_state["generated"]:
                display_conversation(st.session_state)


if __name__ == "__main__":
    main()
