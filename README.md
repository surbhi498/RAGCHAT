## Chat-with-PDF-Chatbot

![Chatbot Image](./images/rag.png)


**Chat-with-PDF-Chatbot** is an interactive application designed to assist users in interacting with their PDF documents using an Open Source Stack.

## Getting Started

### Installation

```sh
# Clone the repository
git clone <repository_url>

# Create necessary folders

To set up the project on your local machine, follow these steps to create the necessary folders:

```sh
# Create necessary folders
mkdir db
mkdir models
mkdir docs

## Usage

### Run the ingestion script to prepare the data

```sh
python ingest.py

To start the chatbot application using Streamlit, run the following command in your terminal:

```sh
streamlit run chatbot_app.py

# RAG-Powered Chatbot for PDF Document Question Answering 📝

## Objective 🌟
Develop a RAG (Retrieval-Augmented Generation) powered chatbot that answers questions based on the content of PDF documents. This chatbot helps users efficiently retrieve specific information from large documents.

## What 🔍
- **Dataset Creation**: Extracted and processed text from PDF documents to create training and validation datasets.
- **Model Fine-Tuning**: Fine-tuned a pre-trained language model for generating high-quality question-answer pairs.
- **RAG Implementation**: Leveraged Retrieval-Augmented Generation to enhance the chatbot's response accuracy and relevance.

## Why 🚀
- **Efficiency**: Enable efficient information retrieval from extensive documents.
- **Usability**: Improve the effectiveness of automated QA systems in various domains like customer support, legal analysis, and education.

## How ⚙️

### Data Preparation 📂
- **PDF Loading and Reading**: Used `SimpleDirectoryReader` to load and read PDF documents.
- **Tokenization and Text Splitting**: Applied `SentenceSplitter` to tokenize and split the text into manageable chunks.
- **QA Pair Generation**: Employed OpenAI GPT-3.5 to generate relevant question-answer pairs from the text chunks.

### Dataset Construction 📊
- **Training and Validation Split**: Split the dataset into training (80%) and validation (20%) sets.
- **Data Storage**: Stored datasets in JSON format for easy access and reuse.

### Model Fine-Tuning 🛠️
- **Embedding Pairs**: Generated embedding pairs for training using OpenAI.
- **Fine-Tuning Engine**: Utilized `SentenceTransformersFinetuneEngine` to fine-tune the pre-trained model (`BAAI/bge-small-en`).
- **Evaluation Metrics**: Measured performance using hit rate, retrieval rate, MRR, NDCG, and MAP.

### Evaluation 📈
- **Hit Rate**: Achieved a hit rate of **0.94**, evaluating the accuracy of retrieving the correct answer within the top-k results.
- **Information Retrieval Rate**: Achieved an information retrieval rate of **0.78**, assessing the model's effectiveness in retrieving relevant information.

### Implementation 💡
- **RAG Model**: Integrated the fine-tuned model with a RAG approach for enhanced response accuracy and relevance.
- **User Interface**: Developed a simple front-end using Streamlit for user interaction and querying the PDF document.

## Skills and Technologies Used 🛠️
- **Programming Languages**: Python
- **Libraries and Frameworks**: LlamaIndex, SentenceTransformers, OpenAI GPT-3.5, Streamlit
- **Tools**: Git, JSON, PDF reader libraries

---

## Evaluation Metrics 📊

| Metric              | Value  |
|---------------------|--------|
| **Hit Rate**        | 0.94   |
| **Retrieval Rate**  | 0.78   |

### Evaluation Details

| epoch | steps | cos_sim-Accuracy@1 | cos_sim-Accuracy@3 | cos_sim-Accuracy@5 | cos_sim-Accuracy@10 | cos_sim-Precision@1 | cos_sim-Recall@1 | cos_sim-Precision@3 | cos_sim-Recall@3 | cos_sim-Precision@5 | cos_sim-Recall@5 | cos_sim-Precision@10 | cos_sim-Recall@10 | cos_sim-MRR@10 | cos_sim-NDCG@10 | cos_sim-MAP@100 | dot_score-Accuracy@1 | dot_score-Accuracy@3 | dot_score-Accuracy@5 | dot_score-Accuracy@10 | dot_score-Precision@1 | dot_score-Recall@1 | dot_score-Precision@3 | dot_score-Recall@3 | dot_score-Precision@5 | dot_score-Recall@5 | dot_score-Precision@10 | dot_score-Recall@10 | dot_score-MRR@10 | dot_score-NDCG@10 | dot_score-MAP@100 |
|-------|-------|--------------------|--------------------|--------------------|---------------------|---------------------|------------------|---------------------|------------------|---------------------|------------------|----------------------|-------------------|----------------|-----------------|-----------------|----------------------|----------------------|----------------------|-----------------------|----------------------|-------------------|----------------------|-------------------|----------------------|-------------------|-----------------------|--------------------|-------------------|--------------------|-------------------|
| -1    | -1    | 0.68               | 0.86               | 0.94               | 0.98                | 0.68                | 0.68             | 0.2866666666666666  | 0.86             | 0.18799999999999997  | 0.94             | 0.09799999999999998  | 0.98              | 0.7881666666666667 | 0.8353412564069735 | 0.789219298245614 | 0.68                  | 0.86                  | 0.94                  | 0.98                   | 0.68                  | 0.68               | 0.2866666666666666    | 0.86               | 0.18799999999999997   | 0.94               | 0.09799999999999998    | 0.98               | 0.7881666666666667    | 0.8353412564069735 | 0.789219298245614 |

### Usage

#### Run the ingestion script to prepare the data

```sh
python ingest.py
