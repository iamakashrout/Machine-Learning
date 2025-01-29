import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

app=Flask(__name__)

# Folder to store uploaded PDFs
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

vector_store = None
qa_chain = None

# Load and Process PDF
def process_pdf(pdf_path):
    global vector_store, qa_chain

    # load PDF
    loader=PyPDFLoader(pdf_path)
    pages=loader.load()
    # split PDF content into chunks
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs=text_splitter.split_documents(pages)
    # create embeddings
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store=FAISS.from_documents(docs, embeddings)
    # create LLM
    llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.1", model_kwargs={"temperature": 0.3, "max_length": 512})
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=False)

# Home Page
@app.route("/")
def index():
    return render_template("index.html")

# Upload PDF and process it
@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Process the PDF
    process_pdf(file_path)

    return jsonify({"message": "PDF uploaded and processed successfully!", "filename": filename})

# Ask a question
@app.route("/ask", methods=["POST"])
def ask_question():
    global qa_chain
    if not qa_chain:
        return jsonify({"error": "No PDF has been uploaded and processed yet."})

    data = request.json
    query = data.get("question")

    if not query:
        return jsonify({"error": "No question provided."})

    response = qa_chain({"query": query})  # Get full response dictionary

    # Extract only the answer
    if isinstance(response, dict) and "result" in response:
        answer = response["result"]
    else:
        answer = response  # Fallback in case of unexpected structure

    return jsonify({"answer": answer.strip()})

if __name__ == "__main__":
    app.run(debug=True)