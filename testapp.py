import os
import streamlit as st
import google.generativeai as genai
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaFileUpload
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import Chroma
import pandas as pd
import io
from googleapiclient.http import MediaIoBaseDownload
from langchain.chains import RetrievalQA

st.title("Dira's Personal Chatbot")

# Load environment variables
load_dotenv()

SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'service_account.json'
PARENT_FOLDER_ID = os.getenv('PARENT_FOLDER_ID')
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('drive', 'v3', credentials=credentials)

# Set the API key for Google Generative AI
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Authenticate and build the service
def authenticate():
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return creds

# Upload a file to Google Drive
def upload_file(file_path):
    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)

    file_metadata = {
        'name': os.path.basename(file_path),
        'parents': [PARENT_FOLDER_ID]
    }

    media = MediaFileUpload(file_path, mimetype='application/pdf', resumable=True)

    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()

    return file.get('id')

def get_files_in_folder(parent_id):
    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)
    
    query = f"'{parent_id}' in parents and trashed=false"
    response = service.files().list(q=query, fields='files(name,id,mimeType,webViewLink,createdTime,modifiedTime)').execute()
    files = response.get('files', [])
    dfs = [pd.DataFrame(files)]
    for file in files:
        if file['mimeType'] == 'application/vnd.google-apps.folder':
            dfs.append(get_files_in_folder(file['id']))
    return pd.concat(dfs, ignore_index=True)

# Download the PDF from Google Drive using its file ID and save it to a temp directory
def download_pdf(file_id, save_path):
    creds = authenticate()
    service = build('drive', 'v3', credentials=creds)
    
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(save_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}% complete.")

# Streamlit Sidebar
with st.sidebar:
    uploaded_files = st.file_uploader("Upload PDF here", accept_multiple_files=True, type=['pdf'])
    if st.button("Submit & Process"):
        if uploaded_files:
            with st.spinner("Processing..."):
                for uploaded_file in uploaded_files:
                    # Save the uploaded file temporarily
                    with open(uploaded_file.name, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Upload the file to Google Drive
                    file_id = upload_file(uploaded_file.name)
                    st.success(f"File {uploaded_file.name} uploaded successfully with ID: {file_id}")
                    
                    # Process the PDF
                    new_folder_path = "downloaded_pdfs"
                    if not os.path.exists(new_folder_path):
                         os.makedirs(new_folder_path)
    
                    # Get list of PDF files
                    df = get_files_in_folder(PARENT_FOLDER_ID)
                    df = df[df['mimeType'] == 'application/pdf']

                    # Download each PDF and save it to the newly created folder
                    for _, row in df.iterrows():
                        pdf_id = row['id']
                        pdf_name = row['name']
                        save_path = os.path.join(new_folder_path, pdf_name)
                        download_pdf(pdf_id, save_path)
                        print(f"Downloaded {pdf_name} to {new_folder_path}")

                    loader = PyPDFDirectoryLoader(new_folder_path)
                    docs = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
                    context = "\n\n".join(str(p.page_content) for p in docs)
                    texts = text_splitter.split_text(context)
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    vector_index = Chroma.from_texts(texts, embeddings).as_retriever()
                    
                    # Store vector_index in session_state
                    st.session_state.vector_index = vector_index
                    
                    prompt_template = """
                        You are a helpful, warm, and friendly chatbot with a positive and engaging personality. Your responses should reflect this personality, making the conversation pleasant and welcoming. 
                        When greeted with common greetings like "Hi" or "Hello," respond in a warm and friendly manner, and offer to assist with any questions or information the user might need.
                        Answer the question as detailed as possible from the provided context, making sure to provide all the details. If the answer is not available in the provided context, just say, "The answer is not available in the context," and avoid providing incorrect information. Your answers can consist emojis too if you want to, to add friendly manner.
                        Context: \n {context}?\n
                        Question: \n{question}\n
                        Answer:
                    """
                    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
                    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
                    #chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
                    chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_index, chain_type_kwargs={"prompt":prompt})
                    # Store chain in session_state
                    st.session_state.chain = chain
        else:
            st.warning("Please upload at least one PDF file.")

# Streamlit Chatbot
chat_session = st.session_state.get('chat_session', None)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    if "vector_index" in st.session_state and "chain" in st.session_state:
        vector_index = st.session_state.vector_index
        chain = st.session_state.chain

        #docs = vector_index.get_relevant_documents(prompt)
        #response_llm = chain({"input_documents": docs, "question": prompt}, return_only_outputs=True)
        result = chain({"query":prompt})

        with st.chat_message("assistant"):
            st.markdown(result['result'])
        st.session_state.messages.append({"role": "assistant", "content": result['result']})
    else:
        st.error("Please process a PDF first before asking questions.")

