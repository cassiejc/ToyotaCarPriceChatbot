import streamlit as st
from pathlib import Path
import qdrant_client
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.llms import ChatMessage
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondensePlusContextChatEngine
import pdfplumber

# Set page config
st.set_page_config(page_title="Carbot", layout="wide")

CONTEXT_PROMPT = """You are an expert system with knowledge of Toyota car prices.
These are documents that may be relevant to the user's question:\n\n
{context_str}
If you deem this piece of information is relevant, you may use it to answer the user. 
Else, you can say that you DON'T KNOW."""

class Chatbot:
    def __init__(self, llm="qwen2.5:7b", embedding_model="intfloat/multilingual-e5-large", vector_store=None):
        self.Settings = self.set_setting(llm, embedding_model)
        self.index = self.load_data()
        self.memory = self.create_memory()
        self.chat_engine = self.create_chat_engine(self.index)

    @staticmethod
    def set_setting(llm, embedding_model):
        Settings.llm = Ollama(model=llm, base_url="http://127.0.0.1:11434")
        Settings.embed_model = OllamaEmbedding(base_url="http://127.0.0.1:11434", model_name="mxbai-embed-large:latest")
        Settings.system_prompt = """
        You are an expert assistant with knowledge of Toyota car prices in the DKI Jakarta area and its surroundings (License Plate B). 
        You will help users find information about car prices, models, feature, car recomendations, car care tips, and types based on the provided price list and the docs. 
        If you don't know the answer, say that you DON'T KNOW.
        """
        return Settings

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_data(vector_store=None):
        with st.spinner(text="Loading and indexing – hang tight! This should take a few minutes."):
            reader = SimpleDirectoryReader(input_dir="./docs", recursive=True)
            documents = reader.load_data()

        if vector_store is None:
            client = qdrant_client.QdrantClient(
                url=st.secrets["qdrant"]["connection_url"], 
                api_key=st.secrets["qdrant"]["api_key"],
            )
            vector_store = QdrantVectorStore(client=client, collection_name="Pricelist Mobil Toyota")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        return index

    def set_chat_history(self, messages):
        self.chat_history = [ChatMessage(role=message["role"], content=message["content"]) for message in messages]
        self.chat_store.store = {"chat_history": self.chat_history}

    def create_memory(self):
        self.chat_store = SimpleChatStore()
        return ChatMemoryBuffer.from_defaults(chat_store=self.chat_store, chat_store_key="chat_history", token_limit=16000)

    def create_chat_engine(self, index):
        return CondensePlusContextChatEngine(
            verbose=True,
            memory=self.memory,
            retriever=index.as_retriever(),
            llm=Settings.llm,
            context_prompt=CONTEXT_PROMPT
        )

# --- Styling tombol sidebar ---
st.markdown(
    """
    <style>
    .sidebar-button {
        width: 100%;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
        text-align: left;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .sidebar-button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar navigation ---
if "page" not in st.session_state:
    st.session_state.page = "Chatbot"

st.sidebar.title("Carbot")

if st.sidebar.button("Chatbot", key="btn_chatbot"):
    st.session_state.page = "Chatbot"
if st.sidebar.button("File Docs", key="btn_docs"):
    st.session_state.page = "File Docs"
if st.sidebar.button("Booking Service", key="btn_booking"):
    st.session_state.page = "Booking"

# --- Upload File Section ---
st.sidebar.markdown("### 📄 Upload File ke 'docs'")
uploaded_file = st.sidebar.file_uploader("Pilih file untuk diunggah", type=["pdf", "txt", "docx"])
if uploaded_file:
    docs_folder = Path("./docs")
    docs_folder.mkdir(exist_ok=True)

    save_path = docs_folder / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"File '{uploaded_file.name}' berhasil diunggah dan disimpan!")

# --- Page: Chatbot ---
if st.session_state.page == "Chatbot":
    st.title("Toyota Car Price Chatbot")
    chatbot = Chatbot()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Halo! 👋\n\nSenang bertemu dengan Anda, ada yang bisa saya bantu? Silakan bertanya tentang harga, spesifikasi, fitur mobil Toyota 😁"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    chatbot.set_chat_history(st.session_state.messages)

    if prompt := st.chat_input("Apa yang ingin Anda tanyakan?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            response = chatbot.chat_engine.chat(prompt)
            st.markdown(response.response)
        st.session_state.messages.append({"role": "assistant", "content": response.response})

# --- Page: File Docs ---
elif st.session_state.page == "File Docs":
    st.title("📁 Daftar File di Folder 'docs'")

    docs_path = Path("./docs")
    if docs_path.exists() and docs_path.is_dir():
        file_list = [f.relative_to(docs_path).as_posix() for f in docs_path.glob("**/*") if f.is_file()]
        if file_list:
            selected_file = st.selectbox("Pilih file untuk dilihat", file_list)
            full_path = docs_path / selected_file

            if selected_file.lower().endswith(".pdf"):
                try:
                    with pdfplumber.open(full_path) as pdf:
                        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                    st.markdown(f"### Isi File PDF: `{selected_file}`")
                    st.text_area("Isi PDF", text, height=400)
                except Exception as e:
                    st.error(f"Gagal membaca file PDF: {e}")
            else:
                try:
                    file_content = full_path.read_text(encoding="utf-8", errors="ignore")
                    st.markdown(f"### Isi File: `{selected_file}`")
                    st.code(file_content[:1000])
                except Exception as e:
                    st.error(f"Gagal membaca file: {e}")
        else:
            st.info("Tidak ada file ditemukan di folder 'docs'.")
    else:
        st.error("Folder 'docs' tidak ditemukan.")

# --- Page: Booking Service ---
elif st.session_state.page == "Booking":
    st.title("🚗 Booking Service Toyota")

    with st.form("booking_form"):
        nama = st.text_input("Nama Lengkap")
        plat_nomor = st.text_input("Nomor Polisi")
        tipe_mobil = st.selectbox("Tipe Mobil", ["Avanza", "Rush", "Yaris", "Innova", "Fortuner", "Lainnya"])
        lokasi = st.text_input("Lokasi Dealer")
        tanggal = st.date_input("Tanggal Booking")
        waktu = st.time_input("Waktu Booking")
        jenis_servis = st.multiselect("Jenis Servis", ["Servis Berkala", "Ganti Oli", "Pemeriksaan Rem", "Servis AC", "Lainnya"])

        submitted = st.form_submit_button("Kirim Booking")

        if submitted:
            st.success(f"Terima kasih {nama}, booking servis Anda untuk mobil {tipe_mobil} dengan plat {plat_nomor} pada {tanggal} pukul {waktu} di {lokasi} telah diterima. Jenis servis: {', '.join(jenis_servis)}")
