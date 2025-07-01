import streamlit as st
from pathlib import Path
import pdfplumber
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
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

# --- Streamlit Config ---
st.set_page_config(page_title="Carbot", layout="wide")

# --- Prompt Template for Chatbot ---
CONTEXT_PROMPT = """You are an expert system with knowledge of Toyota car prices.
These are documents that may be relevant to the user's question:\n\n
{context_str}
If you deem this piece of information is relevant, you may use it to answer the user. 
Else, you can say that you DON'T KNOW."""

# --- Inisialisasi Firestore ---
@st.cache_resource(show_spinner=False)
def init_firestore():
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)
    return firestore.client()

db = init_firestore()

# --- Chatbot Class ---
class Chatbot:
    def __init__(self, llm="qwen2.5:7b", embedding_model="intfloat/multilingual-e5-large"):
        self.Settings = self.set_setting(llm, embedding_model)
        self.index = self.load_data()
        self.memory = self.create_memory()
        self.chat_engine = self.create_chat_engine(self.index)

    @staticmethod
    def set_setting(llm, embedding_model):
        Settings.llm = Ollama(model=llm, base_url="http://127.0.0.1:11434")
        Settings.embed_model = OllamaEmbedding(base_url="http://127.0.0.1:11434",
                                                model_name="mxbai-embed-large:latest")
        Settings.system_prompt = """
        You are an expert assistant with knowledge of Toyota car prices in the DKI Jakarta area (Plat B).
        Help users with price, specs, features, recommendations, care tips. If unsure, say "I DON'T KNOW."
        """
        return Settings

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def load_data():
        with st.spinner("\U0001F504 Loading and indexing documents..."):
            reader = SimpleDirectoryReader(input_dir="./docs", recursive=True)
            docs = reader.load_data()
            client = qdrant_client.QdrantClient(
                url=st.secrets["qdrant"]["connection_url"],
                api_key=st.secrets["qdrant"]["api_key"],
            )
            vs = QdrantVectorStore(client=client, collection_name="Pricelist Mobil Toyota")
            storage = StorageContext.from_defaults(vector_store=vs)
            index = VectorStoreIndex.from_documents(docs, storage_context=storage)
        return index

    def set_chat_history(self, messages):
        self.chat_history = [ChatMessage(role=m["role"], content=m["content"]) for m in messages]
        self.chat_store.store = {"chat_history": self.chat_history}

    def create_memory(self):
        self.chat_store = SimpleChatStore()
        return ChatMemoryBuffer.from_defaults(chat_store=self.chat_store,
                                               chat_store_key="chat_history", token_limit=16000)

    def create_chat_engine(self, index):
        return CondensePlusContextChatEngine(
            verbose=True,
            memory=self.memory,
            retriever=index.as_retriever(),
            llm=Settings.llm,
            context_prompt=CONTEXT_PROMPT
        )

    def get_service_suggestion(self, plat_nomor):
        # Ambil histori booking dari Firestore
        bookings = db.collection("bookings").where("plat_nomor", "==", plat_nomor).order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
        records = [b.to_dict() for b in bookings]
        if not records:
            return "Belum ada histori service untuk plat nomor ini."
        last_service = records[0]
        last_date = last_service.get("tanggal")
        last_types = last_service.get("jenis_servis", [])
        # Logika saran sederhana
        if "Ganti Oli" in last_types:
            return f"Terakhir ganti oli pada {last_date}. Disarankan ganti oli setiap 6 bulan atau 10.000 km."
        elif "Servis Berkala" in last_types:
            return f"Terakhir servis berkala pada {last_date}. Lakukan servis berkala setiap 6 bulan."
        else:
            return f"Terakhir servis pada {last_date} dengan jenis: {', '.join(last_types)}. Silakan cek buku servis untuk jadwal berikutnya."

# --- Sidebar CSS + Navigasi ---
st.markdown("""
<style>
.sidebar-button {
  width:100%; padding:0.75rem 1rem; margin-bottom:0.5rem; font-size:1.1rem;
  text-align:left; background-color:#4CAF50; color:white; border:none;
  border-radius:5px; cursor:pointer; transition:background-color .3s;
}
.sidebar-button:hover { background-color:#45a049; }
</style>
""", unsafe_allow_html=True)

# --- Halaman Aktif ---
if "page" not in st.session_state:
    st.session_state.page = "Chatbot"

st.sidebar.title("\U0001F697 Carbot Navigation")
if st.sidebar.button("Chatbot", key="btn_chatbot"): st.session_state.page = "Chatbot"
if st.sidebar.button("File Docs", key="btn_docs"): st.session_state.page = "File Docs"
if st.sidebar.button("Booking Service", key="btn_booking"): st.session_state.page = "Booking"
if st.sidebar.button("Riwayat Booking", key="btn_history"): st.session_state.page = "Riwayat"

# --- Upload File ---
st.sidebar.markdown("### \U0001F4C4 Upload File ke 'docs'")
uploaded_file = st.sidebar.file_uploader("Pilih file (pdf/txt/docx)", type=["pdf", "txt", "docx"])
if uploaded_file:
    docs_folder = Path("./docs"); docs_folder.mkdir(exist_ok=True)
    save_path = docs_folder / uploaded_file.name
    save_path.write_bytes(uploaded_file.getbuffer())
    st.sidebar.success(f"\U0001F4C4 File '{uploaded_file.name}' berhasil diunggah ke folder docs.")

# === Halaman Chatbot ===
if st.session_state.page == "Chatbot":
    st.title("Toyota Car Price Chatbot")
    chatbot = Chatbot()
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Halo! \U0001F44B Saya siap bantu info harga & perawatan Toyota."}
        ]
    if "awaiting_plat" not in st.session_state:
        st.session_state.awaiting_plat = False
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    chatbot.set_chat_history(st.session_state.messages)
    if prompt := st.chat_input("Apa yang ingin Anda tanyakan?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        # Deteksi permintaan saran service
        if "saran service" in prompt.lower() or "service selanjutnya" in prompt.lower():
            st.session_state.awaiting_plat = True
            st.session_state.messages.append({"role": "assistant", "content": "Silakan masukkan plat nomor untuk saran service:"})
        else:
            with st.chat_message("assistant"):
                resp = chatbot.chat_engine.chat(prompt)
                st.markdown(resp.response)
            st.session_state.messages.append({"role": "assistant", "content": resp.response})
    if st.session_state.awaiting_plat:
        plat_nomor = st.text_input("Masukkan plat nomor untuk saran service:")
        if plat_nomor:
            suggestion = chatbot.get_service_suggestion(plat_nomor)
            st.session_state.messages.append({"role": "assistant", "content": suggestion})
            st.session_state.awaiting_plat = False

# === Halaman File Docs ===
elif st.session_state.page == "File Docs":
    st.title("\U0001F4C1 File dalam Folder 'docs'")
    docs_path = Path("./docs")
    if docs_path.exists():
        files = [f for f in docs_path.glob("**/*") if f.is_file()]
        if files:
            selected = st.selectbox("Pilih file", [f.name for f in files])
            full = docs_path / selected
            if selected.lower().endswith(".pdf"):
                try:
                    with pdfplumber.open(full) as pdf:
                        text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                    st.text_area("Isi PDF", text, height=400)
                except Exception as e:
                    st.error(f"Gagal membaca PDF: {e}")
            else:
                try:
                    txt = full.read_text(encoding="utf-8", errors="ignore")
                    st.code(txt[:1000])
                except Exception as e:
                    st.error(f"Gagal membaca file: {e}")
        else:
            st.info("Belum ada file di folder 'docs'.")
    else:
        st.error("Folder docs belum tersedia.")

# === Halaman Booking Service ===
elif st.session_state.page == "Booking":
    st.title("\U0001F6E0️ Booking Servis Mobil Toyota")
    with st.form("booking_form"):
        nama = st.text_input("Nama Lengkap")
        plat_nomor = st.text_input("Nomor Polisi")
        tipe_mobil = st.selectbox("Tipe Mobil", ["Avanza", "Rush", "Yaris", "Innova", "Fortuner", "Lainnya"])
        lokasi = st.text_input("Lokasi Dealer")
        tanggal = st.date_input("Tanggal Booking")
        waktu = st.time_input("Waktu Booking")
        jenis_servis = st.multiselect("Jenis Servis", ["Servis Berkala", "Ganti Oli", "Pemeriksaan Rem", "Servis AC", "Lainnya"])
        keluhan = st.text_area("Keluhan / Masalah Mobil (Opsional)", placeholder="Contoh: Suara berisik saat mesin dinyalakan...")

        submitted = st.form_submit_button("Kirim Booking")
        if submitted:
            db.collection("bookings").add({
                "nama": nama,
                "plat_nomor": plat_nomor,
                "tipe_mobil": tipe_mobil,
                "lokasi": lokasi,
                "tanggal": tanggal.strftime("%Y-%m-%d"),
                "waktu": waktu.strftime("%H:%M:%S"),
                "jenis_servis": jenis_servis,
                "keluhan": keluhan,
                "timestamp": datetime.utcnow()
            })
            st.success(f"✅ Booking untuk {nama} (Plat {plat_nomor}) berhasil dikirim!")

# === Halaman Riwayat Booking ===
elif st.session_state.page == "Riwayat":
    st.title("\U0001F4CB Riwayat Booking Service")
    docs = db.collection("bookings").stream()
    plat_list = sorted({doc.to_dict().get("plat_nomor", "") for doc in docs})
    if plat_list:
        selected = st.selectbox("Pilih Plat Nomor", plat_list)
        if selected:
            hist = (db.collection("bookings")
                    .where("plat_nomor", "==", selected)
                    .order_by("timestamp", direction=firestore.Query.DESCENDING)
                    .stream())
            records = [r.to_dict() for r in hist]
            if records:
                for i, rec in enumerate(records, 1):
                    st.markdown(f"### Booking #{i}")
                    st.write(f"**Nama:** {rec['nama']}")
                    st.write(f"**Tipe Mobil:** {rec['tipe_mobil']}")
                    st.write(f"**Dealer:** {rec['lokasi']}")
                    st.write(f"**Tanggal & Waktu:** {rec['tanggal']} {rec['waktu']}")
                    st.write(f"**Jenis Servis:** {', '.join(rec['jenis_servis'])}")
                    st.write(f"**Keluhan:** {rec.get('keluhan', '-') or '-'}")
                    st.write("---")
            else:
                st.info("Belum ada booking untuk plat nomor ini.")
    else:
        st.info("Belum ada data booking tersimpan.")
