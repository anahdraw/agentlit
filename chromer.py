import os
import io
import uuid
import time
import streamlit as st

# ---- Dependencies ----
try:
    import chromadb
    # PERUBAHAN: Impor CloudClient, bukan hanya HttpClient
    from chromadb import CloudClient
    from chromadb.utils import embedding_functions
except Exception as e:
    st.error(f"Gagal mengimpor chromadb. Pastikan sudah terpasang. Error: {e}")
    st.stop()

try:
    import tiktoken
except Exception: tiktoken = None
try:
    import docx
except Exception: docx = None
try:
    import PyPDF2
except Exception: PyPDF2 = None
try:
    from openai import OpenAI
except Exception: OpenAI = None

st.set_page_config(page_title="Chroma Uploader + RAG Chat", page_icon="📚", layout="wide")

st.title("📚 Chroma Uploader + RAG Chat")
st.caption("Upload dokumen → simpan ke Chroma → tanya dokumen dengan sitasi")

# ---------------- Sidebar: Credentials & Settings ----------------
with st.sidebar:
    st.header("🔐 Koneksi Chroma")
    chroma_mode = st.radio("Mode", ["Chroma Cloud", "Local (Persistent)"], index=0)
    if chroma_mode == "Chroma Cloud":
        st.info("Salin kredensial dari halaman 'Connect' database Anda di Chroma Cloud.")
        # PERUBAHAN: Menghapus input Host yang tidak lagi diperlukan
        tenant = st.text_input("Tenant", value="", help="Salin dari halaman koneksi database Anda.")
        database = st.text_input("Database", value="n8nsmallcr", help="Salin dari halaman koneksi database Anda.")
        chroma_api_key = st.text_input("Chroma API Key", type="password", help="Buat dengan tombol 'Create API key'.")
    else:
        persist_dir = st.text_input("Persist Directory", value="./chroma_data")
        tenant = database = chroma_api_key = None

    st.divider()
    st.header("🧠 Embedding Model")
    embed_choice = st.selectbox("Embedding function", ["OpenAIEmbeddings", "Sentence-Transformers (all-MiniLM-L6-v2)"], index=0)
    openai_api_key = st.text_input("OPENAI_API_KEY (untuk embeddings & jawaban)", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    openai_model = st.text_input("OpenAI Chat Model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    collection_name = st.text_input("Collection Name", value="docs")
    top_k = st.slider("Top-K retrieval", 1, 10, 5)
    chunk_size = st.slider("Chunk size (chars)", 300, 2000, 900, step=50)
    chunk_overlap = st.slider("Chunk overlap (chars)", 0, 400, 150, step=10)

# ---------------- Helpers ----------------
def chunk_text(text, size=900, overlap=150):
    if not text: return []
    if tiktoken:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            toks = enc.encode(text)
            chunks, i, tok_size, tok_overlap = [], 0, max(50, size // 4), max(0, overlap // 4)
            while i < len(toks):
                chunks.append(enc.decode(toks[i:i+tok_size]))
                i += max(1, tok_size - tok_overlap)
            return chunks
        except Exception: pass
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += max(1, size - overlap)
    return chunks

def read_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith((".txt", ".md")): return data.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        if PyPDF2 is None: raise RuntimeError("PyPDF2 belum terpasang.")
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        return "\n\n".join([p.extract_text() or "" for p in reader.pages])
    if name.endswith(".docx"):
        if docx is None: raise RuntimeError("python-docx belum terpasang.")
        return "\n".join([p.text for p in docx.Document(io.BytesIO(data)).paragraphs])
    return data.decode("utf-8", errors="ignore")

@st.cache_resource(show_spinner=False)
def get_chroma_client():
    if chroma_mode == "Chroma Cloud":
        if not (tenant and database and chroma_api_key):
            st.error("Lengkapi Tenant, Database, dan Chroma API Key.")
            st.stop()
        try:
            # PERUBAHAN BESAR: Menggunakan CloudClient, bukan HttpClient
            client = CloudClient(
                tenant=tenant,
                database=database,
                api_key=chroma_api_key
            )
            client.heartbeat() # Cek koneksi
            return client
        except Exception as e:
            st.error(f"Gagal konek ke Chroma Cloud: {e}")
            st.stop()
    else: # Local Persistent
        try:
            client = chromadb.PersistentClient(path=persist_dir)
            client.heartbeat()
            return client
        except Exception as e:
            st.error(f"Gagal membuat PersistentClient: {e}")
            st.stop()

@st.cache_resource(show_spinner=False)
def get_embedding_function():
    if embed_choice == "OpenAIEmbeddings":
        if not openai_api_key:
            st.error("OPENAI_API_KEY diperlukan.")
            st.stop()
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key, model_name="text-embedding-3-small"
        )
    else:
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def get_or_create_collection():
    client = get_chroma_client()
    emb_func = get_embedding_function()
    return client.get_or_create_collection(
        name=collection_name, embedding_function=emb_func, metadata={"hnsw:space": "cosine"}
    )

# Sisa kode (fungsi RAG, tabs) tidak perlu diubah secara signifikan
# ... (kode lainnya tetap sama) ...
def build_prompt(question, results):
    numbered = []
    for i, (doc, meta) in enumerate(results, start=1):
        src, chunk_idx = meta.get("source", "?"), meta.get("chunk", "?")
        numbered.append(f"[{i}] Source: {src} (chunk {chunk_idx}) — {doc.strip()}")
    context = "\n\n".join(numbered)
    system = "Anda adalah asisten yang menjawab hanya dari konteks berikut. Berikan jawaban ringkas dan tambahkan sitasi [n] pada klaim penting."
    user = f"Pertanyaan: {question}\n\nKonteks:\n{context}\n\nInstruksi: Jawab ringkas, lalu daftar sumber yang dirujuk."
    return system, user

def openai_answer(system_msg, user_msg):
    if OpenAI is None or not openai_api_key:
        st.error("OPENAI_API_KEY tidak tersedia/valid.")
        st.stop()
    client = OpenAI(api_key=openai_api_key)
    try:
        resp = client.chat.completions.create(
            model=openai_model, messages=[{"role":"system","content":system_msg}, {"role":"user","content":user_msg}],
            temperature=0.2)
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"Gagal memanggil OpenAI API: {e}")
        return None

tab_up, tab_list, tab_chat = st.tabs(["⬆️ Upload", "📄 List Dokumen", "💬 Chat"])

with tab_up:
    st.subheader("Upload Dokumen")
    uploader = st.file_uploader("Pilih file (.pdf, .docx, .txt, .md)", accept_multiple_files=True, type=["pdf","docx","txt","md"])
    if uploader and st.button("🚀 Upload ke Chroma"):
        collection = get_or_create_collection()
        with st.spinner("Memproses & mengunggah..."):
            total_chunks = 0
            for f in uploader:
                try:
                    text = read_file(f)
                    chunks = chunk_text(text, size=chunk_size, overlap=chunk_overlap)
                    if not chunks:
                        st.warning(f"File {f.name} tidak menghasilkan chunk.")
                        continue
                    ids = [f"{f.name}-{i}-{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
                    metadatas = [{"source": f.name, "chunk": i} for i in range(len(chunks))]
                    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
                    total_chunks += len(chunks)
                except Exception as e:
                    st.error(f"Gagal upload {f.name}: {e}")
            if total_chunks > 0:
                st.success(f"Selesai. Total chunks diunggah: {total_chunks}")

with tab_list:
    st.subheader("Daftar Dokumen")
    if st.button("🔄 Refresh Daftar"):
        collection = get_or_create_collection()
        count = collection.count()
        st.write(f"Total entri (chunks) dalam koleksi: {count}")
        if count > 0:
            with st.spinner("Mengambil daftar sumber..."):
                entries = collection.get(limit=count, include=["metadatas"])
                sources = sorted(list(set(m.get('source', 'unknown') for m in entries['metadatas'])))
                st.dataframe(sources, use_container_width=True)

with tab_chat:
    st.subheader("Tanya Dokumen Anda")
    question = st.text_input("Pertanyaan")
    if st.button("Kirim Pertanyaan") and question.strip():
        collection = get_or_create_collection()
        with st.spinner("Mengambil konteks dari Chroma..."):
            qres = collection.query(query_texts=[question], n_results=top_k, include=["documents", "metadatas"])
        docs = (qres.get("documents") or [[]])[0]
        metas = (qres.get("metadatas") or [[]])[0]
        if not docs:
            st.warning("Tidak ada hasil relevan ditemukan di dokumen.")
        else:
            pairs = list(zip(docs, metas))
            system_msg, user_msg = build_prompt(question, pairs)
            with st.spinner("Menyusun jawaban..."):
                answer = openai_answer(system_msg, user_msg)
            if answer:
                st.markdown("### 🧾 Jawaban")
                st.write(answer)
                st.markdown("### 📚 Sumber yang Digunakan")
                for i, (doc, m) in enumerate(pairs, start=1):
                    with st.expander(f"Sumber [{i}]: {m.get('source','?')} (chunk {m.get('chunk','?')})"):
                        st.write(doc)
