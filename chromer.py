import os
import io
import uuid
import time
import streamlit as st
from urllib.parse import urlparse

# ---- Optional dependencies for parsing & embeddings ----
try:
    import chromadb
    from chromadb import HttpClient
    from chromadb.utils import embedding_functions
except Exception as e:
    st.error(f"Gagal mengimpor chromadb. Pastikan sudah terpasang. Error: {e}")
    st.stop()

try:
    import tiktoken
except Exception:
    tiktoken = None

try:
    import docx
except Exception:
    docx = None

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# OpenAI (for answering from retrieved context)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(page_title="Chroma Uploader + RAG Chat", page_icon="📚", layout="wide")

st.title("📚 Chroma Uploader + RAG Chat")
st.caption("Upload dokumen → simpan ke Chroma → tanya dokumen dengan sitasi")

# ---------------- Sidebar: Credentials & Settings ----------------
with st.sidebar:
    st.header("🔐 Koneksi Chroma")
    chroma_mode = st.radio("Mode", ["Chroma Cloud (HTTP)", "Local (Persistent)"], index=0)
    if chroma_mode == "Chroma Cloud (HTTP)":
        host = st.text_input(
            "Host",
            value=os.getenv("CHROMA_HOST", ""),
            placeholder="https://your-cluster.region.trychroma.com",
            help="Salin 'Cluster Endpoint' dari dashboard Chroma Cloud Anda."
        )
        tenant = st.text_input("Tenant (UUID / name)", value=os.getenv("CHROMA_TENANT", "default_tenant"))
        database = st.text_input("Database", value=os.getenv("CHROMA_DATABASE", "default_database"))
        chroma_api_key = st.text_input("Chroma API Key", type="password", value=os.getenv("CHROMA_API_KEY", ""))
    else:
        persist_dir = st.text_input("Persist Directory", value="./chroma_data")
        host = tenant = database = chroma_api_key = None

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
    if not text:
        return []
    # Prefer tiktoken-based chunking by token count if available
    if tiktoken is not None:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            toks = enc.encode(text)
            chunks = []
            i = 0
            # approx chars per token ~ 4
            tok_size = max(50, size // 4)
            tok_overlap = max(0, overlap // 4)
            while i < len(toks):
                chunk = enc.decode(toks[i:i+tok_size])
                chunks.append(chunk)
                i += max(1, tok_size - tok_overlap)
            return chunks
        except Exception: # Fallback if encoding fails
            pass
    # Fallback by characters
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += max(1, size - overlap)
    return chunks

def read_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".txt") or name.endswith(".md"):
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        if PyPDF2 is None:
            raise RuntimeError("PyPDF2 belum terpasang (tambahkan ke requirements).")
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(pages)
    if name.endswith(".docx"):
        if docx is None:
            raise RuntimeError("python-docx belum terpasang (tambahkan ke requirements).")
        file_obj = io.BytesIO(data)
        d = docx.Document(file_obj)
        return "\n".join([p.text for p in d.paragraphs])
    # default: treat as text
    return data.decode("utf-8", errors="ignore")

@st.cache_resource(show_spinner=False)
def get_chroma_client():
    if chroma_mode == "Chroma Cloud (HTTP)":
        if not (host and chroma_api_key and tenant and database):
            st.error("Lengkapi Host, Tenant, Database, dan Chroma API Key.")
            st.stop()
        try:
            # FIX: The API key must be passed in the headers.
            headers = {"Authorization": f"Bearer {chroma_api_key}"}

            # FIX: HttpClient expects host without protocol; port and ssl are separate.
            parsed_uri = urlparse(host)
            cleaned_host = parsed_uri.hostname
            port = parsed_uri.port
            ssl = parsed_uri.scheme == 'https'

            if not port:
                port = 443 if ssl else 80

            if not cleaned_host:
                 st.error("Format Host tidak valid. Harusnya seperti 'https://your-cluster.region.trychroma.com'")
                 st.stop()

            client = HttpClient(
                host=cleaned_host,
                port=port,
                ssl=ssl,
                headers=headers,
                tenant=tenant,
                database=database
            )
            _ = client.list_collections() # Quick ping to check connection
            return client
        except Exception as e:
            st.error(f"Gagal konek ke Chroma Cloud: {e}")
            st.stop()
    else: # Local Persistent
        try:
            client = chromadb.PersistentClient(path=persist_dir)
            _ = client.list_collections()
            return client
        except Exception as e:
            st.error(f"Gagal membuat PersistentClient: {e}")
            st.stop()

@st.cache_resource(show_spinner=False)
def get_embedding_function():
    if embed_choice == "OpenAIEmbeddings":
        if not openai_api_key:
            st.error("OPENAI_API_KEY diperlukan untuk OpenAIEmbeddings.")
            st.stop()
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            model_name="text-embedding-3-small"
        )
    else: # Sentence-Transformers
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

def get_or_create_collection():
    client = get_chroma_client()
    emb_func = get_embedding_function()
    # The embedding_function is passed on creation and not needed for get.
    # This logic correctly handles getting an existing collection.
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=emb_func,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def list_sources(col, limit=1000, offset=0):
    try:
        res = col.get(include=["metadatas"], limit=limit, offset=offset)
        metas = res.get("metadatas", []) or []
        sources = [m.get("source","unknown") for m in metas if isinstance(m, dict)]
        return sources, len(metas)
    except Exception as e:
        st.warning(f"Gagal mengambil daftar dokumen: {e}")
        return [], 0

def build_prompt(question, results):
    numbered = []
    for i, (doc, meta) in enumerate(results, start=1):
        src = meta.get("source", "unknown")
        chunk_idx = meta.get("chunk", "?")
        numbered.append(f"[{i}] Source: {src} (chunk {chunk_idx}) — {doc.strip()}")
    context = "\n\n".join(numbered)
    system = "Anda adalah asisten yang menjawab hanya dari konteks berikut. Berikan jawaban ringkas dan tambahkan sitasi [n] pada klaim penting."
    user = f"Pertanyaan: {question}\n\nKonteks:\n{context}\n\nInstruksi: Jawab ringkas, lalu daftar sumber yang dirujuk."
    return system, user

def openai_answer(system_msg, user_msg):
    if OpenAI is None:
        st.error("Paket openai tidak tersedia. Tambahkan ke requirements.")
        st.stop()
    if not openai_api_key:
        st.error("OPENAI_API_KEY kosong.")
        st.stop()
    client = OpenAI(api_key=openai_api_key)
    try:
        resp = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role":"system","content":system_msg},
                {"role":"user","content":user_msg},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"Gagal memanggil OpenAI API: {e}")
        return None

# ---------------- Tabs ----------------
tab_up, tab_list, tab_chat = st.tabs(["⬆️ Upload to Chroma", "📄 List Dokumen", "💬 Chat RAG"])

with tab_up:
    st.subheader("Upload Dokumen")
    uploader = st.file_uploader("Pilih file (.pdf, .docx, .txt)", accept_multiple_files=True, type=["pdf","docx","txt","md"])
    if uploader and st.button("🚀 Upload ke Chroma"):
        collection = get_or_create_collection()
        with st.spinner("Memproses & mengunggah..."):
            total_chunks = 0
            for f in uploader:
                try:
                    text = read_file(f)
                    chunks = chunk_text(text, size=chunk_size, overlap=chunk_overlap)
                    if not chunks:
                        st.warning(f"File {f.name} tidak menghasilkan chunk apapun.")
                        continue
                    ids = [f"{f.name}-{i}-{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
                    metadatas = [{"source": f.name, "chunk": i, "uploaded_at": int(time.time())} for i in range(len(chunks))]
                    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
                    total_chunks += len(chunks)
                except Exception as e:
                    st.error(f"Gagal upload {f.name}: {e}")
            if total_chunks > 0:
                st.success(f"Selesai. Total chunks diunggah: {total_chunks}")

with tab_list:
    st.subheader("Daftar Dokumen (berdasarkan metadata 'source')")
    if st.button("🔄 Refresh Daftar Dokumen"):
        collection = get_or_create_collection()
        unique_sources = set()
        offset = 0
        batch = 1000
        total_docs = 0
        with st.spinner("Mengambil daftar..."):
            while True:
                sources, got = list_sources(collection, limit=batch, offset=offset)
                if got == 0:
                    break
                unique_sources.update(sources)
                total_docs += got
                offset += got
                if got < batch:
                    break
        st.write(f"Total entri dalam koleksi: {total_docs}")
        st.write(f"Dokumen unik (berdasarkan nama file): {len(unique_sources)}")
        if unique_sources:
            st.dataframe(sorted(list(unique_sources)), use_container_width=True)
        else:
            st.info("Belum ada dokumen yang diunggah ke koleksi ini.")


with tab_chat:
    st.subheader("Chat ke Dokumen")
    question = st.text_input("Pertanyaan")
    if st.button("Tanya") and question.strip():
        collection = get_or_create_collection()
        with st.spinner("Mengambil konteks dari Chroma..."):
            qres = collection.query(query_texts=[question], n_results=top_k, include=["documents","metadatas"])

        docs = (qres.get("documents") or [[]])[0]
        metas = (qres.get("metadatas") or [[]])[0]

        if not docs:
            st.warning("Tidak ada hasil relevan dari Chroma.")
        else:
            pairs = list(zip(docs, metas))
            system_msg, user_msg = build_prompt(question, pairs)
            with st.spinner("Menyusun jawaban..."):
                answer = openai_answer(system_msg, user_msg)

            if answer:
                st.markdown("### 🧾 Jawaban")
                st.write(answer)

                st.markdown("### 📚 Sumber")
                for i, (_, m) in enumerate(pairs, start=1):
                    src = m.get("source","unknown")
                    ch = m.get("chunk","?")
                    st.write(f"[{i}] {src} (chunk {ch})")
