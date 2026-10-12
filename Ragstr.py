
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)

PDF_FILE = BASE_DIR / "global-capital-markets-paper.pdf"
INDEX_DIR = BASE_DIR / "faiss_index"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4

st.set_page_config(
    page_title="Global Capital Markets RAG",
    page_icon="📈",
    layout="wide",
)


# ==================================================
# 2. CREATE EMBEDDING MODEL
# ==================================================

def get_embeddings():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=API_KEY,
    )


# ==================================================
# 3. LOAD AND SPLIT THE PDF
# ==================================================

def load_and_split_pdf():
    if not PDF_FILE.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_FILE}\n"
            "Place global-capital-markets-paper.pdf "
            "in the same folder as app.py."
        )

    loader = PyPDFLoader(str(PDF_FILE))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "No text found in the PDF. "
            "Scanned PDFs may require OCR."
        )

    return documents, chunks


# ==================================================
# 4. CREATE AND SAVE THE FAISS VECTOR DATABASE
# ==================================================

def build_vector_database():
    documents, chunks = load_and_split_pdf()

    embeddings = get_embeddings()

    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_db.save_local(str(INDEX_DIR))

    return vector_db, len(documents), len(chunks)


# ==================================================
# 5. LOAD THE EXISTING LOCAL FAISS DATABASE
# ==================================================

def load_vector_database():
    if not (
        (INDEX_DIR / "index.faiss").exists()
        and (INDEX_DIR / "index.pkl").exists()
    ):
        raise FileNotFoundError(
            "FAISS index files are missing. Rebuild the index."
        )

    # Only load indexes you created or trust.
    return FAISS.load_local(
        str(INDEX_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


# ==================================================
# 6. ASK A QUESTION USING RAG
# ==================================================

def get_answer(vector_db, question):
    retriever = vector_db.as_retriever(
        search_kwargs={"k": TOP_K}
    )

    relevant_docs = retriever.invoke(question)

    if not relevant_docs:
        return (
            "I could not find relevant content in the PDF.",
            [],
        )

    context_parts = []

    for i, doc in enumerate(relevant_docs, start=1):
        page = doc.metadata.get("page")
        page_text = (
            f"Page {page + 1}" if page is not None
            else "Page unknown"
        )

        context_parts.append(
            f"[Source {i}: {PDF_FILE.name}, {page_text}]\n"
            f"{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_template(
        """
        You are an assistant specializing in global capital markets.

        Answer the user's question using the provided PDF context.

        Rules:
        1. Use the supplied context to answer.
        2. Do not invent facts or unsupported conclusions.
        3. If the context does not answer the question,
           clearly say so.
        4. Explain financial concepts clearly.
        5. Cite page numbers in your answer when useful.

        PDF context:
        {context}

        User question:
        {question}

        Answer:
        """
    )

    llm = ChatOpenAI(
        model=CHAT_MODEL,
        api_key=API_KEY,
        temperature=0,
    )

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question,
    })

    # Collect source information for the frontend.
    sources = []
    seen = set()

    for doc in relevant_docs:
        page = doc.metadata.get("page")
        source = doc.metadata.get("source", PDF_FILE.name)

        key = (source, page)

        if key not in seen:
            sources.append({
                "file": Path(source).name,
                "page": page + 1 if page is not None else None,
                "text": doc.page_content,
            })
            seen.add(key)

    return answer, sources


# ==================================================
# 7. SIDEBAR
# ==================================================

st.sidebar.title("⚙️ Settings")

st.sidebar.markdown("### Document")

if PDF_FILE.exists():
    st.sidebar.success("PDF found")
    st.sidebar.write(f"**File:** {PDF_FILE.name}")
else:
    st.sidebar.error("PDF file not found")

st.sidebar.markdown("### Vector database")

st.sidebar.caption(f"Storage: `{INDEX_DIR.name}/`")
st.sidebar.caption(f"Embedding model: `{EMBEDDING_MODEL}`")
st.sidebar.caption(f"Chat model: `{CHAT_MODEL}`")
st.sidebar.caption(f"Retrieved chunks per question: {TOP_K}")

if st.sidebar.button(
    "🔄 Rebuild FAISS index",
    use_container_width=True,
):
    if not API_KEY:
        st.sidebar.error("Configure OPENAI_API_KEY in .env.")
    else:
        try:
            with st.spinner("Rebuilding the vector database..."):
                vector_db, page_count, chunk_count = (
                    build_vector_database()
                )

            st.session_state["vector_db"] = vector_db

            st.sidebar.success(
                f"Index rebuilt: {page_count} pages, "
                f"{chunk_count} chunks."
            )
        except Exception as error:
            st.sidebar.error(f"Could not rebuild index: {error}")

if st.sidebar.button("🗑️ Clear chat", use_container_width=True):
    st.session_state["messages"] = []
    st.rerun()


# ==================================================
# 8. MAIN FRONTEND
# ==================================================

st.title("📈 Global Capital Markets RAG")

st.markdown(
    """
    Ask questions about your capital markets paper.
    Answers are generated using relevant passages retrieved
    from your PDF and the OpenAI language model.
    """
)

st.divider()


# ==================================================
# 9. VALIDATE API KEY AND INITIALIZE VECTOR DATABASE
# ==================================================

if not API_KEY:
    st.error(
        "OPENAI_API_KEY is missing. Add your API key to the "
        ".env file next to app.py, then restart Streamlit."
    )
    st.stop()

if not PDF_FILE.exists():
    st.error(
        f"Cannot find `{PDF_FILE.name}`. "
        "Place the PDF next to app.py."
    )
    st.stop()

if "vector_db" not in st.session_state:
    try:
        index_exists = (
            (INDEX_DIR / "index.faiss").exists()
            and (INDEX_DIR / "index.pkl").exists()
        )

        if index_exists:
            with st.spinner("Loading local FAISS database..."):
                st.session_state["vector_db"] = (
                    load_vector_database()
                )
        else:
            with st.spinner(
                "First run: reading PDF and creating FAISS index..."
            ):
                vector_db, page_count, chunk_count = (
                    build_vector_database()
                )

                st.session_state["vector_db"] = vector_db

            st.success(
                f"Indexed {page_count} pages into "
                f"{chunk_count} chunks."
            )

    except Exception as error:
        st.error(f"Initialization failed: {error}")
        st.info(
            "Check your API key, package installation, "
            "PDF file, and internet connection."
        )
        st.stop()


# ==================================================
# 10. CHAT HISTORY
# ==================================================

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("📚 View source pages"):
                for source in message["sources"]:
                    page = source["page"]

                    page_label = (
                        f"Page {page}" if page is not None
                        else "Page unknown"
                    )

                    st.markdown(
                        f"**{source['file']} — {page_label}**"
                    )
                    st.caption(source["text"])


# ==================================================
# 11. CHAT INPUT AND RESPONSE
# ==================================================

question = st.chat_input(
    "Ask a question about global capital markets..."
)

if question:
    st.session_state["messages"].append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the PDF and generating an answer..."):
                answer, sources = get_answer(
                    st.session_state["vector_db"],
                    question,
                )

            st.markdown(answer)

            if sources:
                with st.expander("📚 View source pages"):
                    for source in sources:
                        page = source["page"]

                        page_label = (
                            f"Page {page}" if page is not None
                            else "Page unknown"
                        )

                        st.markdown(
                            f"**{source['file']} — {page_label}**"
                        )
                        st.caption(source["text"])

            st.session_state["messages"].append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })

        except Exception as error:
            error_message = f"Unable to generate an answer: {error}"
            st.error(error_message)

            st.session_state["messages"].append({
                "role": "assistant",
                "content": error_message,
                "sources": [],
            })
