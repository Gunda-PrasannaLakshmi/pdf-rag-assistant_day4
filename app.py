import streamlit as st
import pymupdf
import numpy as np
import faiss
import os

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from groq import Groq


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Assistant")
st.write(
    "Upload a PDF and ask questions about its contents."
)


# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY was not found in the .env FILE."
    )
    st.stop()


# --------------------------------------------------
# GROQ CLIENT
# --------------------------------------------------

client = Groq(
    api_key=GROQ_API_KEY
)


# --------------------------------------------------
# EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("⚙️ Settings")

chunk_size = st.sidebar.slider(
    "Chunk Size",
    min_value=200,
    max_value=2000,
    value=1000,
    step=100
)

chunk_overlap = st.sidebar.slider(
    "Chunk Overlap",
    min_value=0,
    max_value=500,
    value=200,
    step=50
)

top_k = st.sidebar.slider(
    "Top-K Results",
    min_value=1,
    max_value=10,
    value=3
)


# --------------------------------------------------
# CLEAR CHAT
# --------------------------------------------------

if st.sidebar.button("🗑️ Clear Chat"):

    st.session_state.messages = []

    st.rerun()


# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


if uploaded_file is None:

    st.info(
        "Please upload a PDF to start asking questions."
    )

    st.stop()


# --------------------------------------------------
# EXTRACT PDF TEXT
# --------------------------------------------------

try:

    pdf_bytes = uploaded_file.read()

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    if len(document) == 0:

        st.error("The PDF is empty.")

        st.stop()


    documents_data = []


    for page_number, page in enumerate(document):

        text = page.get_text().strip()

        if text:

            documents_data.append({

                "document_name":
                    uploaded_file.name,

                "page_number":
                    page_number + 1,

                "text":
                    text
            })


    if not documents_data:

        st.warning(
            "No readable text was found in this PDF."
        )

        st.stop()


    st.success(
        f"Successfully extracted text from "
        f"{len(documents_data)} page(s)."
    )


    # --------------------------------------------------
    # CREATE CHUNKS
    # --------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=chunk_size,

        chunk_overlap=chunk_overlap,

        separators=[
            "\n\n",
            "\n",
            " ",
            ""
        ]
    )


    chunks = []


    for item in documents_data:

        split_texts = text_splitter.split_text(
            item["text"]
        )


        for chunk_number, chunk_text in enumerate(
            split_texts
        ):

            chunks.append({

                "document_name":
                    item["document_name"],

                "page_number":
                    item["page_number"],

                "chunk_number":
                    chunk_number + 1,

                "text":
                    chunk_text
            })


    # --------------------------------------------------
    # CREATE EMBEDDINGS
    # --------------------------------------------------

    chunk_texts = [

        chunk["text"]

        for chunk in chunks
    ]


    embeddings = embedding_model.encode(

        chunk_texts,

        show_progress_bar=False
    )


    embeddings = np.array(
        embeddings
    ).astype("float32")


    # --------------------------------------------------
    # CREATE FAISS INDEX
    # --------------------------------------------------

    embedding_dimension = embeddings.shape[1]


    faiss_index = faiss.IndexFlatL2(
        embedding_dimension
    )


    faiss_index.add(
        embeddings
    )


    # --------------------------------------------------
    # DISPLAY PDF INFORMATION
    # --------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Pages",
            len(documents_data)
        )


    with col2:

        st.metric(
            "Chunks",
            len(chunks)
        )


    with col3:

        st.metric(
            "Vectors",
            faiss_index.ntotal
        )


except Exception as e:

    st.error(
        f"Unable to process the PDF: {e}"
    )

    st.stop()


# --------------------------------------------------
# DISPLAY PREVIOUS CHAT MESSAGES
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

user_question = st.chat_input(
    "Ask a question about your PDF..."
)


if user_question:

    # --------------------------------------------------
    # DISPLAY USER QUESTION
    # --------------------------------------------------

    with st.chat_message("user"):

        st.markdown(
            user_question
        )


    # Save user question

    st.session_state.messages.append({

        "role":
            "user",

        "content":
            user_question
    })


    # --------------------------------------------------
    # CREATE QUESTION EMBEDDING
    # --------------------------------------------------

    query_embedding = embedding_model.encode(

        [user_question]
    )


    query_embedding = np.array(
        query_embedding
    ).astype("float32")


    # --------------------------------------------------
    # SEARCH FAISS
    # --------------------------------------------------

    actual_top_k = min(
        top_k,
        len(chunks)
    )


    distances, indices = faiss_index.search(

        query_embedding,

        actual_top_k
    )


    # --------------------------------------------------
    # GET RELEVANT CHUNKS
    # --------------------------------------------------

    retrieved_chunks = []


    for index in indices[0]:

        if index != -1:

            retrieved_chunks.append(
                chunks[index]
            )


    # --------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------

    context_parts = []


    for chunk in retrieved_chunks:

        context_parts.append(

            f"Document: "
            f"{chunk['document_name']}\n"

            f"Page: "
            f"{chunk['page_number']}\n"

            f"Content:\n"
            f"{chunk['text']}"
        )


    context = "\n\n---\n\n".join(
        context_parts
    )


    # --------------------------------------------------
    # GROQ PROMPT
    # --------------------------------------------------

    system_prompt = """
You are a PDF question-answering assistant.

Use ONLY the information contained in the
provided PDF context to answer the user's question.

Do not use outside information.

Do not invent or guess answers.

If the answer is not present in the provided
context, say:

"I could not find this information in the PDF."

Give a clear and concise answer.
"""


    user_prompt = f"""
PDF CONTEXT:

{context}


USER QUESTION:

{user_question}


Answer using only the PDF context.
"""


    # --------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the PDF and generating an answer..."
        ):

            try:

                response = client.chat.completions.create(

                    model="openai/gpt-oss-20b",

                    messages=[

                        {
                            "role":
                                "system",

                            "content":
                                system_prompt
                        },

                        {
                            "role":
                                "user",

                            "content":
                                user_prompt
                        }
                    ],

                    temperature=0
                )


                answer = (

                    response
                    .choices[0]
                    .message
                    .content
                )


                st.markdown(
                    answer
                )


                # --------------------------------------------------
                # SOURCES
                # --------------------------------------------------

                st.markdown(
                    "### 📚 Sources"
                )


                for rank, chunk in enumerate(
                    retrieved_chunks,
                    start=1
                ):

                    with st.expander(

                        f"Source {rank} — "
                        f"{chunk['document_name']} "
                        f"(Page {chunk['page_number']})"
                    ):

                        st.write(
                            f"**Document:** "
                            f"{chunk['document_name']}"
                        )

                        st.write(
                            f"**Page:** "
                            f"{chunk['page_number']}"
                        )

                        st.write(
                            f"**Chunk:** "
                            f"{chunk['chunk_number']}"
                        )

                        st.write(
                            chunk["text"]
                        )


                # Save assistant answer

                st.session_state.messages.append({

                    "role":
                        "assistant",

                    "content":
                        answer
                })


            except Exception as e:

                error_message = (
                    f"Groq API error: {e}"
                )

                st.error(
                    error_message
                )


# --------------------------------------------------
# INFORMATION SECTION
# --------------------------------------------------

with st.expander(
    "ℹ️ How this RAG system works"
):

    st.write(
        """
        1. PDF text is extracted using PyMuPDF.

        2. The text is divided into smaller chunks.

        3. Sentence Transformers converts chunks
           into embeddings.

        4. FAISS stores the embeddings.

        5. Your question is converted into an
           embedding.

        6. FAISS retrieves the most relevant chunks.

        7. Groq generates an answer using the
           retrieved PDF context.

        8. Source document and page information
           are displayed with the answer.
        """
    )