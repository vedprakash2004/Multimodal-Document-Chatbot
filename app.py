import streamlit as st

from config import configure_page
from input_processing import process_files
from rag import create_rag, retrieve_context
from generation import (
    generate_answer,
    generate_image,
    generate_audio
)


# =========================================================
# PAGE CONFIG
# =========================================================

configure_page()


# =========================================================
# SESSION STATE
# =========================================================

if "documents" not in st.session_state:
    st.session_state.documents = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed" not in st.session_state:
    st.session_state.processed = False

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""


# =========================================================
# TITLE
# =========================================================

st.title("🤖 Multimodal RAG Q&A")

st.write(
    "Upload PDF, text, audio or images and ask questions "
    "from your uploaded content."
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📂 Upload Content")

    pdf_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    text_files = st.file_uploader(
        "Upload Text / CSV / Markdown",
        type=["txt", "csv", "md"],
        accept_multiple_files=True
    )

    audio_files = st.file_uploader(
        "Upload Audio",
        type=["mp3", "wav", "m4a", "mpeg"],
        accept_multiple_files=True
    )

    image_files = st.file_uploader(
        "Upload Images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )

    st.divider()

    # -----------------------------------------------------
    # CLEAR CHAT
    # -----------------------------------------------------

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.session_state.last_answer = ""

        st.rerun()


# =========================================================
# DIRECT TEXT
# =========================================================

st.subheader("✍️ Direct Text")

text_input = st.text_area(
    "Enter text",
    height=150,
    placeholder="Enter text for your RAG system..."
)


# =========================================================
# PROCESS CONTENT
# =========================================================

if st.button(
    "🚀 Process Content",
    use_container_width=True
):

    with st.spinner("Processing content..."):

        try:

            documents, processed_info = process_files(

                pdf_files or [],

                text_files or [],

                audio_files or [],

                image_files or [],

                text_input
            )

            if not documents:

                st.warning(
                    "⚠️ Please upload a file or enter text."
                )

            else:

                st.session_state.documents = documents

                st.session_state.vector_db = create_rag(
                    documents
                )

                st.session_state.processed = True

                # New content means new conversation context
                st.session_state.chat_history = []

                st.session_state.last_answer = ""

                st.success(
                    f"✅ {len(documents)} document(s) processed"
                )

                for info in processed_info:

                    st.write(info)

        except Exception as e:

            st.error(
                f"❌ Processing error: {e}"
            )


# =========================================================
# DOCUMENT STATUS
# =========================================================

if st.session_state.documents:

    st.divider()

    st.subheader("📚 Loaded Content")

    st.info(
        f"Documents loaded: "
        f"{len(st.session_state.documents)}"
    )


# =========================================================
# CHAT HISTORY
# =========================================================

st.divider()

st.subheader("💬 Chat")


# Display old messages first

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# =========================================================
# CHAT INPUT
# =========================================================

if st.session_state.vector_db is not None:

    question = st.chat_input(
        "Ask something about your uploaded content..."
    )

    if question:

        # -------------------------------------------------
        # USER MESSAGE
        # -------------------------------------------------

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )


        # Show user message immediately

        with st.chat_message("user"):

            st.markdown(question)


        # -------------------------------------------------
        # GENERATE ANSWER
        # -------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching your content..."
            ):

                try:

                    context = retrieve_context(
                        st.session_state.vector_db,
                        question
                    )

                    answer = generate_answer(
                        context,
                        question
                    )

                    st.markdown(answer)

                    # Save answer

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                    st.session_state.last_answer = answer

                except Exception as e:

                    error_message = (
                        f"❌ Error: {e}"
                    )

                    st.error(error_message)

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": error_message
                        }
                    )


else:

    st.info(
        "👆 First upload content and click "
        "**Process Content**."
    )


# =========================================================
# ADDITIONAL OUTPUT
# =========================================================

if st.session_state.last_answer:

    st.divider()

    st.subheader(
        "🎨 Generate Additional Output"
    )

    col1, col2 = st.columns(2)


    # =====================================================
    # IMAGE
    # =====================================================

    with col1:

        if st.button(
            "🖼️ Generate Educational Image",
            use_container_width=True
        ):

            with st.spinner(
                "Generating image..."
            ):

                try:

                    image_bytes = generate_image(
                        st.session_state.last_answer
                    )

                    st.image(
                        image_bytes,
                        caption="Generated Educational Image",
                        use_container_width=True
                    )

                except Exception as e:

                    st.error(
                        f"❌ Image error: {e}"
                    )


    # =====================================================
    # AUDIO
    # =====================================================

    with col2:

        if st.button(
            "🔊 Generate Audio Answer",
            use_container_width=True
        ):

            with st.spinner(
                "Generating audio..."
            ):

                try:

                    audio_bytes = generate_audio(
                        st.session_state.last_answer
                    )

                    st.audio(
                        audio_bytes,
                        format="audio/wav"
                    )

                except Exception as e:

                    st.error(
                        f"❌ Audio error: {e}"
                    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🤖 Multimodal RAG Q&A | "
    "LangChain + HuggingFace + Groq + OpenAI"
)