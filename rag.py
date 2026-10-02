from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_community.vectorstores import (
    InMemoryVectorStore
)


# ---------------------------------------------------------
# CREATE RAG
# ---------------------------------------------------------

def create_rag(documents):

    if not documents:

        return None


    # Split documents into smaller chunks

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=800,

        chunk_overlap=100
    )


    chunks = splitter.split_documents(
        documents
    )


    # HuggingFace local embeddings

    embeddings = HuggingFaceEmbeddings(

        model_name=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        )
    )


    # Create vector database

    vector_db = InMemoryVectorStore.from_documents(

        documents=chunks,

        embedding=embeddings
    )


    return vector_db


# ---------------------------------------------------------
# RETRIEVE CONTEXT
# ---------------------------------------------------------

def retrieve_context(
    vector_db,
    question
):

    if vector_db is None:

        return ""


    # Retrieve only top 3 chunks

    documents = vector_db.similarity_search(

        question,

        k=3
    )


    context_parts = []


    for doc in documents:

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        file_type = doc.metadata.get(
            "type",
            "Unknown"
        )


        context_parts.append(

            f"""
SOURCE: {source}

TYPE: {file_type}

CONTENT:
{doc.page_content}
"""
        )


    context = (
        "\n"
        "--------------------------------\n"
    ).join(context_parts)


    # Extra protection against large Groq request

    return context[:8000]