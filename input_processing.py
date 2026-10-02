import os
import base64
import tempfile

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from config import groq


# ---------------------------------------------------------
# PDF PROCESSING
# ---------------------------------------------------------

def pdf_to_text(file):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp:

        temp.write(file.getvalue())
        temp_path = temp.name

    try:

        loader = PyPDFLoader(temp_path)

        documents = loader.load()

        return documents

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)


# ---------------------------------------------------------
# AUDIO TRANSCRIPTION
# ---------------------------------------------------------

def audio_to_text(file):

    suffix = os.path.splitext(file.name)[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp:

        temp.write(file.getvalue())
        temp_path = temp.name

    try:

        with open(temp_path, "rb") as audio_file:

            result = groq.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3-turbo",
                response_format="text"
            )

        if isinstance(result, str):
            return result

        if hasattr(result, "text"):
            return result.text

        return str(result)

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)


# ---------------------------------------------------------
# IMAGE UNDERSTANDING
# ---------------------------------------------------------

def image_to_text(file):

    image_bytes = file.getvalue()

    encoded_image = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    mime_type = file.type or "image/jpeg"

    response = groq.chat.completions.create(

        model="qwen/qwen3.8-27b",

        messages=[
            {
                "role": "user",

                "content": [

                    {
                        "type": "text",

                        "text": """
Analyze this image carefully.

Extract and explain:

1. Visible text
2. OCR information
3. Tables
4. Charts
5. Diagrams
6. Objects
7. Numbers
8. Important information
9. Relationships between objects
10. Overall meaning

Create a detailed textual description that can be used
as context for a RAG system.

Do not ignore important visual information.
"""
                    },

                    {
                        "type": "image_url",

                        "image_url": {
                            "url": (
                                f"data:{mime_type};base64,"
                                f"{encoded_image}"
                            )
                        }
                    }
                ]
            }
        ]
    )

    return response.choices[0].message.content


# ---------------------------------------------------------
# MAIN FILE PROCESSOR
# ---------------------------------------------------------

def process_files(
    pdf_files,
    text_files,
    audio_files,
    image_files,
    text_input
):

    documents = []

    processed_info = []


    # -----------------------------------------------------
    # DIRECT TEXT
    # -----------------------------------------------------

    if text_input and text_input.strip():

        documents.append(

            Document(

                page_content=text_input,

                metadata={
                    "source": "User Text",
                    "type": "text"
                }
            )
        )

        processed_info.append(
            "📝 Direct text processed"
        )


    # -----------------------------------------------------
    # PDF FILES
    # -----------------------------------------------------

    for file in pdf_files:

        try:

            docs = pdf_to_text(file)

            for doc in docs:

                doc.metadata["source"] = file.name
                doc.metadata["type"] = "pdf"

            documents.extend(docs)

            processed_info.append(
                f"📄 PDF processed: {file.name}"
            )

        except Exception as e:

            processed_info.append(
                f"❌ PDF error: {file.name} → {e}"
            )


    # -----------------------------------------------------
    # TEXT / CSV / MARKDOWN FILES
    # -----------------------------------------------------

    for file in text_files:

        try:

            content = file.getvalue().decode(
                "utf-8",
                errors="ignore"
            )

            documents.append(

                Document(

                    page_content=content,

                    metadata={
                        "source": file.name,
                        "type": "text"
                    }
                )
            )

            processed_info.append(
                f"📃 Text file processed: {file.name}"
            )

        except Exception as e:

            processed_info.append(
                f"❌ Text file error: {file.name} → {e}"
            )


    # -----------------------------------------------------
    # AUDIO FILES
    # -----------------------------------------------------

    for file in audio_files:

        try:

            transcript = audio_to_text(file)

            documents.append(

                Document(

                    page_content=transcript,

                    metadata={
                        "source": file.name,
                        "type": "audio"
                    }
                )
            )

            processed_info.append(
                f"🎵 Audio processed: {file.name}"
            )

        except Exception as e:

            processed_info.append(
                f"❌ Audio error: {file.name} → {e}"
            )


    # -----------------------------------------------------
    # IMAGE FILES
    # -----------------------------------------------------

    for file in image_files:

        try:

            description = image_to_text(file)

            documents.append(

                Document(

                    page_content=description,

                    metadata={
                        "source": file.name,
                        "type": "image"
                    }
                )
            )

            processed_info.append(
                f"🖼️ Image processed: {file.name}"
            )

        except Exception as e:

            processed_info.append(
                f"❌ Image error: {file.name} → {e}"
            )


    return documents, processed_info