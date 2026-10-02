import base64

from config import groq, openai_client


# ---------------------------------------------------------
# TEXT ANSWER
# ---------------------------------------------------------

def generate_answer(
    context,
    question
):

    # Additional safety limit

    context = context[:8000]


    prompt = f"""
You are a Multimodal RAG assistant.

Answer the user's question using ONLY
the retrieved context.

Do not invent information.

If the answer is not available in the
retrieved context, say:

"The information is not available in
the uploaded content."

Retrieved Context:

{context}


User Question:

{question}


Give a clear and concise answer.
"""


    response = groq.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[

            {
                "role": "system",

                "content": (
                    "You are a helpful "
                    "RAG assistant."
                )
            },

            {
                "role": "user",

                "content": prompt
            }
        ],

        temperature=0,

        max_completion_tokens=700
    )


    return response.choices[0].message.content


# ---------------------------------------------------------
# IMAGE GENERATION
# ---------------------------------------------------------

def generate_image(answer):

    prompt = f"""
Create a professional educational visual
based on the following answer.

ANSWER:

{answer}

Requirements:

- Professional educational design
- Clear visual hierarchy
- Easy to understand
- Use diagrams when appropriate
- Use icons when useful
- Use short labels
- Avoid long paragraphs
- Make the visual directly related
  to the answer
"""


    result = openai_client.images.generate(

        model="gpt-image-1",

        prompt=prompt,

        size="1024x1024"
    )


    image_data = base64.b64decode(

        result.data[0].b64_json
    )


    return image_data


# ---------------------------------------------------------
# AUDIO GENERATION
# ---------------------------------------------------------

def generate_audio(answer):

    # Limit audio text

    text = answer[:200]


    response = groq.audio.speech.create(

        model="canopylabs/orpheus-v1-english",

        voice="austin",

        input=text,

        response_format="wav"
    )


    return response.read()