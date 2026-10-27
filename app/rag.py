from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )
    return llm

def generate_answer(llm, context, question):
    prompt = f"""
Answer the question using only the provided context.

For each factual claim, cite the specific source that supports it
using [Source 1], [Source 2], etc.

Do not cite a source unless it supports the claim.

Context:
{context}

Question:
{question}

If the answer cannot be found in the context, say:
"I don't know based on the provided documents."
"""

    response = llm.invoke(prompt)

    return response.content

 


# 5. Build prompt
# prompt = f"""
# Answer the question using only the provided context.

# For each factual claim, cite the specific source that supports it
# using [Source 1], [Source 2], etc.

# Do not cite a source unless it supports the claim.
# do not cite a source unless it supports the claim 
# Context:
# {context}

# Question:
# {QUERY}

# If the answer cannot be found in the context, say:
# "I don't know based on the provided documents."
# """