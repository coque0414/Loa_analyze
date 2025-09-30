import textwrap
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os

from services.db import posts_col
from services.embedder import get_embedder
from dotenv import load_dotenv
load_dotenv()

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def semantic_search(question: str, limit: int = 5):
    embedder = get_embedder()
    q_emb = embedder.encode(question, convert_to_numpy=True).reshape(1, -1)
    docs = await posts_col.find({"embedding": {"$exists": True}}, {"title":1, "text":1, "embedding":1, "_id":0}).to_list(length=10000)
    if not docs:
        return []
    embs = np.vstack([d["embedding"] for d in docs])
    sims = cosine_similarity(q_emb, embs)[0]
    idxs = sims.argsort()[::-1][:limit]
    return [docs[i] for i in idxs]

async def rag_qa(question: str, k: int = 5) -> str:
    ctx_docs = await semantic_search(question, limit=k)
    if not ctx_docs:
        return "⚠️ 관련 문서를 찾지 못했습니다."
    context = "\n".join(f"- {d['title']}: {d['text']}" for d in ctx_docs)
    prompt = textwrap.dedent(f'''
    너는 로스트아크 커뮤니티 게시물만 참고해서 대답하는 어시스턴트야.
    다음 게시물들과 관련 없는 질문이 들어오면 반드시 이렇게 말해:
    "⚠️ 이 질문은 게시물 내용과 관련성이 낮아 답변할 수 없습니다."

    [게시물 요약]
    {context}

    [질문]
    {question}

    [답변 - 한국어로 정확하게 설명하되, 친절하게:]
    ''')
    chat_res = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return chat_res.choices[0].message.content.strip()