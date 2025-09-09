#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
daily_summary_local.py
로컬 환경에서 실행 가능한 스크립트 (GPU 지원, KMeans & Centroid 듀얼 대표 추출)
Requires: torch, transformers, sklearn, pymongo, nltk
Usage: python daily_summary_local.py
"""
import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from transformers import pipeline, AutoTokenizer
from pymongo import MongoClient
import torch
from collections import defaultdict
# from nltk.tokenize import sent_tokenize

# === 설정 ===
MONGO_URI = os.getenv("MONGODB_URI", "mongodb+srv://coque:hoo8176@clusterloa.tdpglbb.mongodb.net/?retryWrites=true&w=majority")
DB_NAME = os.getenv("DB_NAME", "lostark")
POST_COL = os.getenv("POST_COL", "community_posts")
DAILY_COL = os.getenv("DAILY_COL", "daily_summary")

def make_token_chunks(texts, max_tokens=512):
    chunks, current, cur_len = [], [], 0
    for txt in texts:
        toks = tokenizer(txt, add_special_tokens=False)["input_ids"]
        if cur_len + len(toks) > max_tokens:
            chunks.append(" ".join(current))
            current, cur_len = [], 0
        current.append(txt)
        cur_len += len(toks)
    if current:
        chunks.append(" ".join(current))
    return chunks


def fast_fallback_summarize(text, word_limit=80):
    words = text.strip().split()
    return " ".join(words[:word_limit])


def select_representatives_kmeans(embs, docs, k=5):
    k = min(k, len(docs))
    km = KMeans(n_clusters=k, random_state=42).fit(embs)
    reps = []
    for center in km.cluster_centers_:
        sims = cosine_similarity([center], embs)[0]
        idx = int(np.argmax(sims))
        doc = docs[idx]
        reps.append({
            "title": doc.get("title", ""),
            "url": doc.get("url", ""),
            "text": doc["text"],
            "similarity": float(sims[idx]),
            "method": "kmeans"
        })
    return reps


def select_representatives_centroid(embs, docs, top_n=5):
    centroid = embs.mean(axis=0, keepdims=True)
    sims = cosine_similarity(centroid, embs)[0]
    idxs = np.argsort(-sims)[:min(top_n, len(docs))]
    reps = []
    for i in idxs:
        doc = docs[i]
        reps.append({
            "title": doc.get("title", ""),
            "url": doc.get("url", ""),
            "text": doc["text"],
            "similarity": float(sims[i]),
            "method": "centroid"
        })
    return reps


def main():
    # 디바이스 설정
    device = 0 if torch.cuda.is_available() else -1
    print(f"✅ Using device: {'cuda' if device == 0 else 'cpu'}")

    # MongoDB 연결
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    posts_col = db[POST_COL]
    daily_col = db[DAILY_COL]

    # 요약기 로드
    global summarizer, tokenizer
    summarizer = pipeline(
        "summarization",
        model="lcw99/t5-base-korean-text-summary",
        tokenizer="lcw99/t5-base-korean-text-summary",
        device=device
    )
    tokenizer = AutoTokenizer.from_pretrained("lcw99/t5-base-korean-text-summary")

    # 문서 로드 및 그룹핑
    daily = defaultdict(list)
    cursor = posts_col.find(
        {"keyword": "유각", "date": {"$exists": True, "$ne": None}},
        {"_id":0, "date":1, "text":1, "embedding":1, "title":1, "url":1}
    )
    for doc in cursor:
        date_val = doc.get("date")
        if not date_val or not isinstance(date_val, str):
            continue
        day = date_val[:10]
        daily[day].append(doc)

    # 요약 및 저장
    for day, docs in sorted(daily.items()):
        texts = [d["text"] for d in docs]
        embs = np.vstack([d["embedding"] for d in docs]).astype(np.float32)

        # 추상 요약 (토큰 청크)
        chunks = make_token_chunks(texts)
        partial_summaries = []
        for c in chunks:
            torch.cuda.empty_cache()
            try:
                ps = summarizer(c, max_length=80, min_length=20, do_sample=False)[0]["summary_text"]
            except RuntimeError:
                ps = fast_fallback_summarize(c, word_limit=80)
            partial_summaries.append(ps)

        final_input = " ".join(partial_summaries)
        try:
            summary = summarizer(final_input, max_length=100, min_length=30)[0]["summary_text"]
        except RuntimeError:
            summary = fast_fallback_summarize(final_input, word_limit=100)

        reps_k = select_representatives_kmeans(embs, docs)
        reps_c = select_representatives_centroid(embs, docs)
        representatives = reps_k + reps_c

        # DB 저장
        daily_col.update_one(
            {"date": day},
            {"$set": {"date": day, "count": len(docs), "summary": summary, "representatives": representatives}},
            upsert=True
        )

        # 출력
        print(f"===== {day} =====")
        print(f"📌 Summary: {summary}")
        for i, r in enumerate(representatives,1):
            print(f"{i}. ({r['similarity']:.3f}) [{r['method']}] {r['title']}\nURL: {r['url']}")
        print(f"[{day}] 저장 완료")

if __name__ == "__main__":
    main()
