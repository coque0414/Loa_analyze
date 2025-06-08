    # 5) RAG 프롬프트 생성
    context = "\n\n".join(f"- {d['title']}: {d['text']}" for d in top_docs)
    prompt = textwrap.dedent(f'''
    다음은 로스트아크 커뮤니티 게시물 요약입니다.
    {context}
    질문: {question}
    답변(한국어):
    ''')