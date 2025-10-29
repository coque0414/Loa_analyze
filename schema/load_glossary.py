# -*- coding: utf-8 -*-
"""
glossary.schema.json -> Atlas 업서트 (docs_glossary)
- 문자열 바깥에서만 // 주석 제거
- '...' 자리표시자 라인 제거
- 문자열 내부 개행/제어문자 안전 치환
- 트레일링 콤마 제거
- 여러 JSON 객체 붙어있는 형태 {..}{..} 분할
- 인덱스 생성
"""
import os, re, json
from typing import List, Dict, Any
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI", "")
DB_NAME = os.getenv("DB_NAME", "lostark")
COLL_NAME = "docs_glossary"
SRC_PATH = "schema\glossary.schema.json"

# ───────────────────────────────────────── helpers

def _strip_js_comments_safely(text: str) -> str:
    """문자열 바깥에서만 // ~ 줄끝 주석 제거"""
    out = []
    i, n = 0, len(text)
    in_str, esc = False, False
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
        else:
            if ch == '"':
                in_str = True
                out.append(ch)
                i += 1
            elif ch == '/' and i + 1 < n and text[i+1] == '/':
                # 주석: 줄 끝까지 스킵
                while i < n and text[i] != '\n':
                    i += 1
            else:
                out.append(ch)
                i += 1
    return "".join(out)

def _remove_ellipsis_lines(text: str) -> str:
    """한 줄에 '...' 만 있는 자리표시자 제거"""
    return re.sub(r"^\s*\.\.\.\s*$", "", text, flags=re.MULTILINE)

def _split_json_objects(text: str) -> List[str]:
    """{...}{...} 형태 분할 (문자열 인식)"""
    objs, buf, depth, in_str, esc = [], [], 0, False, False
    for ch in text:
        buf.append(ch)
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    obj = "".join(buf).strip()
                    if obj:
                        objs.append(obj)
                    buf = []
    return objs

def _fix_trailing_commas(s: str) -> str:
    """배열/객체 끝 트레일링 콤마 제거"""
    return re.sub(r',\s*([}\]])', r'\1', s)

def _sanitize_control_in_strings(s: str) -> str:
    """문자열 내부의 개행/제어문자 치환"""
    out = []
    in_str, esc = False, False
    for ch in s:
        if in_str:
            if esc:
                out.append(ch); esc = False
            else:
                if ch == '\\':
                    out.append(ch); esc = True
                elif ch == '"':
                    out.append(ch); in_str = False
                elif ch in '\r\n':
                    out.append(' ')
                elif ord(ch) < 0x20:
                    out.append(' ')
                else:
                    out.append(ch)
        else:
            out.append(ch)
            if ch == '"':
                in_str = True
    return "".join(out)

def _norm(s: str) -> str:
    return "".join((s or "").lower().split())

def _derive_aliases(doc: Dict[str, Any]) -> List[str]:
    """줄임말 보강(특히 각인서) + 공백/빈값 정리"""
    base = list(dict.fromkeys((doc.get("aliases") or []) + [doc.get("name_ko","")]))
    name = doc.get("name_ko") or ""
    typ  = doc.get("type") or ""
    if typ == "engraving" and name:
        base += [f"{name} 각인서", f"{name} 유각", f"{name} 유물 각인서"]
    # 빈 문자열 제거
    base = [a.strip() for a in base if a and a.strip()]
    # 중복 제거
    seen, out = set(), []
    for a in base:
        if a not in seen:
            out.append(a); seen.add(a)
    return out

def _to_glossary(doc: Dict[str, Any]) -> Dict[str, Any]:
    """원본 스키마 -> docs_glossary 포맷"""
    slug = doc.get("_id") or f"{doc.get('type','item')}:{_norm(doc.get('name_ko',''))}"
    term = doc.get("name_ko") or ""
    aliases = _derive_aliases(doc)
    g = {
        "_id": slug,
        "slug": slug,
        "term_ko": term,
        "aliases": aliases,
        "type": doc.get("type"),
        "raw": doc,
        "term_norm": _norm(term),
        "aliases_norm": [_norm(a) for a in aliases],
        "resolver": {}
    }
    # 각인서는 거래소 '각인서' 카테고리에서 "{이름} 각인서"로 조회
    if doc.get("type") == "engraving":
        g["resolver"]["market"] = {
            "category_name": "각인서",
            "category_code": None,  # 런타임에 /markets/options로 자동 조회·캐시
            "query_name": f"{term} 각인서" if term else (doc.get("full_name") or "")
        }
    return g

# ───────────────────────────────────────── main

def main():
    client = MongoClient(MONGODB_URI)
    col = client[DB_NAME][COLL_NAME]

    with open(SRC_PATH, "r", encoding="utf-8") as f:
        raw = f.read()

    txt = _strip_js_comments_safely(raw)
    txt = _remove_ellipsis_lines(txt)

    chunks = _split_json_objects(txt)
    if not chunks:
        raise SystemExit("⚠️ 파싱할 JSON 객체를 못 찾았어요. 파일 내용을 확인하세요.")

    ops, total = [], 0
    for i, c in enumerate(chunks, 1):
        s = _fix_trailing_commas(_sanitize_control_in_strings(c))
        try:
            obj = json.loads(s)
        except Exception as e:
            print(f"❌ JSON 파싱 실패 (#{i}):", e)
            # 문제가 된 앞부분 일부를 같이 출력해주면 원인 추적 쉬움
            print(s[:200].replace("\n"," ⏎ "))
            continue
        g = _to_glossary(obj)
        ops.append(UpdateOne({"_id": g["_id"]}, {"$set": g}, upsert=True))
        total += 1

    if ops:
        res = col.bulk_write(ops, ordered=False)
        print(f"✅ 업서트 완료: 입력 {total}건 / matched={res.matched_count}, upserted={len(res.upserted_ids)}")
    else:
        print("⚠️ 업서트할 문서가 없습니다.")

    # 인덱스
    col.create_index("slug", unique=True)
    col.create_index("term_norm")
    col.create_index("aliases_norm")
    col.create_index([("term_ko", "text"), ("aliases", "text")], default_language="none")
    print("✅ 인덱스 생성 완료")

if __name__ == "__main__":
    main()
