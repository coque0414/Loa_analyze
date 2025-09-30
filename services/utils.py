from typing import Any, Dict, Iterable, Optional
from datetime import date, datetime

def build_match_multi(field: str, values: Iterable[Any]) -> Dict[str, Any]:
    """
    MongoDB 쿼리에서 field IN(values) 조건을 생성합니다.
    예: {"$match": {"item_code": {"$in": [1,2,3]}}}
    """
    return {"$match": {field: {"$in": list(values)}}}

def build_match_single(field: str, value: Any) -> Dict[str, Any]:
    """
    MongoDB 쿼리에서 단일 값 매치 조건을 생성합니다.
    예: {"$match": {"item_code": 12345}}
    """
    return {"$match": {field: value}}

def _normalize_date_val(v: Any) -> Any:
    """date/datetime -> ISO 문자열 변환(필요 시). 문자열/숫자는 그대로 반환."""
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, date):
        return datetime.combine(v, datetime.min.time()).isoformat()
    return v

def build_date_range_match(field: str = "date", start: Optional[Any] = None, end: Optional[Any] = None) -> Dict[str, Any]:
    """
    날짜 범위 매치 빌더.
    start 또는 end 중 존재하는 것만 포함합니다.
    예: {"$match": {"date": {"$gte": "2024-01-01T00:00:00", "$lte": "2024-01-31T23:59:59"}}}
    입력으로 date/datetime/문자열 가능.
    """
    cond: Dict[str, Any] = {}
    if start is not None:
        cond["$gte"] = _normalize_date_val(start)
    if end is not None:
        cond["$lte"] = _normalize_date_val(end)
    if not cond:
        return {"$match": {}}
    return {"$match": {field: cond}}

__all__ = ["build_match_multi", "build_match_single", "build_date_range_match"]