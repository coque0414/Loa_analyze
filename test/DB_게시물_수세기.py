from pymongo import MongoClient
from datetime import datetime
from collections import Counter

# ✅ MongoDB 연결
client = MongoClient("mongodb+srv://coque:hoo8176@clusterloa.tdpglbb.mongodb.net/?retryWrites=true&w=majority")
db = client['lostark']
post_col = db['community_posts']

# ✅ 날짜 수집 + 샘플 출력
dates = []
sample_count = 0

# ✅ "keyword" 필드에 "유각" 포함된 문서만 찾기
query = {'keyword': {'$regex': '보석'}}  # 부분일치 검색

print("📌 [샘플 date 문자열 확인]")
for post in post_col.find(query, {'date': 1, 'keyword': 1}):
    date_str = post.get('date')
    
    # 최대 5개까지만 샘플로 출력
    if date_str and sample_count < 5:
        print(f"샘플 {sample_count + 1}: {date_str} (type: {type(date_str)})")
        sample_count += 1

    if date_str:
        try:
            # 문자열을 datetime 객체로 파싱
            date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M')
            dates.append(date_obj.date())  # 날짜만 저장
        except Exception as e:
            print(f"❌ 날짜 파싱 오류: {date_str}, 오류: {e}")

# ✅ 날짜별로 카운트
date_counts = Counter(dates)

# ✅ 날짜순 정렬 출력
print("\n📅 [일자별 게시물 수]")
for date, count in sorted(date_counts.items(), reverse=True):
    print(f"{date.month}월 {date.day}일 게시물: {count}개")
