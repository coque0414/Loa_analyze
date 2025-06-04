from pymongo import MongoClient
import pandas as pd

# ✅ MongoDB 연결
client = MongoClient("mongodb+srv://coque:hoo8176@clusterloa.tdpglbb.mongodb.net/?retryWrites=true&w=majority")
db = client['lostark']
market_col = db['market_items']

# ✅ 데이터 조회
cursor = list(market_col.find(
    {'item_code': 65201005},
    {'_id': 0, 'name':1, 'date': 1, 'avg_price': 1, 'trade_count': 1}
))

# ✅ DataFrame 생성
df = pd.DataFrame(cursor)

if df.empty:
    print("❌ 데이터가 없습니다.")
else:
    # ✅ 문자열 → datetime 변환
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')  # format은 필요에 따라 조정
    print(df.dtypes)  # 컬럼별 데이터 타입 확인
    print(df)
