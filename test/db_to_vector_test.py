from pymongo import MongoClient
import pandas as pd
from pykospacing import Spacing
import re

MONGODB_URI = "mongodb+srv://coque:hoo8176@clusterloa.tdpglbb.mongodb.net/?retryWrites=true&w=majority&appName=ClusterLoa"
DB_NAME = "lostark"
COLLECTION_NAME = "community_posts"

client = MongoClient(MONGODB_URI)
collection = client[DB_NAME][COLLECTION_NAME]

# 필요한 필드만 불러오기
cursor = collection.find({}, {"title": 1, "text": 1, "date": 1})
df = pd.DataFrame(cursor)

# 설정 초기화 (기본값으로 되돌리기)
pd.reset_option('all')  # 모든 설정 초기화

# 모든 설정 변경
# pd.set_option('display.max_rows', None)        # 모든 행 보기
# pd.set_option('display.max_columns', None)     # 모든 열 보기
# pd.set_option('display.max_colwidth', None)    # 열 내용 전부 보기
# pd.set_option('display.expand_frame_repr', False)  # 가로로 넓게 보이기


df['title'] = df['title'].fillna("").str.replace('\n', ' ')
df['text'] = df['text'].fillna("").str.replace('\n', ' ')
df['input_text'] = df['title'] + " " + df['text']
df['input_text'] = df['input_text'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
# print(df['input_text']) -> 공백 제거 완료

#띄어쓰기 교정티비
spacing = Spacing()
df['input_text'] = df['input_text'].apply(spacing)
print(df['input_text'])