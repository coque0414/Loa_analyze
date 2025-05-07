import requests
import json
from loa_api_token import Token
import pandas as pd

headers = {
    'accept': 'application/json',
    'Authorization' : Token
}

url = f'https://developer-lostark.game.onstove.com/markets/options'

response = requests.get(url, headers=headers)
json_data = response.json()

# for category in json_data["Categories"]:
#     print(f"[{category['Code']}] {category['CodeName']}")
#     if category.get("Subs"):  # 소분류가 있을 경우만
#         for sub in category["Subs"]:
#             print(f"    └─ [{sub['Code']}] {sub['CodeName']}")

# print(json.dumps(json_data["ItemGrades"], indent=4, ensure_ascii=False))

category_rows = []

for category in json_data["Categories"]:
    main_code = category["Code"]
    main_name = category["CodeName"]
    
    if category.get("Subs"):
        for sub in category["Subs"]:
            category_rows.append({
                "대분류 코드": main_code,
                "대분류 이름": main_name,
                "소분류 코드": sub["Code"],
                "소분류 이름": sub["CodeName"]
            })
    else:
        category_rows.append({
            "대분류 코드": main_code,
            "대분류 이름": main_name,
            "소분류 코드": None,
            "소분류 이름": None
        })

df_categories = pd.DataFrame(category_rows)

# -----------------------------
# ✅ 2. ItemGrades: 아이템 등급
# -----------------------------
df_grades = pd.DataFrame({
    "등급": json_data["ItemGrades"]
})

# -----------------------------
# ✅ 3. ItemTiers: 아이템 티어
# -----------------------------
df_tiers = pd.DataFrame({
    "아이템 티어": json_data["ItemTiers"]
})

# -----------------------------
# ✅ 4. Classes: 직업 목록
# -----------------------------
df_classes = pd.DataFrame({
    "직업": json_data["Classes"]
})

# -----------------------------
# ✅ 표 출력
# -----------------------------
print("\n[1] Categories (대분류/소분류)\n", df_categories)
print("\n[2] ItemGrades (아이템 등급)\n", df_grades)
print("\n[3] ItemTiers (아이템 티어)\n", df_tiers)
print("\n[4] Classes (직업 목록)\n", df_classes)