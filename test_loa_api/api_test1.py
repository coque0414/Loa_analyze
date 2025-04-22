import requests
import json
from loa_api_token import Token

headers = {
    'accept': 'application/json',
    'Authorization' : Token
}

characterName = '경민네'
url = f'https://developer-lostark.game.onstove.com/characters/{characterName}/siblings'

response = requests.get(url, headers=headers)
json_data = response.json()

# print("응답내용:", response.text)
# print(json_data)
for list1 in json_data:
    print(json.dumps(list1, indent=4, ensure_ascii=False))