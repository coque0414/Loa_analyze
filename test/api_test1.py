import requests
import json
from loa_api_token import Token
from pprint import pprint

headers = {
    'accept': 'application/json',
    'Authorization' : Token
}

# characterName = '경민네'
# url = f'https://developer-lostark.game.onstove.com/characters/{characterName}/siblings'
itemcode = 65203305
url = f'https://developer-lostark.game.onstove.com/markets/items/{itemcode}'
# url = f'https://developer-lostark.game.onstove.com/auctions/options'


response = requests.get(url, headers=headers)
json_data = response.json()
print(json.dumps(json_data[1], indent=4, ensure_ascii=False))

# print("응답내용:", response.text)
# print(json_data)
# print(response.text)

# for list1 in json_data:
#     print(json.dumps(list1, indent=4, ensure_ascii=False))