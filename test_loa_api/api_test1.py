import requests
import json
from loa_api_token import Token

headers = {
    'accept': 'application/json',
    'authorization': Token
}

url = 'https://developer-lostark.game.onstove.com/news/notice'

response = requests.get(url, headers=headers)
jsonObject = response.json()

print(jsonObject)

# json 출력 코드
for list in jsonObject :
    print(list.get("Title"))