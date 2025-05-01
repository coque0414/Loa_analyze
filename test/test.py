import requests

url = "https://www.inven.co.kr/board/lostark/6271?p=1"

html = requests.get(url).text

print(html)