import json
import urllib.request

url = 'http://127.0.0.1:8000/ask'
data = json.dumps({'prompt': 'Which country has the most wins?'}).encode('utf-8')
req = urllib.request.Request(url, data, headers={'Content-Type': 'application/json'})
print(urllib.request.urlopen(req, timeout=5).read().decode())
