import requests

files = {'file': open("269.png", "rb")}
response = requests.post("http://127.0.0.1:5000/predict", files=files)
print(response.json())
