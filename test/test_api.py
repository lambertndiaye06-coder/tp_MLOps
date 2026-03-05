import requests
data = {
    "A1": "b",
    "A2": 30.5,
    "A3": 1.5,
    "A4": "u",
    "A5": "g",
    "A6": "c",
    "A7": "v",
    "A8": 6.0,
    "A9": "t",
    "A10": "f",
    "A11": 0,
    "A12": "f",
    "A13": "g",
    "A14": 200.0,
    "A15": 0
}
response = requests.post("http://127.0.0.1:8001/predict", json=data)
print(response.json())
# → {"prediction": 1, "label": "accordé", "probabilité": 0.87}