import requests

url = "http://127.0.0.1:5000/predict"


sample_data = {
    "Gender": 1,
    "age_group_order": 3,
    "scheduled_hour": 14,
    "Scholarship": 0,
    "Hypertension": 1,
    "Diabetes": 0,
    "Alcoholism": 0,
    "Handicap": 0,
    "SMS_received": 1,
    "wait_group_order": 2
}

response = requests.post(url, json=sample_data)
print(response.json())


