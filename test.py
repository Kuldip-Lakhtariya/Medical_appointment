import requests

url = "http://127.0.0.1:5000/predict"

sample_data = {
  "Age": 75,
  "scheduled_date": "2026-01-15",
  "scheduled_time": "23:00",
  "appointment_date": "2026-06-15",
  "Gender": 0,
  "Scholarship": 0,
  "Hypertension": 1,
  "Diabetes": 0,
  "Alcoholism": 0,
  "Handicap": 0,
  "SMS_received": 0
}

response = requests.post(url, json=sample_data)
print(response.json())