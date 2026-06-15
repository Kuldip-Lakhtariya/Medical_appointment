from flask import Flask,request,jsonify,render_template
import joblib
import pandas as pd
import numpy as np
from datetime import datetime


app = Flask(__name__)

model = joblib.load("noshow_model.pkl")

def get_age_group_order(age):
    if age <=12:
        age_group_order = 1
    elif age >12 and age<=17:
        age_group_order = 2
    elif age >17 and age<=35:
        age_group_order = 3
    elif age >35 and age<=60:
        age_group_order = 4
    else:
        age_group_order = 5
    return age_group_order

def get_wait_group_order(wait_day):
    if wait_day ==0:
        wait_day_order = 1
    elif wait_day >0 and wait_day <=7:
        wait_day_order = 2
    elif wait_day >7 and wait_day <=30:
        wait_day_order = 3
    elif wait_day >30 and wait_day<=90:
        wait_day_order = 4
    else:
        wait_day_order = 5
    return wait_day_order

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    data = request.get_json()
    scheduled = datetime.strptime(data["scheduled_date"],"%Y-%m-%d")
    appointment  = datetime.strptime(data["appointment_date"],"%Y-%m-%d")
    diff = appointment - scheduled
    wait_day = diff.days

    if wait_day < 0:
        return jsonify({"error": "Appointment date cannot be before scheduled date"}), 400
    
    age_order = get_age_group_order(data["Age"])
    wait_order = get_wait_group_order(wait_day)
    scheduled_hour = int(data["scheduled_time"].split(":")[0])

    model_input = {
        "Gender"          :data["Gender"],
        "age_group_order" :age_order,
        "scheduled_hour"  :scheduled_hour,
        "Scholarship"     :data["Scholarship"],
        'Hypertension'    :data["Hypertension"],
        "Diabetes"        :data["Diabetes"],
        "Alcoholism"      :data["Alcoholism"],
        "Handicap"        :data["Handicap"],
        "SMS_received"    :data["SMS_received"],
        "wait_group_order":wait_order,
    }
    df = pd.DataFrame([model_input])
    proba = model.predict_proba(df)[0][1]
    if proba<=0.20:
        show_confidence = 100 - (proba * 100)
        result = {"Result" : f"There is a {show_confidence:.1f}% chance this patient will show up for their appointment."}
        return jsonify(result)
    elif proba>0.20 and proba<=0.45:
        show_confidence = 100 - (proba * 100)
        result = {"Result" : f"There is a {show_confidence:.1f}% chance this patient will show up for their appointment."}
        return jsonify(result)
    elif proba<0.55 and proba>0.45:
        show_confidence = 100 - (proba * 100)
        result = {"Result" : f"There is a {show_confidence:.1f}% chance thispatient will show up so don't trust the prediction"}
        return jsonify(result)
    elif proba>0.55 and proba<0.80:
        show_confidence = (proba * 100)
        result = {"Result" : f"There is a {show_confidence:.1f}% chance this patient will not show up for their appointment."}
        return jsonify(result)
    else:
        show_confidence = (proba * 100)
        result = {"Result" : f"There is a {show_confidence:.1f}% chance this patient will not show up for their appointment."}
        return jsonify(result)

@app.route("/health",methods=["GET"])
def health():
    if model is not None:
        return jsonify({"status":"ok","model loaded":True})
    else:
        return jsonify({"status":"error","model loaded":False}),500

if __name__ == "__main__":
    app.run(debug=True)

