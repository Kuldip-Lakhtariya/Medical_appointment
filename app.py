from flask import Flask,request,jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

model = joblib.load("noshow_model.pkl")



@app.route("/predict",methods=["POST"])
def predict():
    data = request.json()
    df = pd.DataFrame([data])
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
        result = {"Result" : f"There is a {show_confidence:.1f}% chance this patient will show up so don't trust the prediction"}
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

