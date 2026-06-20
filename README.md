#  Hospital Appointment No-Show Prediction
 
A complete end-to-end ML system that predicts whether a patient will miss their hospital appointment — from raw data and EDA to a deployed Flask API with a Power BI dashboard.
 
**Live Demo:** [medical-appointment-4gt9.onrender.com](https://medical-appointment-4gt9.onrender.com)
*(Free tier — allow 30–60 seconds for cold start)*
 
---
 
## 📌 The Problem
 
In Brazil's public health system, **~20% of scheduled appointments go unfulfilled** every day. No-shows waste doctor time, delay care for other patients, and strain an already limited healthcare system.
 
The question: **Can we predict, at the time of scheduling, whether a patient is likely to miss their appointment?**
 
If yes, clinics can take action early — call high-risk patients, adjust overbooking, or allocate resources more efficiently.
 
---
 
##  My Approach
 
Rather than jumping straight to modeling, I spent significant time on EDA to understand *why* patients don't show up. Several of my initial assumptions turned out to be wrong — and those discoveries shaped the entire project.
 
### What I found that I didn't expect
 
**SMS reminders don't help — or do they?**
 
At first glance: patients who received SMS reminders had a **higher** no-show rate (27.6%) than those who didn't (16.7%). That seems backwards.
 
The real explanation: SMS reminders were sent only to long-wait patients. Wait time was the real driver. SMS was just correlated with it — a classic **confounding variable**. Treating SMS as a causal factor would have been a modeling mistake.
 
**Wait time is the strongest signal:**
 
| Wait Group | No-Show Rate |
|---|---|
| Same Day | ~7% |
| Up to 1 Week | ~14% |
| Up to 1 Month | ~25% |
| 1–3 Months | ~32% |
| 3+ Months | ~27% ↓ |
 
The drop at 3+ months is also interesting — these are likely chronic disease patients with recurring appointments who are more committed to showing up.
 
**Other confirmed patterns:**
- Teenagers have the highest no-show rate (26.6%); Seniors the lowest (15.2%)
- Neighbourhood matters — top no-show area: Santos Dumont at 28.9%
- Evening bookings (8–9 PM) have 30–33% no-show rate vs 15% for 7 AM
---
 
##  Tech Stack
 
| Layer | Tool |
|---|---|
| Data & EDA | pandas, NumPy, Matplotlib, Seaborn |
| Feature Engineering | pandas (datetime, binning) |
| Modeling | XGBoost, scikit-learn, Optuna |
| Deployment | Flask, Docker, Render, Gunicorn |
| Dashboard | Power BI |
| Database | PostgreSQL, SQLAlchemy |
 
---
 
##  Dataset
 
**Source:** KaggleV2-May-2016 (Brazilian public health appointments)
**Size:** 110,527 rows × 14 columns
**Target:** `No_show` — whether a patient missed their appointment (1 = no-show)
 
---
 
## 🛠️ What I Built
 
### Phase 1 — Data Cleaning
- Renamed inconsistent columns (`Hipertension` → `Hypertension`, `Handcap` → `Handicap`)
- Fixed dtypes: dates to datetime, binary columns to int
- Resolved `Age = -1` with median imputation
- Fixed negative `wait_days` caused by swapped date entries
### Phase 2 — Feature Engineering
- `wait_days` = appointment date − scheduled date
- `wait_groups` = categorical buckets (Same Day / 1 Week / 1 Month / 1–3 Months / 3+ Months)
- `appt_weekday` = day of week the appointment falls on
- `scheduled_hour` = hour the patient booked their slot
- `age_groups` = Child / Teenager / Young Adult / Middle Aged / Senior
### Phase 3 — EDA
Five hypotheses tested. fout confirmed, one revealed a confounding variable (SMS — explained above). Full analysis in `hospital_noshow_eda.ipynb`.
 
### Phase 4 — Modeling
- Model: **XGBoost**, tuned with **Optuna** (50 trials, recall-optimized)
- Objective: recall over precision — missing a likely no-show is more costly than a false alarm
- Handled class imbalance with `scale_pos_weight`
**Final model metrics (test set):**
 
| Metric | Show (Class 0) | No-Show (Class 1) |
|---|---|---|
| Precision | 0.93 | 0.29 |
| Recall | 0.47 | 0.86 |
| F1-Score | 0.62 | 0.43 |
| **Accuracy** | | **0.55** |
 
**Why precision (0.29) looks low — and why that's expected:**
 
Precision of 0.29 for no-shows means roughly 1 in 3 flagged patients actually miss their appointment. This isn't a modeling failure — it's a **data ceiling**. Across all model variations and tuning runs, precision stayed at ~0.29–0.31 regardless of algorithm or parameters. The dataset simply doesn't contain enough discriminating signal to push precision higher. Recall (0.86) was prioritised because catching likely no-shows is more valuable than perfect precision in a clinical scheduling context.
 
### Phase 5 — Flask API + Frontend
- Doctor fills in patient details (age, wait time, conditions, neighbourhood, etc.)
- Clicks predict — gets a clear message: "Patient likely to show" or "High risk of no-show"
- Response includes confidence-based messaging (not just a raw label)
### Phase 6 — Deployment
- Dockerized with Gunicorn (production WSGI server, not Flask dev server)
- Deployed on Render free tier
- Power BI dashboard included: `Hospital_NoShow_Dashboard.pbix`
---
 
## 📁 Repo Structure
 
```
Medical_appointment/
│
├── hospital_noshow_eda.ipynb        # Full EDA + modeling notebook
├── app.py                           # Flask API
├── noshow_model.pkl                 # Trained XGBoost model
├── templates/                       # HTML frontend
├── Hospital_NoShow_Dashboard.pbix   # Power BI dashboard
├── Dockerfile                       # Container configuration
├── requirements.txt                 # Python dependencies
└── test.py                          # API test script
```
 
---
 
##  Run Locally
 
**Option 1 — Python**
```bash
git clone https://github.com/Kuldip-Lakhtariya/Medical_appointment.git
cd Medical_appointment
pip install -r requirements.txt
python app.py
```
Visit `http://localhost:5000`
 
**Option 2 — Docker**
```bash
docker build -t noshow-api .
docker run -p 5000:5000 noshow-api
```
 
---
 
##  Key Learnings
 
- **Confounding variables are real and dangerous.** SMS appeared harmful in a naive analysis. Digging deeper revealed wait time was the true driver — a lesson that changes how I approach any correlation.
- **Precision plateau ≠ model failure.** When all tuning runs converge to the same precision, the ceiling is the data, not the model. Recognising this early saves wasted effort.
- **Recall vs precision is a business decision.** In healthcare scheduling, a false alarm (flagging a patient who shows up) costs a phone call. Missing a likely no-show costs a wasted appointment slot. That tradeoff drives the entire modeling objective.
- **Production deployment has its own set of problems.** Gunicorn vs Flask dev server, dynamic `$PORT` for Render, Docker layer caching order — these aren't taught in ML courses but they matter when something actually needs to run.
---
 
## 🔮 Planned Improvements
 
- [ ] Improve Power BI dashboard with drill-through filters by neighbourhood and age group
- [ ] Add SHAP values for per-prediction explainability in the API response
- [ ] Experiment with patient-level historical features (repeat no-show behaviour)
---
 
## 👤 Author
 
**Kuldip Lakhtariya**
B.Tech ECE — LD College of Engineering, Ahmedabad
[GitHub](https://github.com/Kuldip-Lakhtariya) · [LinkedIn](https://linkedin.com/in/kuldip-lakhtariya) · kuldip2611lakhtariya@gmail.com
