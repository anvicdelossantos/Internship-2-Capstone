# Setup Guide

Follow this in order — each step depends on the one before it.

## Prerequisites

- Python 3.10+
- Node.js 18+ (only needed if you self-host n8n instead of using n8n Cloud)
- Git

## Step 1 — Clone / open the project

```bash
cd logistics_ai
```

## Step 2 — Train the ML model

```bash
cd ml_model
pip install scikit-learn pandas numpy joblib
python train_model.py
```

This prints accuracy, a confusion matrix, and feature importances, then
saves `model.pkl`. **You should see this file appear in `ml_model/`
before moving on** — the API will fail to start without it.

> Swap in real data later: replace the `generate_dataset()` function's
> output with `pd.read_csv("your_real_data.csv")`, as long as the column
> names match `feature_cols` in the script.

## Step 3 — Start the FastAPI service

```bash
cd ../api
pip install fastapi uvicorn "sqlalchemy>=2.0" pandas scikit-learn joblib
uvicorn main:app --reload --port 8000
```

Check it worked: open **http://localhost:8000/docs** — you should see the
Swagger UI listing `/predict_driver_behavior`, `/analyze_risk_score`,
`/log_data`, and `/logs`. Try one directly from that page with "Try it out".

A `logistics.db` SQLite file will appear in `api/` the first time the
server starts — that's expected, it's your database.

## Step 4 — Start the dashboard

Open a **second terminal** (keep the API running in the first one):

```bash
cd dashboard
pip install streamlit pandas
streamlit run app.py
```

It'll open at **http://localhost:8501**. It'll say "No data yet" until you
log at least one result (Step 5 or 6).

## Step 5 — Manually test the pipeline (before wiring up n8n)

```bash
curl -X POST http://localhost:8000/analyze_risk_score \
  -H "Content-Type: application/json" \
  -d '{"driver_id":"DRV-1001","avg_speed_kmh":95,"speeding_events":5,
       "harsh_braking_events":4,"harsh_acceleration_events":3,"sharp_turns":2,
       "driving_hours":9,"night_driving_pct":0.6,"phone_usage_events":2,
       "fatigue_score":0.7}'
```

Then log the result:

```bash
curl -X POST http://localhost:8000/log_data \
  -H "Content-Type: application/json" \
  -d '{"driver_id":"DRV-1001","risk_level":"high","risk_score":82.5,"source":"manual"}'
```

Refresh the dashboard — you should see the row appear.

## Step 6 — Set up n8n automation

1. Install n8n: `npx n8n` (or use n8n Cloud / Docker: `docker run -it --rm -p 5678:5678 n8nio/n8n`)
2. Open n8n at http://localhost:5678
3. Click **Import from File** and select `automation_n8n/workflow.json`
4. Open the **"Send Alert"** node and either:
   - connect a real Slack/Telegram credential, or
   - replace it with a generic **Webhook** / **Set** node if you don't have
     an alerting channel for the demo
5. Click **Activate** on the workflow
6. Copy the webhook URL n8n shows you (something like
   `http://localhost:5678/webhook/driver-telemetry`)
7. Send test telemetry to that URL instead of directly to FastAPI:

```bash
curl -X POST http://localhost:5678/webhook/driver-telemetry \
  -H "Content-Type: application/json" \
  -d '{"driver_id":"DRV-2002","avg_speed_kmh":110,"speeding_events":8,
       "harsh_braking_events":6,"harsh_acceleration_events":5,"sharp_turns":3,
       "driving_hours":11,"night_driving_pct":0.8,"phone_usage_events":4,
       "fatigue_score":0.9}'
```

n8n will call FastAPI, log the result to the DB, and fire the alert node
since this payload is designed to score above the 70 threshold.

## Step 7 — Final packaging (Day 6)

- `git init && git add . && git commit -m "Smart AI Logistics Intelligence System"`
- Push to GitHub, make sure `logistics.db` is in `.gitignore` (it's
  generated data, not source code)
- Record a short screen-capture walking through: send telemetry → watch
  it hit the API → watch it appear in the dashboard → show the n8n alert firing
