# Smart AI Logistics Intelligence System

An end-to-end capstone project that predicts driver risk from telemetry data,
automates alerting, and visualizes results in real time.

## Architecture

```
Driver Telemetry (simulated) 
        │
        ▼
   n8n Webhook  ──────────────►  FastAPI (/analyze_risk_score)
        │                                │
        │                                ▼
        │                        ML Model (RandomForest)
        │                                │
        ▼                                │
  n8n IF node  ◄────────────────────────┘
   (risk_score >= 70?)
        │
        ├── YES → Send Alert (Slack/Email/Webhook)
        └── ALWAYS → /log_data → SQLite Database → Streamlit Dashboard
```

**Data flow in one sentence:** telemetry comes in through n8n, n8n calls the
FastAPI service which runs it through the ML model, the result gets logged
to the database, the dashboard reads that database to show live risk status,
and n8n separately fires an alert if the score crosses the threshold.

## Folder structure

```
logistics_ai/
├── api/                  FastAPI microservice
│   ├── main.py           app + 3 required endpoints
│   ├── schemas.py        Pydantic request/response models
│   ├── database.py       SQLAlchemy models + SQLite setup
│   └── logistics.db      (created automatically on first run)
├── ml_model/              
│   ├── train_model.py    generates data + trains + saves model.pkl
│   └── model.pkl         trained RandomForest bundle
├── automation_n8n/
│   └── workflow.json     importable n8n workflow
├── dashboard/
│   └── app.py            Streamlit real-time dashboard
└── docs/
    ├── README.md          (this file)
    ├── SETUP_GUIDE.md
    └── API_DOCUMENTATION.md
```

## Quick start

See `SETUP_GUIDE.md` for full step-by-step instructions. Short version:

```bash
# 1. Train the model (only needed once, or when you have new data)
cd ml_model && python train_model.py

# 2. Start the API
cd ../api && pip install fastapi uvicorn sqlalchemy pandas scikit-learn joblib
uvicorn main:app --reload --port 8000

# 3. Start the dashboard (separate terminal)
cd ../dashboard && pip install streamlit
streamlit run app.py

# 4. Import automation_n8n/workflow.json into n8n and activate it
```

## What each component actually does

- **ML Model**: a RandomForestClassifier trained on 9 driving-behavior
  features (speed, harsh braking, fatigue, etc.) to classify a trip as
  low/medium/high risk. It outputs both a hard label and probabilities,
  which lets the API build a smooth 0–100 risk score instead of just 3 buckets.
- **FastAPI**: the "brain" that other systems talk to. It loads the model
  once at startup (fast inference) and exposes it as clean JSON endpoints.
- **n8n**: the automation glue. It doesn't know anything about ML — it just
  moves data between systems and makes a decision (alert or not) based on
  a number the API gives it.
- **Database (SQLite)**: the single source of truth for "what happened."
  Both the API (writer) and dashboard (reader) point at the same file.
- **Dashboard (Streamlit)**: turns rows in the database into something a
  human can glance at — color-coded table, counts, and trend charts.
