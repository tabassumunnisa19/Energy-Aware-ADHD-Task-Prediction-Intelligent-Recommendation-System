<<<<<<< HEAD
# Energy-Aware-ADHD-Task-Prediction-Intelligent-Recommendation-System
=======
# 🧠 AI-Powered ADHD Task Scheduler

This project is an **AI-driven adaptive task scheduling system** designed to support
neurodivergent (ADHD-friendly) productivity patterns.

The system predicts task completion likelihood, recommends optimal scheduling actions,
and continuously adapts using reinforcement learning.

---

## 🚀 Features

- Task completion prediction using Machine Learning (Random Forest)
- Context-aware recommendations:
  - Schedule full task
  - Schedule with breaks
  - Convert to micro-task
  - Recommend rest
- Reinforcement Learning–based adaptive scheduler
- User-friendly Streamlit dashboard
- Ethical, non-diagnostic, behavior-based AI design

---

## 🧩 Project Structure

```text
ADASD Scheduler app/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── synthetic_adhd_task_data.csv
│
├── models/
│   ├── rf_model.pkl
│   └── q_table.pkl
│
└── utils/
    ├── recommender.py
    └── rl_agent.py
```




---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
>>>>>>> 6388d56 (Initial commit - ADHD AI Scheduler App)
