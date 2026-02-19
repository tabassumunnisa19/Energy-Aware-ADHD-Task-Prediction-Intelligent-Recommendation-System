# 🧠 AI-Powered ADHD Task Scheduler  
### Energy-Aware Intelligent Task Prediction & Recommendation System

An AI-driven adaptive task scheduling system designed to support
neurodivergent (ADHD-friendly) productivity patterns.

This system predicts task completion likelihood, recommends optimal scheduling
actions, and continuously adapts using reinforcement learning.

---

## 🚀 Features

- ✅ Task completion prediction using Machine Learning (Random Forest)
- ✅ Context-aware intelligent recommendations:
  - Schedule full task
  - Schedule with breaks
  - Convert to micro-task
  - Recommend rest
- ✅ Reinforcement Learning–based adaptive scheduler (Q-Learning)
- ✅ Streamlit interactive dashboard
- ✅ Ethical, non-diagnostic, behavior-based AI design

---

## 🧠 How It Works

1. User inputs:
   - Energy level
   - Focus level
   - Task duration
   - Task priority
   - Time of day

2. Random Forest model predicts probability of completion

3. Reinforcement Learning agent selects optimal action

4. System adapts based on user behavior over time

---

## 🧩 Project Structure

```
Adhd_AI_scheduler_app/
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

```
----

🛠 Tech Stack

Python
Scikit-learn
Pandas
Streamlit
Reinforcement Learning (Q-Learning)

----

⚠️ Ethical Disclaimer


This project is not a medical diagnostic tool.
It is a productivity-support system designed for behavioral adaptation.

---

🌍 Future Improvements

Real user data integration
Cloud deployment (Render / Railway)
User authentication system
Performance tracking dashboard
Personalized AI fine-tuning

---

👩‍💻 Author

Tabassum Unnisa

AI Developer | Data Scientist | ML Enthusiast


