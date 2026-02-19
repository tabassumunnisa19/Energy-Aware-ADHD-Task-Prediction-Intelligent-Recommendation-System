import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict

# ---------------------------------
# Page configuration
# ---------------------------------
st.set_page_config(
    page_title="AI-Powered ADHD Task Scheduler",
    page_icon="🧠",
    layout="centered"
)

# ---------------------------------
# Load image
# ---------------------------------
st.image("ADHD.png", use_column_width=True)

# ---------------------------------
# Title & Description
# ---------------------------------
st.title("🧠 AI-Powered ADHD Task Scheduler")
st.markdown(
    """
    This dashboard uses **Machine Learning + Reinforcement Learning**
    to recommend the **best next action** for a task.

    ⚠️ This is not a medical tool.
    """
)

st.divider()

# ---------------------------------
# Train model dynamically (ROBUST)
# ---------------------------------
@st.cache_resource
def train_model():
    data = pd.read_csv("data/synthetic_adhd_task_data.csv")

    # Automatically encode ALL categorical columns
    data = pd.get_dummies(data)

    # Ensure target column exists
    if "completed" not in data.columns:
        st.error("Target column 'completed' not found in dataset.")
        st.stop()

    X = data.drop("completed", axis=1)
    y = data["completed"]

    # Ensure all features are numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    model = RandomForestClassifier()
    model.fit(X, y)

    return model, X.columns


rf_model, model_columns = train_model()

# ---------------------------------
# Initialize RL table
# ---------------------------------
Q = defaultdict(lambda: np.zeros(4))

# ---------------------------------
# Helper Functions
# ---------------------------------
def recommend_action(task_row, model):
    task_row = pd.get_dummies(task_row)

    # Align columns
    for col in model_columns:
        if col not in task_row:
            task_row[col] = 0

    task_row = task_row[model_columns]

    # Ensure numeric
    task_row = task_row.apply(pd.to_numeric, errors="coerce").fillna(0)

    prob = model.predict_proba(task_row)[0, 1]

    if prob >= 0.7:
        action = "Schedule full task"
        reason = "High energy and focus detected."
    elif prob >= 0.4:
        action = "Schedule task with break buffer"
        reason = "Moderate focus detected."
    elif prob >= 0.25:
        action = "Convert to micro-task"
        reason = "Low focus detected."
    else:
        action = "Recommend break / delay"
        reason = "High overload risk."

    return float(round(prob, 2)), action, reason


def discretize_state(prob, energy, break_time, interruptions):
    return (
        int(prob * 10),
        energy - 1,
        min(break_time // 30, 4),
        min(interruptions, 4)
    )


def rl_recommend(prob, energy, break_time, interruptions):
    state = discretize_state(prob, energy, break_time, interruptions)
    action = np.argmax(Q[state])

    actions_map = {
        0: "Schedule full task",
        1: "Schedule with break buffer",
        2: "Convert to micro-task",
        3: "Recommend break"
    }

    return actions_map[action]

# ---------------------------------
# User Input
# ---------------------------------
st.subheader("🔧 Current Task Context")

task_type = st.selectbox("Task Type", ["creative", "admin", "deep", "routine"])

task_complexity = st.slider("Task Complexity", 1, 5, 3)

estimated_duration = st.slider("Estimated Duration (minutes)", 10, 120, 45)

energy_level = st.slider("Current Energy Level", 1, 5, 3)

time_since_break = st.slider("Time Since Last Break (minutes)", 0, 180, 30)

interruptions = st.slider("Interruptions in last hour", 0, 10, 2)

st.divider()

# ---------------------------------
# Recommendation
# ---------------------------------
if st.button("🔮 Get AI Recommendation"):

    input_df = pd.DataFrame([{
        "task_type": task_type,
        "previous_task_type": task_type,
        "task_complexity": task_complexity,
        "estimated_duration": estimated_duration,
        "start_hour": 10,
        "time_since_last_break": time_since_break,
        "interruptions": interruptions,
        "energy_level": energy_level
    }])

    prob, rule_action, reason = recommend_action(input_df, rf_model)
    rl_action = rl_recommend(prob, energy_level, time_since_break, interruptions)

    st.subheader("📊 AI Decision")
    st.metric("Completion Probability", prob)

    st.success(f"📌 Rule-Based Recommendation: **{rule_action}**")
    st.info(f"🤖 RL Recommendation: **{rl_action}**")

    st.markdown(f"**Why this recommendation?** {reason}")

    st.caption("Model retrains safely on each deployment.")
