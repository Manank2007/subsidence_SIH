import streamlit as st
import plotly.graph_objects as go
from collections import deque
import json
import os
import time

RISK_FILE = "latest_risk.json"
TRIGGER_FILE = "trigger.flag"
HISTORY_LEN = 40

st.set_page_config(page_title="Mine Subsidence Early Warning System", layout="wide")

if "risk_history" not in st.session_state:
    st.session_state.risk_history = deque([0] * HISTORY_LEN, maxlen=HISTORY_LEN)


def get_latest_data():
    if not os.path.exists(RISK_FILE):
        return {"risk_score": 0, "eta": None, "tilt": 0, "vibration": 0, "state": "calibrating"}
    try:
        with open(RISK_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {"risk_score": 0, "eta": None, "tilt": 0, "vibration": 0, "state": "calibrating"}


data = get_latest_data()
risk = data.get("risk_score", 0)
eta = data.get("eta")
tilt = data.get("tilt", 0)
vibration = data.get("vibration", 0)
state = data.get("state", "calibrating")

st.session_state.risk_history.append(risk)

st.title("⛏️ AI-Enabled Mine Subsidence Early Warning System")
st.caption("Real-time surface deformation monitoring — Node 01 | SIH 2026")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Ground Tilt", f"{tilt:.2f}°")
with col2:
    st.metric("Vibration Level", f"{vibration:.0f}")
with col3:
    eta_display = f"{eta:.1f} readings" if eta else "—"
    st.metric("Est. Time to Threshold", eta_display)
with col4:
    state_display = {"stable": "🟢 Stable", "watch": "🟡 Watch",
                      "critical": "🔴 Critical", "calibrating": "⚙️ Calibrating"}.get(state, state)
    st.metric("System Status", state_display)
with col5:
    if st.button("⚠ Trigger Ground Event"):
        with open(TRIGGER_FILE, "w") as f:
            f.write("1")
        st.toast("Event triggered!")

st.divider()

colA, colB = st.columns([1, 1.3])

with colA:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=risk,
        title={'text': "Subsidence Risk Score", 'font': {'size': 20}},
        number={'suffix': " / 100", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#c0392b" if risk > 60 else "#e67e22" if risk > 30 else "#27ae60", 'thickness': 0.3},
            'steps': [{'range': [0, 30], 'color': "#d4f4dd"},
                      {'range': [30, 60], 'color': "#fff3cd"},
                      {'range': [60, 100], 'color': "#f8d7da"}],
            'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 0.85, 'value': 80}
        }
    ))
    fig_gauge.update_layout(height=380, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with colB:
    colors = ['#c0392b' if v > 60 else '#e67e22' if v > 30 else '#27ae60'
              for v in st.session_state.risk_history]
    fig_trend = go.Figure(go.Scatter(
        y=list(st.session_state.risk_history), mode='lines+markers',
        line=dict(color='#3498db', width=2), marker=dict(size=5, color=colors),
        fill='tozeroy', fillcolor='rgba(52, 152, 219, 0.1)'
    ))
    fig_trend.update_layout(title="Risk Score Trend (Live)", yaxis_range=[0, 100],
                             height=380, margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig_trend, use_container_width=True)

if state == "critical":
    st.error("⚠️ **CRITICAL ALERT** — Subsidence risk has crossed the safety threshold.")
elif state == "watch":
    st.warning("⚠️ **ELEVATED RISK** — Ground deformation pattern deviating from baseline.")
elif state == "calibrating":
    st.info("⚙️ **Calibrating** — Learning baseline ground conditions. Please wait...")
else:
    st.success("✅ **Ground conditions stable** — No anomalies detected.")

st.divider()
st.caption("Node 01 · MPU6050 + Piezoelectric sensor · Isolation Forest anomaly detection")

time.sleep(1)
st.rerun()