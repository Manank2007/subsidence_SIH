import serial
import time
import json
import os
import threading
import numpy as np
from collections import deque
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
# ---------------- CONFIG ----------------
SERIAL_PORT = 'COM3'
BAUD_RATE = 115200
WINDOW_SIZE = 20
RISK_FILE = "latest_risk.json"
TRIGGER_FILE = "trigger.flag"
CALIBRATION_SAMPLES = 300
THRESHOLD_ETA = 80

# ---------------- SETUP ----------------
print(f"Connecting to {SERIAL_PORT}...")
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)
print("Connected. Waiting for sensor data...\n")

tilt_history = deque(maxlen=WINDOW_SIZE)
vib_history = deque(maxlen=WINDOW_SIZE)

baseline_tilt = None
risk_score = 0.0
iso_forest = None
score_min_buffered = 0
score_max_buffered = 0


def compute_features(history):
    if len(history) < 2:
        return 0, 0, 0
    values = list(history)
    mean_val = sum(values) / len(values)
    variance = sum((v - mean_val) ** 2 for v in values) / len(values)
    drift_rate = values[-1] - values[0]
    return mean_val, variance, drift_rate


def anomaly_score_to_risk(raw_score, score_min, score_max):
    """
    Maps raw_score linearly to 0-100, calibrated against the actual
    observed score range from baseline data (see calibration phase).
    """
    if score_max == score_min:
        return 0
    normalized = (score_max - raw_score) / (score_max - score_min)
    risk = normalized * 100
    return max(0, min(100, risk))


def estimate_time_to_threshold(current_score, score_change, threshold=THRESHOLD_ETA):
    if score_change <= 0:
        return None
    remaining = threshold - current_score
    readings_needed = remaining / score_change
    return max(readings_needed, 0)


def write_status(risk, eta, tilt, vibration, state):
    with open(RISK_FILE, "w") as f:
        json.dump({
            "risk_score": risk,
            "eta": eta,
            "tilt": tilt,
            "vibration": vibration,
            "state": state
        }, f)


def read_sensor_line():
    line = ser.readline().decode('utf-8', errors='ignore').strip()
    if not line:
        return None
    parts = line.split(',')
    if len(parts) != 3:
        return None
    try:
        tilt, vibration, _ts = map(float, parts)
        return tilt, vibration
    except ValueError:
        return None


# ---------------- TRIGGER LISTENER THREAD ----------------
def listen_for_trigger():
    """Runs in the background: press Enter in terminal, OR the dashboard
    can drop a trigger.flag file -- either sends 'J' to the ESP32."""
    while True:
        # Check for dashboard-triggered file flag
        if os.path.exists(TRIGGER_FILE):
            ser.write(b'J')
            os.remove(TRIGGER_FILE)
            print("\n>>> JERK TRIGGERED (via dashboard) <<<\n")
        time.sleep(0.2)


def listen_for_keypress():
    """Press Enter in this terminal to trigger a jerk manually."""
    while True:
        input()
        ser.write(b'J')
        print("\n>>> JERK TRIGGERED (via keypress) <<<\n")


threading.Thread(target=listen_for_trigger, daemon=True).start()
threading.Thread(target=listen_for_keypress, daemon=True).start()

# ---------------- HARDWARE SANITY CHECK ----------------
print("=== HARDWARE CHECK: printing 10 raw readings ===")
count = 0
while count < 10:
    result = read_sensor_line()
    if result:
        tilt, vibration = result
        print(f"  Reading {count+1}: tilt={tilt:.2f}  vibration={vibration:.0f}")
        count += 1
print("=== Hardware check complete. ===\n")

# ---------------- CALIBRATION PHASE ----------------
print(f"Calibrating baseline... keep the rig UNDISTURBED. Do NOT trigger jerk yet.")
write_status(0, None, 0, 0, "calibrating")

calibration_features = []
baseline_tilt = None

while len(calibration_features) < CALIBRATION_SAMPLES:
    result = read_sensor_line()
    if not result:
        continue
    tilt, vibration = result

    if baseline_tilt is None:
        baseline_tilt = tilt

    tilt_history.append(tilt - baseline_tilt)
    vib_history.append(vibration)

    if len(tilt_history) == WINDOW_SIZE:
        tilt_mean, tilt_var, tilt_drift = compute_features(tilt_history)
        vib_mean, vib_var, vib_drift = compute_features(vib_history)
        calibration_features.append([tilt_drift, tilt_var, vib_mean])
        if len(calibration_features) % 30 == 0:
            print(f"  Calibration progress: {len(calibration_features)}/{CALIBRATION_SAMPLES}")
            

# After collecting calibration_features, before training:
scaler = StandardScaler()
calibration_array = np.array(calibration_features)
calibration_scaled = scaler.fit_transform(calibration_array)

iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(calibration_scaled)

# Recompute calibration scores using scaled data
calibration_scores = iso_forest.decision_function(calibration_scaled)
score_min = calibration_scores.min()
score_max = calibration_scores.max()

# Train the model
calibration_array = np.array(calibration_features)
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(calibration_array)

# Derive real score range from calibration data itself
calibration_scores = iso_forest.decision_function(calibration_array)
score_min = calibration_scores.min()
score_max = calibration_scores.max()
score_range = score_max - score_min
score_min_buffered = score_min - (score_range * 0.5)
score_max_buffered = score_max + (score_range * 0.1)

print(f"\nCalibration score range: min={score_min:.4f}, max={score_max:.4f}")
print(f"Buffered scoring range: min={score_min_buffered:.4f}, max={score_max_buffered:.4f}")
print("=== Model trained. Switching to live monitoring. ===")
print("(Press Enter anytime to trigger a jerk event.)\n")

# ---------------- LIVE MONITORING ----------------
tilt_history.clear()
vib_history.clear()

while True:
    result = read_sensor_line()
    if not result:
        continue
    tilt, vibration = result

    tilt_history.append(tilt - baseline_tilt)
    vib_history.append(vibration)

    if len(tilt_history) == WINDOW_SIZE:
        tilt_mean, tilt_var, tilt_drift = compute_features(tilt_history)
        vib_mean, vib_var, vib_drift = compute_features(vib_history)

        features = np.array([[tilt_drift, tilt_var, vib_mean]])
        raw_score = iso_forest.decision_function(features)[0]
        new_score = anomaly_score_to_risk(raw_score, score_min_buffered, score_max_buffered)

        score_change = new_score - risk_score
        risk_score = new_score
        eta = estimate_time_to_threshold(risk_score, score_change)

        state = "critical" if risk_score > THRESHOLD_ETA else "watch" if risk_score > 40 else "stable"

        write_status(risk_score, eta, tilt, vibration, state)

        print(f"Risk: {risk_score:5.1f} | State: {state:9s} | Tilt drift: {tilt_drift:6.2f} | "
              f"Vib: {vib_mean:6.1f} | ETA: {eta}")