import serial
import time
import json
import numpy as np
from collections import deque
from sklearn.ensemble import IsolationForest

# ---------------- CONFIG ----------------
SERIAL_PORT = 'COM3'          # update to match your actual port
BAUD_RATE = 115200
WINDOW_SIZE = 20
RISK_FILE = "latest_risk.json"
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


def compute_features(history):
    if len(history) < 2:
        return 0, 0, 0
    values = list(history)
    mean_val = sum(values) / len(values)
    variance = sum((v - mean_val) ** 2 for v in values) / len(values)
    drift_rate = values[-1] - values[0]
    return mean_val, variance, drift_rate


def anomaly_score_to_risk(raw_score):
    risk = (0.5 - raw_score) * 200
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


# ---------------- HARDWARE SANITY CHECK ----------------
print("=== HARDWARE CHECK: printing 10 raw readings ===")
count = 0
while count < 10:
    result = read_sensor_line()
    if result:
        tilt, vibration = result
        print(f"  Reading {count+1}: tilt={tilt:.2f}  vibration={vibration:.0f}")
        count += 1
print("=== Hardware check complete. If these look sane, proceeding. ===\n")

# ---------------- CALIBRATION PHASE ----------------
print(f"Calibrating baseline... keep the rig UNDISTURBED for ~{CALIBRATION_SAMPLES // 5} seconds.")
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
        if len(calibration_features) % 10 == 0:
            print(f"  Calibration progress: {len(calibration_features)}/{CALIBRATION_SAMPLES}")

iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(np.array(calibration_features))
print("\n=== Calibration complete. Model trained. Switching to live monitoring. ===\n")

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
        new_score = anomaly_score_to_risk(raw_score)

        score_change = new_score - risk_score
        risk_score = new_score
        eta = estimate_time_to_threshold(risk_score, score_change)

        state = "critical" if risk_score > THRESHOLD_ETA else "watch" if risk_score > 40 else "stable"

        write_status(risk_score, eta, tilt, vibration, state)

        print(f"Risk: {risk_score:5.1f} | State: {state:9s} | Tilt drift: {tilt_drift:6.2f} | "
              f"Vib: {vib_mean:6.1f} | ETA: {eta}")