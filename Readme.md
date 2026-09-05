# AI-Enabled Mine Subsidence Early Warning System
### Single-Node Prototype — SIH 2026

---

## 1. Problem Statement

Surface subsidence caused by underground coal mining poses significant risks to nearby communities, public infrastructure, agricultural land, forest areas, and the surrounding environment. Subsidence monitoring in India is largely dependent on conventional field observations, periodic surveys, and post-facto damage assessments, which fail to provide timely warning before ground failure occurs.

This project develops an **AI-enabled smart mine subsidence monitoring and early warning system** using low-cost sensor nodes, real-time feature extraction, and unsupervised anomaly detection — deployable using widely accessible hardware (ESP32, MPU6050, piezoelectric sensors).

**Scope note:** This is a single-node proof-of-concept prototype, built under a internal hackathon timeline. The full problem statement envisions a wireless mesh network of multiple nodes (LoRa/Zigbee) across a mine panel with GIS visualization. This prototype validates the core sensing → feature extraction → anomaly detection → alert pipeline on one node, with mesh networking as the documented scale-up path (see Section 8).

---

## 2. System Architecture

```
[Sensors: MPU6050 + Piezo] → [ESP32 Firmware] → [USB Serial]
        → [Python: Feature Extraction] → [Isolation Forest Model]
        → [Risk Score + State] → [latest_risk.json]
        → [Streamlit Dashboard: Live Gauge + Trend + Alerts]
```

The system runs in two phases:
1. **Calibration phase** — collect baseline "normal" sensor behavior, train the anomaly detection model
2. **Live monitoring phase** — continuously score new sensor readings against the trained baseline, compute a risk score, and display it on a live dashboard

---

## 3. Hardware Components

| Component | Role | Approx. Cost |
|---|---|---|
| ESP32 DevKit (ESP32-WROOM-32) | Core microcontroller — reads sensors, runs servo, sends data over USB serial | ₹350–450 |
| MPU6050 (GY-521 breakout) | Accelerometer + gyroscope — used to compute ground tilt angle | ₹150–165 |
| Piezoelectric vibration sensor | Detects ground-borne vibration/micro-tremors | ₹50–150 |
| SG90/MG90 micro servo | Simulates ground vibration events for demo purposes (drives a rod/paddle into sand) | ₹100–250 |
| Perfboard, jumper wires, USB cable | Assembly and power/data connection to laptop | — |

**Pin Mapping:**
| Component | Pin | ESP32 GPIO |
|---|---|---|
| MPU6050 | SDA | GPIO 21 |
| MPU6050 | SCL | GPIO 22 |
| MPU6050 | VCC / GND | 3V3 / GND |
| Piezo sensor | Signal | GPIO 4 |
| Piezo sensor | GND | GND |
| Servo | Signal | GPIO 18 |
| Servo | VCC | 5V (USB rail) |
| Servo | GND | GND |

---

## 4. Firmware Logic (ESP32)

The firmware performs three jobs in a single non-blocking loop (`millis()`-based timing, no `delay()` blocking):

1. **Sensor sampling** (every 100ms): reads MPU6050 accelerometer values, converts to a tilt angle via `atan2(ay, az) * 180/π`; reads piezo sensor as a raw analog value (0–4095 on ESP32's 12-bit ADC)
2. **Servo behavior**: defaults to a gentle, smooth, narrow-range motion (±15° around center, 1° steps) to simulate baseline ground micro-movement. A single byte `'J'` sent over serial switches it into a 1.5-second sharp, wide-range jerk burst — used to simulate a genuine subsidence/vibration event on demand during the live demo
3. **Data transmission**: sends `tilt,vibration,timestamp` as a CSV line over USB serial at 115200 baud

---

## 5. Feature Engineering (Python)

Raw sensor readings are noisy and carry little meaning individually. The system computes features over a **rolling window** of the last 20 readings (~2 seconds at 10Hz):

| Feature | Formula | What it captures |
|---|---|---|
| **Mean** | `mean = Σxᵢ / n` | Typical/average level over the window |
| **Variance** | `variance = Σ(xᵢ − mean)² / n` | How noisy/unstable the signal is |
| **Drift rate** | `drift = x_last − x_first` | Net change (trend/slope proxy) across the window |

Three features are computed per window and fed to the model: `[tilt_drift, tilt_variance, vibration_mean]`.

---

## 6. Anomaly Detection — Isolation Forest

### Why Isolation Forest
- **Unsupervised**: does not require labeled "subsidence event" data, which is unavailable/scarce in practice
- Learns the shape of "normal" behavior from baseline data, then scores new data by how easily it can be isolated from that learned normal
- Computationally lightweight — runs comfortably on a laptop CPU in real time

### Core Principle
Isolation Forest builds an ensemble of **isolation trees (iTrees)**. Each tree is built by:
1. Randomly selecting a feature
2. Randomly selecting a split value between that feature's min and max in the current data subset
3. Recursively partitioning the data until every point is isolated in its own leaf, or a maximum tree depth is reached

**Key intuition:** anomalies are "few and different" — because they lie far from the dense cluster of normal points, they get isolated (separated into their own leaf) after only a **few random splits**. Normal points, being surrounded by many similar points, require **many more splits** to isolate.

### Anomaly Score Formula
For a data point `x`, the anomaly score is defined using the average path length `h(x)` across all trees in the forest:

```
s(x, n) = 2^( −E[h(x)] / c(n) )
```

Where:
- `E[h(x)]` = average path length (number of splits) to isolate point `x`, averaged across all trees in the forest
- `c(n)` = average path length of unsuccessful search in a Binary Search Tree, used to normalize for sample size `n`:

```
c(n) = 2·H(n−1) − (2·(n−1)/n)
```

- `H(i)` is the harmonic number, approximated as `H(i) ≈ ln(i) + 0.5772156649` (Euler's constant)

### Interpreting the Score
- `s(x,n) → 1`: point has a very short average path length → **strongly anomalous**
- `s(x,n) → 0.5`: point has an average path length similar to the whole sample → **normal**
- `s(x,n) → 0` (rare in practice): point required unusually long paths → **very normal/central**

### What This Project Uses in Practice
scikit-learn's `IsolationForest.decision_function()` returns a related but rescaled value where:
- **Higher (closer to positive)** = more normal
- **Lower/negative** = more anomalous

This project converts that raw score into a human-readable 0–100 risk scale using:
```python
risk = max(0, min(100, (0.5 - raw_score) * 200))
```

This is a linear rescaling: as `raw_score` decreases (more anomalous), `risk` increases toward 100; as `raw_score` increases (more normal), `risk` decreases toward 0. The constants (`0.5` offset, `×200` scale) were empirically tuned against observed score ranges during testing, not derived analytically — a legitimate and disclosed simplification for a prototype.

### Model Parameter
```python
IsolationForest(contamination=0.05, random_state=42)
```
- `contamination=0.05`: assumes ~5% of baseline calibration data may itself be borderline/noisy, setting the model's internal decision threshold accordingly
- `random_state=42`: fixes the random seed for reproducible tree structures across runs

---

## 7. Calibration Phase

Before live monitoring begins, the system collects **300 feature samples** (~configurable, tuned up from an initial 100 after testing showed richer baseline data improved model stability) while the sensor rig runs its default gentle-motion behavior, undisturbed by any deliberate "event" trigger.

This calibration data is used **once** to fit (`iso_forest.fit()`) the Isolation Forest model. After training, the model is not retrained — all subsequent readings are only **scored** against this fixed baseline for the remainder of the session.

**Important operational rule:** the jerk-trigger (`'J'` command) must not be sent during calibration, as doing so would teach the model that jerking is "normal" behavior, defeating the purpose of anomaly detection.

---

## 8. Risk Scoring and Early Warning Logic

In addition to the raw anomaly score, the system computes:

**State classification:**
```python
state = "critical" if risk_score > 80 else "watch" if risk_score > 40 else "stable"
```

**Estimated Time to Threshold (ETA)** — a simple linear projection of how many readings remain until the risk score crosses the critical threshold, based on the current rate of change:
```python
readings_needed = (threshold − current_score) / score_change
```
This directly addresses the problem statement's requirement to "estimate severity and progression," not just detect a current anomaly. It is a lightweight trend-extrapolation approach, disclosed as a prototype-stage simplification — a full LSTM-based sequence model is the intended production-scale upgrade (see Section 9).

---

## 9. Dashboard (Streamlit)

A live, auto-refreshing web dashboard displays:
- Real-time tilt and vibration metrics
- A gauge visualization of the current risk score (0–100), color-coded (green/yellow/red)
- A live trend chart of risk score over recent readings
- Automated alert banners corresponding to system state (stable / watch / critical / calibrating)

The dashboard reads from a shared `latest_risk.json` file, which the Python monitoring script updates on every processed window — decoupling the sensor-processing script from the dashboard-rendering script as two independent, concurrently running processes.

---

## 10. Known Limitations & Scope Decisions (Prototype vs. Production)

This section is intentionally explicit for transparency during evaluation:

| Design Decision | Reason (4-day constraint) | Production-Scale Path |
|---|---|---|
| Single node, not mesh | Time/budget | LoRa/Zigbee mesh network of multiple nodes, as specified in the original PS |
| No crack/strain sensor | Simplification | Add resistive strain gauge or DIY graphite-strip sensor |
| Laptop-based inference, USB-tethered | Reliability for live demo | Edge (TFLite) inference on-device + cloud aggregation (AWS IoT) for network-level prediction |
| Linear ETA projection | No historical event data available for training a sequence model | LSTM-based multivariate time-series prediction across multiple correlated nodes |
| Servo-simulated vibration events | No access to real subsidence event data | Field validation against real, logged ground movement events over an extended deployment period |
| USB power | Demo simplicity | 18650 Li-ion + TP4056 charging + solar for field deployment |

---

## 11. Libraries and Dependencies

```bash
pip install pyserial numpy scikit-learn streamlit plotly
```

| Library | Purpose |
|---|---|
| `pyserial` | Serial communication with ESP32 |
| `numpy` | Array operations for feature vectors |
| `scikit-learn` | Isolation Forest implementation |
| `streamlit` | Dashboard web framework |
| `plotly` | Gauge and trend chart visualizations |

Arduino/firmware side requires the `MPU6050` and `ESP32Servo` libraries (installable via Arduino IDE Library Manager).

---

## 12. References

- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation Forest*. IEEE International Conference on Data Mining (ICDM).
- scikit-learn documentation: `sklearn.ensemble.IsolationForest`
- Smart India Hackathon 2026 — Problem Statement (Ministry of Coal / Ministry of Mines), Disaster Management theme: *Development of an AI-enabled Low Cost Real Time Mine Subsidence Monitoring, Prediction and Early Warning System for Underground Coal Mines in India*