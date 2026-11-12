# AI-Assisted Smart Vehicle Accident Detection and Emergency Alert System
## Team Member 4 — Cloud & AI Deployment Service

> **Department of Electronics and Communication Engineering**  
> **B.M.S. College of Engineering, Bangalore**  
> **Major Project Work-2 (23EC7PWPJ2)**  
> **Authors**: Meghana R, Mohit Groch, Poornima S, Pratheeksha S Dodamani  
> **Guide**: Dr. Latha H N (Assistant Professor)

---

## 📌 Role & Responsibility (Member 4)

As **Team Member 4 (Cloud & AI Deployment)**, this module owns the cloud infrastructure:
* **Receiving Multi-Modal Data**: Video clip + GPS coordinates + MPU6050 sensor telemetry from the Raspberry Pi.
* **Phase 1 Stub Engine**: Provides realistic simulated responses immediately so Member 1 & Member 2 can integrate without waiting for model training.
* **Phase 2 Real Model Inference**: Modular deployment wrapper ready to ingest Member 3's trained YOLO/PyTorch model behind the exact same API contract.
* **SLA & Timeout Enforcement**: Built-in timeout guards ensuring requests are terminated within 5 seconds if inference hangs, triggering Member 1's offline fail-safe.
* **Strict Section 5 Data Contracts**: Guarantees zero field mismatches between edge and cloud.
* **Resilience & Fault Tolerance**: Catches dropped connections, corrupted video uploads, and malformed inputs with structured error responses.
* **Integration & Load Testing**: Automated test suite executing all Section 6 integration scenarios.

---

## 🏗️ System Architecture & Data Flow

```
+-------------------------------------------------------------------------------+
|                             RASPBERRY PI (EDGE)                               |
|                                                                               |
|   [MPU6050 IMU] ----> Jerk Threshold Exceeded? ----> [Freeze Video Buffer]   |
|   (Member 1)                                          (Member 2 - Pi Cam 3)   |
|                                                                 |             |
|                                                                 v             |
|                                                   Package Clip & GPS Telemetry|
+-----------------------------------------------------------------+-------------+
                                                                  |
                                              Contract A: POST /api/v1/detect
                                              (Multipart: video + metadata)
                                                                  |
                                                                  v
+-------------------------------------------------------------------------------+
|                     MEMBER 4: CLOUD AI INFERENCE SERVICE                      |
|                                                                               |
|   1. Ingestion & Validation (Contract A Schema Validation)                    |
|   2. Timeout SLA Guard (Active cancellation if compute > 5.0s)                |
|   3. Inference Pipeline:                                                      |
|        - Phase 1: StubInferenceEngine (Keyword simulation & latency test)     |
|        - Phase 2: ModelInferenceEngine (YOLOv8/11 / PyTorch / ONNX)           |
|   4. Response Formulation (Contract B format)                                 |
+-----------------------------------------------------------------+-------------+
                                                                  |
                                              Contract B: JSON Response
                                              {event_id, accident_confirmed,
                                               confidence, status}
                                                                  |
                                                                  v
+-------------------------------------------------------------------------------+
|                       MEMBER 1: ALERT & FAIL-SAFE LOGIC                       |
|                                                                               |
|   status == "ok" & accident_confirmed == true  --> Dispatch GSM/GPS SMS Alert |
|   status == "ok" & accident_confirmed == false --> Suppress Alert (Pothole)   |
|   status in ("timeout", "error")               --> Dispatch Precautionary Alert|
+-------------------------------------------------------------------------------+
```

---

## 📋 Data Contracts (Section 5)

### Contract A — Pi ➔ Cloud (`POST /api/v1/detect`)
* `event_id` (`string`): Unique event identifier generated at trigger time.
* `video_clip` (`file binary`): Compressed pre+post impact video clip.
* `timestamp` (`string`): ISO 8601 event time (e.g. `2026-11-11T00:46:13Z`).
* `gps_lat`, `gps_long` (`float`): From NEO-6M GPS (null if no satellite lock).
* `sensor_summary` (`string JSON`): Optional peak acceleration, jerk, and gyro tilt.

### Contract B — Cloud ➔ Pi (`JSON Response`)
```json
{
  "event_id": "evt_crash_1731283200",
  "accident_confirmed": true,
  "confidence": 0.9412,
  "status": "ok"
}
```
* `status` values:
  - `"ok"`: Inference successful.
  - `"timeout"`: Exceeded 5s deadline (triggers Member 1 fail-safe alert).
  - `"error"`: Malformed upload or server issue (triggers Member 1 fail-safe alert).

---

## 🚀 Quick Start Guide

### 1. Installation
Ensure Python 3.10+ is installed:
```bash
cd cloud-ai-deployment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
```
Default `.env` configuration runs **Phase 1 Stub Mode**:
```env
INFERENCE_MODE=stub
TIMEOUT_SECONDS=5.0
CONFIDENCE_THRESHOLD=0.65
```

### 3. Run Development Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation is available at: **http://localhost:8000/docs**

---

## 🧪 Testing & Validation

### 1. Run Unit & Schema Tests
```bash
pytest tests/ -v
```
Verifies Contract A validation, Contract B field compliance, keyword simulation, and timeout safety.

### 2. Simulate Raspberry Pi Upload (Single Event)
```bash
python scripts/simulate_pi_client.py --scenario crash
python scripts/simulate_pi_client.py --scenario pothole
python scripts/simulate_pi_client.py --scenario null_gps
python scripts/simulate_pi_client.py --scenario timeout_test
```

### 3. Run Section 6 Load & Integration Test Suite
```bash
python scripts/load_test.py --url http://localhost:8000
```
Executes all 6 integration scenarios including concurrent load testing (10 simultaneous uploads) and reports P50, P95, and average latency.

---

## 🐳 Docker & Cloud Deployment

### Run with Docker Compose
```bash
cd deploy
docker compose up --build -d
```

### Deploy to Google Cloud Run
```bash
bash deploy/cloud_run_deploy.sh
```

### Deploy to Ubuntu Cloud VM (AWS EC2 / DigitalOcean)
Refer to [deploy/aws_deploy_guide.md](deploy/aws_deploy_guide.md) and [deploy/accident-cloud.service](deploy/accident-cloud.service).

---

## 📂 Project Directory Structure

```
cloud-ai-deployment/
├── app/
│   ├── main.py                  # FastAPI application & error middleware
│   ├── api/
│   │   └── endpoints.py         # /api/v1/detect, /health, /status, /admin/mode
│   ├── core/
│   │   ├── config.py            # Environment configuration
│   │   └── logger.py            # Rotating file and console logger
│   ├── schemas/
│   │   ├── contract_a.py        # Contract A input validation schema
│   │   └── contract_b.py        # Contract B strict output schema
│   └── inference/
│       ├── base.py              # Base inference engine interface
│       ├── stub_engine.py       # Phase 1 dummy/stub simulation
│       ├── model_engine.py      # Phase 2 real AI model inference
│       └── factory.py           # Inference manager with timeout SLA guard
├── tests/                       # Automated pytest test cases
├── scripts/
│   ├── generate_sample_clip.py  # Generates test MP4 video
│   ├── simulate_pi_client.py    # Simulates Member 1 & 2 Pi client
│   └── load_test.py             # Section 6 integration & concurrency load tester
├── deploy/                      # Dockerfile, docker-compose, GCP & AWS guides
├── TEAM_HANDOFF_GUIDE.md        # Dedicated instructions for teammates
├── requirements.txt
└── .env.example
```

