# Cloud Deployment Guide (AWS, GCP, Render, or VPS)

This guide outlines deployment options for **Team Member 4 (Cloud & AI Deployment)**.

---

## Option 1: Google Cloud Run (Recommended for Low Cost & Autoscaling)

GCP Cloud Run provides automatic HTTPS, zero idle costs when traffic is low, and container scaling.

1. **Prerequisites**: Install the `gcloud` CLI and authenticate.
2. Run the deployment script:
   ```bash
   bash deploy/cloud_run_deploy.sh
   ```
3. Use the generated HTTPS URL as the endpoint for Member 2's Raspberry Pi script:
   `https://accident-detection-cloud-xyz.a.run.app/api/v1/detect`

---

## Option 2: AWS App Runner or EC2 (Ubuntu 22.04 / 24.04)

### Deploying on an EC2 Instance (t3.medium or g4dn.xlarge if GPU inference):
1. **Launch EC2 instance** with Ubuntu 22.04 LTS and attach an Elastic IP.
2. **Configure Security Group**: Open inbound port `8000` (or `80`/`443`) to the Raspberry Pi's cellular IP range or `0.0.0.0/0`.
3. **SSH into the instance**:
   ```bash
   ssh -i key.pem ubuntu@<EC2_PUBLIC_IP>
   ```
4. **Clone / copy repository & setup environment**:
   ```bash
   sudo apt update && sudo apt install -y python3-pip python3-venv git ffmpeg libgl1
   git clone <repo-url>
   cd cloud-ai-deployment
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
5. **Install and start systemd service**:
   ```bash
   sudo cp deploy/accident-cloud.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable accident-cloud
   sudo systemctl start accident-cloud
   sudo systemctl status accident-cloud
   ```

---

## Option 3: Docker / Docker Compose (Fastest local or remote)

Run locally or on any server with Docker installed:
```bash
docker compose -f deploy/docker-compose.yml up --build -d
```
Check health:
```bash
curl http://localhost:8000/health
```

---

## Option 4: Render / Railway / Fly.io (One-Click Free/Low-Cost Cloud)

1. Connect this repository to **Render** or **Railway**.
2. Set Build Command: `pip install -r requirements.txt`
3. Set Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Copy environment variables from `.env.example`.

