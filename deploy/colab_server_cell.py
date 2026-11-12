# ==============================================================================
# PASTE THIS CELL AT THE VERY BOTTOM OF YOUR GOOGLE COLAB NOTEBOOK
# (For Team Member 3 — to connect your model live to Member 4's Cloud Service)
# ==============================================================================

# 1. Install lightweight server dependencies inside Colab
!pip install -q fastapi uvicorn pyngrok python-multipart nest-asyncio

import nest_asyncio
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from pyngrok import ngrok
import os
import shutil

app = FastAPI(title="Colab MobileNetTemporal Inference Service")

# Apply nest_asyncio so uvicorn can run inside Colab's asyncio loop
nest_asyncio.apply()

@app.get("/")
def home():
    return {
        "status": "online",
        "model": "MobileNetTemporal (Crash-Anticipation)",
        "framework": "PyTorch",
        "threshold": 0.50,
        "consecutive_windows": 2
    }

@app.post("/predict")
async def predict_api(video: UploadFile = File(...), event_id: str = Form("unknown")):
    temp_path = f"/content/temp_{video.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Calls Member 3's exact inference function from Cell 168:
        result = predict_video_fast(temp_path, threshold=0.50, consecutive_windows=2)
        
        # Return format expected by Member 4's cloud service
        return {
            "event_id": event_id,
            "prediction": result["prediction"],                # "CRASH CANDIDATE" or "NORMAL"
            "max_probability": max(result["window_probabilities"]) if result["window_probabilities"] else 0.0,
            "window_probabilities": result["window_probabilities"],
            "total_windows": result["total_windows"],
            "total_processing_time": result["total_processing_time"]
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Authenticate ngrok (get free token from https://dashboard.ngrok.com)
# NGROK_AUTH_TOKEN = "your_ngrok_token_here"
# ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Expose port 8000
public_url = ngrok.connect(8000)
print("\n" + "=" * 65)
print(f"🚀 COLAB MODEL API IS LIVE AT: {public_url}")
print("Copy this URL and give it to Member 4 to put in .env as COLAB_TUNNEL_URL!")
print("=" * 65 + "\n")

uvicorn.run(app, host="0.0.0.0", port=8000)

