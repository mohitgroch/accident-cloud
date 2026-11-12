
import time
import torch

from .preprocessing import create_video_windows


class AccidentPredictor:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        video_path,
        threshold=0.50,
        consecutive_windows=2
    ):
        start = time.perf_counter()

        windows = create_video_windows(video_path)
        preprocessing_time = time.perf_counter() - start

        windows = windows.to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        inference_start = time.perf_counter()

        logits = self.model(windows)["logits"]
        probabilities = torch.sigmoid(logits).cpu().numpy().flatten()

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        inference_time = time.perf_counter() - inference_start

        window_predictions = probabilities >= threshold

        consecutive_count = 0
        accident_detected = False

        for prediction in window_predictions:
            if prediction:
                consecutive_count += 1
                if consecutive_count >= consecutive_windows:
                    accident_detected = True
                    break
            else:
                consecutive_count = 0

        return {
            "video_path": str(video_path),
            "prediction": (
                "ACCIDENT" if accident_detected else "NORMAL"
            ),
            "probabilities": probabilities.tolist(),
            "windows": len(probabilities),
            "preprocessing_time": preprocessing_time,
            "inference_time": inference_time,
            "total_processing_time": (
                preprocessing_time + inference_time
            )
        }

