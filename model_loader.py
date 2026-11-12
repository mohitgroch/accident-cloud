
from pathlib import Path
from types import SimpleNamespace

import torch

from .mobilenet_temporal import MobileNetTemporal


def load_accident_model(checkpoint_path, device=None):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_config = SimpleNamespace(
        pretrained=False,
        temporal_hidden=256,
        dropout=0.2
    )

    model = MobileNetTemporal.from_config(model_config)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.to(device)
    model.eval()

    return model, device

