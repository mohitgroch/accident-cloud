
import cv2
import torch
import numpy as np


def create_video_windows(
    video_path,
    frame_size=224,
    clip_len=16,
    stride=8,
    target_fps=10,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225)
):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)

    if source_fps <= 0:
        cap.release()
        raise ValueError("Invalid video FPS")

    frames = []
    frame_index = 0

    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        timestamp = frame_index / source_fps

        if timestamp * target_fps >= len(frames):
            resized = cv2.resize(
                frame,
                (frame_size, frame_size),
                interpolation=cv2.INTER_LINEAR
            )

            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            tensor = torch.from_numpy(
                np.ascontiguousarray(rgb)
            ).permute(2, 0, 1).float() / 255.0

            tensor = (tensor - mean_tensor) / std_tensor
            frames.append(tensor)

        frame_index += 1

    cap.release()

    if not frames:
        raise ValueError(f"No frames extracted: {video_path}")

    starts = list(
        range(0, max(1, len(frames) - clip_len + 1), stride)
    )

    if len(frames) >= clip_len:
        last_start = len(frames) - clip_len
        if starts[-1] != last_start:
            starts.append(last_start)

    windows = []

    for start in starts:
        clip_frames = frames[start:start + clip_len]

        if len(clip_frames) == clip_len:
            windows.append(torch.stack(clip_frames))

    if not windows:
        raise ValueError(
            f"Video too short to form a {clip_len}-frame window"
        )

    return torch.stack(windows)

