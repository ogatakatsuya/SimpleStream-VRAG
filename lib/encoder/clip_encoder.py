from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


@dataclass
class VideoChunk:
    chunk_index: int
    start_time: float
    end_time: float
    embedding: np.ndarray  # (768,) L2-normalized


class CLIPVideoEncoder:
    MODEL_NAME = "openai/clip-vit-large-patch14-336"
    EMBED_DIM = 768

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
        encode_batch_size: int = 32,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encode_batch_size = encode_batch_size
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode_frames(self, frames: list[Image.Image]) -> np.ndarray:
        """Encode PIL frames and return their L2-normalized mean embedding."""
        all_features: list[torch.Tensor] = []
        for i in range(0, len(frames), self.encode_batch_size):
            batch = frames[i : i + self.encode_batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            features = self.model.get_image_features(pixel_values=pixel_values)
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features)

        stacked = torch.cat(all_features, dim=0)
        mean = stacked.mean(dim=0)
        mean = mean / mean.norm()
        return mean.cpu().numpy()

    @torch.inference_mode()
    def encode_text(self, text: str) -> np.ndarray:
        """Encode a text query and return its L2-normalized embedding."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        features = self.model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy()

    def encode_video(
        self,
        video_path: str | Path,
        fps: float,
        chunk_duration: float = 1.0,
    ) -> list[VideoChunk]:
        """
        Decode video at `fps`, bucket frames into `chunk_duration`-second windows,
        and return one CLIP embedding per chunk.

        The sampling interval is clamped so that each chunk receives at least
        one frame even when `fps` is very low relative to `chunk_duration`.

        Args:
            video_path: Path to the video file.
            fps: Target sampling rate (frames per second).
            chunk_duration: Length of each chunk in seconds.

        Returns:
            List of VideoChunk, one per temporal segment, in order.
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # Upper-bound the interval so at least 1 frame lands in every chunk
        max_interval = max(1, int(chunk_duration * video_fps))
        sample_interval = min(max(1, round(video_fps / fps)), max_interval)

        frame_buckets: dict[int, list[Image.Image]] = {}
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_interval == 0:
                ts = frame_idx / video_fps
                chunk_idx = int(ts // chunk_duration)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_buckets.setdefault(chunk_idx, []).append(Image.fromarray(rgb))
            frame_idx += 1

        cap.release()

        if not frame_buckets:
            return []

        chunks: list[VideoChunk] = []
        for chunk_idx in sorted(frame_buckets):
            embedding = self.encode_frames(frame_buckets[chunk_idx])
            chunks.append(
                VideoChunk(
                    chunk_index=chunk_idx,
                    start_time=chunk_idx * chunk_duration,
                    end_time=(chunk_idx + 1) * chunk_duration,
                    embedding=embedding,
                )
            )

        return chunks
