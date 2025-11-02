import os
import io
import base64
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_v2_s as tv_efficientnet_v2_s
from torchvision.models import EfficientNet_V2_S_Weights as TV_EFFNET_V2_S_WEIGHTS


class EfficientNetV2Binary(nn.Module):
    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        weights = TV_EFFNET_V2_S_WEIGHTS.IMAGENET1K_V1
        tv_model = tv_efficientnet_v2_s(weights=weights)
        in_features = tv_model.classifier[-1].in_features
        tv_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )
        self.backbone = tv_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(1)


class FaceDetector:
    def __init__(self, margin: int = 20) -> None:
        self.margin = margin
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_and_crop(self, img_rgb: np.ndarray, out_size: int) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        # Slightly more permissive parameters to improve recall on images
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)
        if len(faces) == 0:
            return None
        # Use largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        h_img, w_img = img_rgb.shape[:2]
        m = self.margin
        x1 = max(0, int(x) - m)
        y1 = max(0, int(y) - m)
        x2 = min(w_img, int(x + w) + m)
        y2 = min(h_img, int(y + h) + m)
        if x2 <= x1 or y2 <= y1:
            return None
        face = img_rgb[y1:y2, x1:x2]
        return cv2.resize(face, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


class Preprocessor:
    def __init__(self, image_size: int = 256) -> None:
        self.image_size = image_size
        self.detector = FaceDetector(margin=20)
        self.resize = transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR)
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def prepare_rgb(self, rgb: np.ndarray) -> Optional[torch.Tensor]:
        crop = self.detector.detect_and_crop(rgb, self.image_size)
        if crop is None:
            return None
        pil_like = transforms.functional.to_pil_image(crop)
        pil_like = self.resize(pil_like)
        return self.to_tensor(pil_like)


def load_model(weights_path: str, device: torch.device) -> EfficientNetV2Binary:
    model = EfficientNetV2Binary(dropout=0.3)
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_image(model: nn.Module, pre: Preprocessor, image_bgr: np.ndarray, device: torch.device) -> Dict[str, float]:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    t = pre.prepare_rgb(rgb)
    no_face = False
    if t is None:
        # Center-crop fallback for images where face detection fails
        h, w = rgb.shape[:2]
        side = min(h, w)
        y1 = (h - side) // 2
        x1 = (w - side) // 2
        crop = rgb[y1:y1+side, x1:x1+side]
        crop = cv2.resize(crop, (pre.image_size, pre.image_size), interpolation=cv2.INTER_LINEAR)
        pil_like = transforms.functional.to_pil_image(crop)
        pil_like = pre.resize(pil_like)
        t = pre.to_tensor(pil_like)
        no_face = True
    batch = t.unsqueeze(0).to(device)
    logits = model(batch)
    prob = torch.sigmoid(logits)[0].item()
    label = 1 if prob >= 0.5 else 0
    return {"prob": float(prob), "label": int(label), "no_face_detected": bool(no_face)}


def gradcam_on_image(model: EfficientNetV2Binary, pre: Preprocessor, image_bgr: np.ndarray, device: torch.device) -> Tuple[Optional[np.ndarray], Optional[float]]:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    crop = pre.detector.detect_and_crop(rgb, pre.image_size)
    if crop is None:
        return None, None
    pil_like = transforms.functional.to_pil_image(crop)
    pil_like = pre.resize(pil_like)
    t = pre.to_tensor(pil_like).unsqueeze(0).to(device)

    target_module = model.backbone.features[-1]
    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    def fwd_hook(_, __, output):
        activations.append(output)

    def bwd_hook(_, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = target_module.register_forward_hook(fwd_hook)
    h2 = target_module.register_full_backward_hook(bwd_hook)

    logits = model(t)
    prob = torch.sigmoid(logits)[0]
    model.zero_grad(set_to_none=True)
    logits.backward(torch.ones_like(logits))

    h1.remove(); h2.remove()
    if not activations or not gradients:
        return None, float(prob.item())

    act = activations[0]  # [1, C, H, W]
    grad = gradients[-1]  # [1, C, H, W]
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
    cam = torch.relu(torch.sum(weights * act, dim=1, keepdim=False))  # [1, H, W]
    cam = cam[0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (crop.shape[1], crop.shape[0]))
    heatmap = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)
    return overlay, float(prob.item())


def encode_image_b64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buf.tobytes()).decode('utf-8')



