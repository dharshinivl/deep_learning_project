import os
import sys
import json
import math
import time
import random
import shutil
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

try:
    from facenet_pytorch import MTCNN
    FACENET_AVAILABLE = True
except Exception:
    MTCNN = None
    FACENET_AVAILABLE = False

try:
    import timm  # optional; we fall back to torchvision if unavailable
except Exception:
    timm = None
from torchvision.models import efficientnet_v2_s as tv_efficientnet_v2_s
from torchvision.models import EfficientNet_V2_S_Weights as TV_EFFNET_V2_S_WEIGHTS


# ================================
# Configuration
# ================================
@dataclass
class TrainConfig:
    # Paths
    dataset_root: str = os.environ.get("DFDC_DATASET_ROOT", "./data/DFDC")  # root containing videos and labels.csv
    labels_csv: str = os.environ.get("DFDC_LABELS_CSV", "./data/DFDC/labels.csv")
    processed_root: str = os.environ.get("DFDC_PROCESSED_ROOT", "./data/processed_faces")  # face crops cache
    output_dir: str = os.environ.get("OUTPUT_DIR", "./outputs")
    # Auto-download (Kaggle). If labels.csv is missing, try to download.
    auto_download: bool = True # set True to enable automatic Kaggle download
    download_full: bool = False  # False -> download sample zip only; True -> full competition (HUGE)
    kaggle_competition: str = "deepfake-detection-challenge"
    # Alternative dataset: Celeb-DF
    use_celebf: bool = True  # Use Celeb-DF dataset instead of DFDC
    celebf_dataset: str = "deepfake-detection-challenge"  # Kaggle dataset name for Celeb-DF
    # Use custom testing videos file
    use_custom_txt: bool = False  # Use custom testing_videos.txt file
    custom_txt_path: str = os.path.join(dataset_root, "List_of_testing_videos.txt")  # Path to your testing videos file
    # Auto-scan video folders
    auto_scan_folders: bool = True  # Automatically scan video folders and create labels
    # Dataset balancing
    balance_dataset: bool = True  # Balance real/fake ratio
    max_fake_ratio: float = 2.5  # Max fake:real ratio (e.g., 2.5:1)
    # If you want kaggle.json inside the project, set this (default ./kaggle.json)
    kaggle_json_path: str = os.environ.get("KAGGLE_JSON_PATH", "./kaggle.json")

    # Data
    image_size: int = 256
    frames_per_video: int = 16
    min_frames_interval: int = 2  # minimum stride between sampled frames
    cache_faces_to_disk: bool = True
    use_face_cache_if_exists: bool = True
    face_detection_margin: int = 20  # extra pixels around detected bbox
    mtcnn_device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataloader
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

    # Model
    backbone_name: str = "efficientnetv2_s"
    dropout: float = 0.3

    # Train
    epochs: int = 30  # More epochs for better convergence
    learning_rate: float = 1e-4  # Lower LR for stability
    weight_decay: float = 1e-4
    pos_weight: Optional[float] = 1.2  # Handle class imbalance (more fake videos)
    grad_accum_steps: int = 2  # Effective batch size = 32 * 2 = 64
    mixed_precision: bool = True

    # Scheduler
    reduce_on_plateau_factor: float = 0.5
    reduce_on_plateau_patience: int = 2
    reduce_on_plateau_min_lr: float = 1e-6

    # Split
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42

    # Logging / Checkpoint
    save_best_only: bool = True
    best_model_name: str = "best_model.pth"
    log_interval: int = 50


CFG = TrainConfig()


# ================================
# Utilities
# ================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def video_id_from_path(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


def compute_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:8]


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    imgs = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    return imgs, labels


# ================================
# Dataset: DFDC video -> face crops
# ================================
class VideoFaceDataset(Dataset):
    def __init__(
        self,
        items: List[Tuple[str, int]],
        cfg: TrainConfig,
        mtcnn: Optional[MTCNN],
        augment: bool,
    ) -> None:
        self.items = items
        self.cfg = cfg
        self.mtcnn = mtcnn
        self.augment = augment

        # Transforms
        aug_t = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)], p=0.5),
            transforms.RandomRotation(degrees=10, interpolation=transforms.InterpolationMode.BILINEAR),
        ] if augment else []

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.pre_resize = transforms.Resize((cfg.image_size, cfg.image_size), interpolation=transforms.InterpolationMode.BILINEAR)
        self.augmentations = transforms.Compose(aug_t)

    def __len__(self) -> int:
        return len(self.items)

    def _cached_faces_dir(self, video_path: str) -> str:
        vid = video_id_from_path(video_path)
        # Use a short hash to differentiate identical names across folders
        vhash = compute_hash(os.path.abspath(video_path))
        faces_dir = os.path.join(self.cfg.processed_root, f"{vid}_{vhash}")
        return faces_dir

    def _extract_and_cache_faces(self, video_path: str) -> List[str]:
        faces_dir = self._cached_faces_dir(video_path)
        ensure_dir(faces_dir)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        # Determine frame indices to sample uniformly
        num_samples = max(1, self.cfg.frames_per_video)
        indices = np.linspace(0, total_frames - 1, num=num_samples, dtype=int)
        # Enforce minimal spacing
        indices = np.unique(np.clip(indices, 0, total_frames - 1))
        if len(indices) > 1:
            diffs = np.diff(indices)
            # If spacing is too tight, thin out
            keep = [0]
            for i, d in enumerate(diffs, start=1):
                if d >= self.cfg.min_frames_interval:
                    keep.append(i)
            indices = indices[keep]

        saved_paths = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            # BGR -> RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run face detection (MTCNN or OpenCV)
            try:
                if self.mtcnn is not None:
                    # MTCNN detection
                    boxes, probs = self.mtcnn.detect(rgb)
                    if boxes is None or len(boxes) == 0 or (probs is not None and float(np.max(probs)) < 0.7):
                        continue
                    # select highest prob box
                    best_idx = int(np.argmax(probs)) if probs is not None else 0
                    x1, y1, x2, y2 = boxes[best_idx]
                else:
                    # OpenCV Haar cascade fallback
                    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    if len(faces) == 0:
                        continue
                    # Use largest face
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = largest_face
                    x1, y1, x2, y2 = x, y, x + w, y + h
                
                h, w = rgb.shape[:2]
                m = self.cfg.face_detection_margin
                x1 = max(0, int(x1) - m)
                y1 = max(0, int(y1) - m)
                x2 = min(w, int(x2) + m)
                y2 = min(h, int(y2) + m)
                if x2 <= x1 or y2 <= y1:
                    continue
                face = rgb[y1:y2, x1:x2]
            except Exception:
                continue

            if face.size == 0:
                continue

            face = cv2.resize(face, (self.cfg.image_size, self.cfg.image_size), interpolation=cv2.INTER_LINEAR)
            # Save to cache
            out_path = os.path.join(faces_dir, f"frame_{int(idx):06d}.jpg")
            try:
                cv2.imwrite(out_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                saved_paths.append(out_path)
            except Exception:
                # ignore failed write
                pass

        cap.release()
        return saved_paths

    def _load_face_paths(self, video_path: str) -> List[str]:
        faces_dir = self._cached_faces_dir(video_path)
        if self.cfg.use_face_cache_if_exists and os.path.isdir(faces_dir):
            files = [os.path.join(faces_dir, f) for f in os.listdir(faces_dir) if f.lower().endswith(".jpg")]
            if len(files) > 0:
                files.sort()
                # possibly subsample to frames_per_video
                if len(files) > self.cfg.frames_per_video:
                    idxs = np.linspace(0, len(files) - 1, num=self.cfg.frames_per_video, dtype=int)
                    files = [files[i] for i in idxs]
                return files

        if not self.cfg.cache_faces_to_disk:
            # When not caching, still extract but return empty (handled per __getitem__)
            return []

        # Extract and cache if missing
        return self._extract_and_cache_faces(video_path)

    def _choose_face_path(self, face_paths: List[str]) -> Optional[str]:
        if not face_paths:
            return None
        # Randomly choose one face crop among available frames for this sample
        return random.choice(face_paths)

    def __getitem__(self, idx: int):
        try:
            video_path, label = self.items[idx]
            face_paths = self._load_face_paths(video_path)
            face_path = self._choose_face_path(face_paths)

            if face_path is None:
                # Fallback to on-the-fly extraction if caching disabled or no cache produced
                # For performance, we try a quick single center frame
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                center = max(0, total_frames // 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(center))
                ok, frame = cap.read()
                cap.release()
                if not ok or frame is None:
                    return None
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    if self.mtcnn is not None:
                        # MTCNN detection
                        boxes, probs = self.mtcnn.detect(rgb)
                        if boxes is None or len(boxes) == 0 or (probs is not None and float(np.max(probs)) < 0.7):
                            return None
                        best_idx = int(np.argmax(probs)) if probs is not None else 0
                        x1, y1, x2, y2 = boxes[best_idx]
                    else:
                        # OpenCV Haar cascade fallback
                        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                        if len(faces) == 0:
                            return None
                        # Use largest face
                        largest_face = max(faces, key=lambda f: f[2] * f[3])
                        x, y, w, h = largest_face
                        x1, y1, x2, y2 = x, y, x + w, y + h
                    
                    h, w = rgb.shape[:2]
                    m = self.cfg.face_detection_margin
                    x1 = max(0, int(x1) - m)
                    y1 = max(0, int(y1) - m)
                    x2 = min(w, int(x2) + m)
                    y2 = min(h, int(y2) + m)
                    if x2 <= x1 or y2 <= y1:
                        return None
                    face = rgb[y1:y2, x1:x2]
                except Exception:
                    return None
                if face.size == 0:
                    return None
                img = cv2.resize(face, (self.cfg.image_size, self.cfg.image_size), interpolation=cv2.INTER_LINEAR)
            else:
                img_bgr = cv2.imread(face_path, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    return None
                img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            pil_like = transforms.functional.to_pil_image(img)
            if self.augment:
                pil_like = self.augmentations(pil_like)
            pil_like = self.pre_resize(pil_like)
            tensor = self.transform(pil_like)
            return tensor, float(label)
        except Exception:
            return None


# ================================
# Model
# ================================
class EfficientNetV2Binary(nn.Module):
    def __init__(self, backbone_name: str, dropout: float = 0.3) -> None:
        super().__init__()
        self.using_timm = False
        if timm is not None:
            try:
                self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool="avg")
                in_features = self.backbone.num_features if hasattr(self.backbone, "num_features") else 1280
                self.head = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features, 1),
                )
                self.using_timm = True
            except Exception:
                self.using_timm = False
        if not self.using_timm:
            # Fallback to torchvision EfficientNetV2-S
            weights = TV_EFFNET_V2_S_WEIGHTS.IMAGENET1K_V1
            tv_model = tv_efficientnet_v2_s(weights=weights)
            # Replace classifier with custom binary head
            # Original classifier: Dropout, Linear(1280, 1000)
            in_features = tv_model.classifier[-1].in_features
            tv_model.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, 1),
            )
            self.backbone = tv_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.using_timm:
            feats = self.backbone(x)
            logits = self.head(feats).squeeze(1)
            return logits
        # Torchvision path returns [N,1]; squeeze to [N]
        logits = self.backbone(x).squeeze(1)
        return logits


# ================================
# Metrics
# ================================
@torch.no_grad()
def compute_metrics_from_outputs(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()
    labels_long = labels.long()

    tp = torch.sum((preds == 1) & (labels_long == 1)).item()
    tn = torch.sum((preds == 0) & (labels_long == 0)).item()
    fp = torch.sum((preds == 1) & (labels_long == 0)).item()
    fn = torch.sum((preds == 0) & (labels_long == 1)).item()

    eps = 1e-8
    accuracy = (tp + tn) / max(1, (tp + tn + fp + fn))
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 2 * precision * recall / max(eps, (precision + recall))

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


# ================================
# Data split utilities
# ================================
def stratified_split(df: pd.DataFrame, label_col: str, val_ratio: float, test_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    train_parts, val_parts, test_parts = [], [], []
    for label_value, group in df.groupby(label_col):
        idx = np.arange(len(group))
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        test_idx = idx[:n_test]
        val_idx = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]
        test_parts.append(group.iloc[test_idx])
        val_parts.append(group.iloc[val_idx])
        train_parts.append(group.iloc[train_idx])
    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, val_df, test_df


def read_labels(labels_csv: str, dataset_root: str) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    # Expected columns: video_path (relative or absolute), label in {0,1}
    if "video_path" not in df.columns or "label" not in df.columns:
        raise ValueError("labels.csv must contain columns: video_path,label")
    def resolve_path(p):
        p = str(p)
        if os.path.isabs(p):
            return p
        return os.path.join(dataset_root, p)
    df["video_path"] = df["video_path"].map(resolve_path)
    # Filter existing files only
    df = df[df["video_path"].map(os.path.isfile)].reset_index(drop=True)
    return df


# ================================
# Kaggle download helpers (optional)
# ================================
def kaggle_available() -> bool:
    try:
        import importlib.util
        spec = importlib.util.find_spec("kaggle")
        return spec is not None
    except Exception:
        return False


def unzip_file(zip_path: str, dst_dir: str) -> None:
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dst_dir)


def generate_labels_from_metadata(root: str, out_csv: str) -> int:
    rows = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "metadata.json" in filenames:
            meta_path = os.path.join(dirpath, "metadata.json")
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                continue
            for fname, info in meta.items():
                label = 1 if str(info.get("label", "")).lower() == "fake" else 0
                fullp = os.path.join(dirpath, fname)
                if os.path.isfile(fullp):
                    rows.append([fullp, label])
    if len(rows) == 0:
        return 0
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["video_path", "label"])
        w.writerows(rows)
    return len(rows)


def maybe_kaggle_download(cfg: TrainConfig) -> None:
    if not cfg.auto_download:
        return
    if os.path.isfile(cfg.labels_csv):
        return
    if not kaggle_available():
        print("Kaggle package not available; install 'kaggle' or disable auto_download.")
        return
    from kaggle.api.kaggle_api_extended import KaggleApi
    # Prefer project-local kaggle.json if present
    try:
        if cfg.kaggle_json_path and os.path.isfile(cfg.kaggle_json_path):
            kj_abs = os.path.abspath(cfg.kaggle_json_path)
            kdir = os.path.dirname(kj_abs) or "."
            # Ensure filename is kaggle.json as expected by Kaggle client
            if os.path.basename(kj_abs) != "kaggle.json":
                dst = os.path.join(kdir, "kaggle.json")
                try:
                    shutil.copy(kj_abs, dst)
                    kj_abs = dst
                except Exception:
                    pass
            os.environ["KAGGLE_CONFIG_DIR"] = kdir
    except Exception:
        pass
    ensure_dir(cfg.dataset_root)
    print("Attempting Kaggle download using your kaggle.json (ensure it is in %USERPROFILE%/.kaggle).")
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        print(f"Kaggle authentication failed: {e}")
        return

    if cfg.use_celebf:
        # Try to download Celeb-DF dataset
        print("Trying to download Celeb-DF dataset...")
        try:
            # Try different possible dataset names for Celeb-DF
            celebf_datasets = [
                "deepfake-detection-challenge",
                "celebf-dataset", 
                "celebf",
                "deepfake-detection"
            ]
            
            downloaded = False
            for dataset_name in celebf_datasets:
                try:
                    print(f"Trying dataset: {dataset_name}")
                    api.dataset_download_files(dataset_name, path=cfg.dataset_root, unzip=True, quiet=False)
                    downloaded = True
                    print(f"Successfully downloaded from {dataset_name}")
                    break
                except Exception as e:
                    print(f"Failed to download from {dataset_name}: {e}")
                    continue
            
            if not downloaded:
                print("Could not find Celeb-DF dataset. Creating a simple test setup...")
                create_simple_celebf_setup(cfg)
                return
                
        except Exception as e:
            print(f"Celeb-DF download failed: {e}")
            print("Creating a simple test setup...")
            create_simple_celebf_setup(cfg)
            return
    else:
        # Original DFDC download logic (likely won't work)
        if cfg.download_full:
            print("Downloading full DFDC (this is VERY large)...")
            api.competition_download_files(cfg.kaggle_competition, path=cfg.dataset_root, quiet=False)
        else:
            print("DFDC dataset no longer available. Creating test setup...")
            create_simple_celebf_setup(cfg)
            return

    # Generate labels.csv from extracted metadata or create simple setup
    print("Generating labels.csv...")
    ensure_dir(os.path.dirname(cfg.labels_csv) or ".")
    
    # Try to find metadata files first
    metadata_found = False
    for root, dirs, files in os.walk(cfg.dataset_root):
        if "metadata.json" in files:
            metadata_found = True
            break
    
    if metadata_found:
        n = generate_labels_from_metadata(cfg.dataset_root, cfg.labels_csv)
        print(f"Labels generated from metadata: {n} entries at {cfg.labels_csv}")
    else:
        print("No metadata found, creating simple test setup...")
        create_simple_celebf_setup(cfg)


def create_celebf_labels_from_txt(cfg: TrainConfig, txt_file_path: str) -> None:
    """Create labels.csv from the provided testing_videos.txt file with optional balancing."""
    import csv
    import random
    
    ensure_dir(cfg.dataset_root)
    
    # Read the testing videos file
    real_videos = []
    fake_videos = []
    
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Parse format: "1 YouTube-real/00170.mp4" or "0 Celeb-synthesis/id1_id0_0007.mp4"
                parts = line.split(' ', 1)
                if len(parts) != 2:
                    print(f"Warning: Skipping malformed line {line_num}: {line}")
                    continue
                
                label_str, video_path = parts
                
                # Convert label: 1 = real, 0 = fake (but we want 1 = fake, 0 = real for binary classification)
                # So we flip: 1 -> 0 (real), 0 -> 1 (fake)
                label = 1 - int(label_str)  # Flip the labels
                
                # Construct full path (assuming videos are in the dataset_root)
                full_video_path = os.path.join(cfg.dataset_root, video_path)
                
                if label == 0:  # Real
                    real_videos.append([full_video_path, label])
                else:  # Fake
                    fake_videos.append([full_video_path, label])
        
        print(f"Original dataset:")
        print(f"Real videos: {len(real_videos)}, Fake videos: {len(fake_videos)}")
        print(f"Original ratio: {len(fake_videos)/len(real_videos):.1f}:1 (fake:real)")
        
        # Balance dataset if requested
        if cfg.balance_dataset and len(fake_videos) > len(real_videos) * cfg.max_fake_ratio:
            # Calculate how many fake videos to keep
            max_fake_videos = int(len(real_videos) * cfg.max_fake_ratio)
            
            # Randomly sample fake videos
            random.seed(cfg.random_seed)
            fake_videos = random.sample(fake_videos, max_fake_videos)
            
            print(f"\nBalanced dataset:")
            print(f"Real videos: {len(real_videos)}, Fake videos: {len(fake_videos)}")
            print(f"New ratio: {len(fake_videos)/len(real_videos):.1f}:1 (fake:real)")
        
        # Combine all videos
        labels_data = real_videos + fake_videos
        
        # Shuffle the dataset
        random.seed(cfg.random_seed)
        random.shuffle(labels_data)
        
        print(f"\nFinal dataset: {len(labels_data)} videos")
        
        # Write labels.csv
        with open(cfg.labels_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(["video_path", "label"])
            w.writerows(labels_data)
        
        print(f"Labels saved to {cfg.labels_csv}")
        
    except FileNotFoundError:
        print(f"Testing videos file not found: {txt_file_path}")
        print("Please ensure the file exists and try again.")
        raise
    except Exception as e:
        print(f"Error processing testing videos file: {e}")
        raise


def auto_scan_video_folders(cfg: TrainConfig) -> None:
    """Automatically scan video folders and create labels.csv."""
    import csv
    import random
    
    ensure_dir(cfg.dataset_root)
    
    # Define folder structure and labels
    video_folders = [
        ("YouTube-real", 0),    # Real videos
        ("Celeb-real", 0),      # Real videos  
        ("Celeb-synthesis", 1), # Fake videos
    ]
    
    real_videos = []
    fake_videos = []
    
    print("Scanning video folders...")
    
    for folder_name, label in video_folders:
        folder_path = os.path.join(cfg.dataset_root, folder_name)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found: {folder_path}")
            continue
            
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        video_files = []
        
        for ext in video_extensions:
            pattern = os.path.join(folder_path, f"*{ext}")
            import glob
            video_files.extend(glob.glob(pattern))
        
        print(f"Found {len(video_files)} videos in {folder_name}")
        
        # Add to appropriate list
        for video_path in video_files:
            # Convert to absolute path to avoid path issues
            abs_video_path = os.path.abspath(video_path)
            if label == 0:  # Real
                real_videos.append([abs_video_path, label])
            else:  # Fake
                fake_videos.append([abs_video_path, label])
    
    print(f"\nOriginal dataset:")
    print(f"Real videos: {len(real_videos)}, Fake videos: {len(fake_videos)}")
    if len(real_videos) > 0:
        print(f"Original ratio: {len(fake_videos)/len(real_videos):.1f}:1 (fake:real)")
    
    # Balance dataset if requested
    if cfg.balance_dataset and len(fake_videos) > len(real_videos) * cfg.max_fake_ratio:
        # Calculate how many fake videos to keep
        max_fake_videos = int(len(real_videos) * cfg.max_fake_ratio)
        
        # Randomly sample fake videos
        random.seed(cfg.random_seed)
        fake_videos = random.sample(fake_videos, max_fake_videos)
        
        print(f"\nBalanced dataset:")
        print(f"Real videos: {len(real_videos)}, Fake videos: {len(fake_videos)}")
        print(f"New ratio: {len(fake_videos)/len(real_videos):.1f}:1 (fake:real)")
    
    # Combine all videos
    labels_data = real_videos + fake_videos
    
    # Shuffle the dataset
    random.seed(cfg.random_seed)
    random.shuffle(labels_data)
    
    print(f"\nFinal dataset: {len(labels_data)} videos")
    
    # Write labels.csv
    with open(cfg.labels_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["video_path", "label"])
        w.writerows(labels_data)
    
    print(f"Labels saved to {cfg.labels_csv}")


def create_simple_celebf_setup(cfg: TrainConfig) -> None:
    """Create a simple Celeb-DF style test setup."""
    import tempfile
    import shutil
    
    ensure_dir(cfg.dataset_root)
    
    # Create a simple test structure
    test_videos = [
        ("real_001.mp4", 0),
        ("real_002.mp4", 0), 
        ("real_003.mp4", 0),
        ("fake_001.mp4", 1),
        ("fake_002.mp4", 1),
        ("fake_003.mp4", 1),
    ]
    
    video_dir = os.path.join(cfg.dataset_root, "videos")
    ensure_dir(video_dir)
    
    # Create dummy video files (just text files for now)
    for video_name, label in test_videos:
        video_path = os.path.join(video_dir, video_name)
        if not os.path.exists(video_path):
            with open(video_path, 'w') as f:
                f.write(f"# Test video file: {video_name} (label: {label})\n")
                f.write("# This is a placeholder. Replace with actual video files for real training.\n")
    
    # Create labels.csv
    labels_data = []
    for video_name, label in test_videos:
        video_path = os.path.join(video_dir, video_name)
        labels_data.append([video_path, label])
    
    with open(cfg.labels_csv, 'w', newline='', encoding='utf-8') as f:
        import csv
        w = csv.writer(f)
        w.writerow(["video_path", "label"])
        w.writerows(labels_data)
    
    print(f"Created Celeb-DF style test setup with {len(test_videos)} videos at {cfg.dataset_root}")
    print(f"Labels saved to {cfg.labels_csv}")
    print("Note: These are placeholder files. Replace with actual video files for real training.")


# ================================
# Train & Eval
# ================================
def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device, scaler: Optional[torch.cuda.amp.GradScaler], log_interval: int, grad_accum_steps: int = 1) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_tp = running_tn = running_fp = running_fn = 0
    total_samples = 0

    step = 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        if batch is None:
            continue
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, labels)
            loss = loss / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Metrics
        with torch.no_grad():
            metrics = compute_metrics_from_outputs(logits, labels)
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size * grad_accum_steps
            running_tp += metrics["tp"]
            running_tn += metrics["tn"]
            running_fp += metrics["fp"]
            running_fn += metrics["fn"]
            total_samples += batch_size

        if (step + 1) % log_interval == 0:
            acc = (running_tp + running_tn) / max(1, total_samples)
            pbar.set_postfix({"loss": f"{running_loss / max(1,total_samples):.4f}", "acc": f"{acc:.4f}"})

        step += 1

    eps = 1e-8
    precision = running_tp / max(1, (running_tp + running_fp))
    recall = running_tp / max(1, (running_tp + running_fn))
    f1 = 2 * precision * recall / max(eps, (precision + recall))
    accuracy = (running_tp + running_tn) / max(1, total_samples)
    epoch_loss = running_loss / max(1, total_samples)
    return {"loss": epoch_loss, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    running_tp = running_tn = running_fp = running_fn = 0
    total_samples = 0

    pbar = tqdm(loader, desc="Eval", leave=False)
    for batch in pbar:
        if batch is None:
            continue
        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        metrics = compute_metrics_from_outputs(logits, labels)
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_tp += metrics["tp"]
        running_tn += metrics["tn"]
        running_fp += metrics["fp"]
        running_fn += metrics["fn"]
        total_samples += batch_size

    eps = 1e-8
    precision = running_tp / max(1, (running_tp + running_fp))
    recall = running_tp / max(1, (running_tp + running_fn))
    f1 = 2 * precision * recall / max(eps, (precision + recall))
    accuracy = (running_tp + running_tn) / max(1, total_samples)
    epoch_loss = running_loss / max(1, total_samples)
    return {"loss": epoch_loss, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# ================================
# Main
# ================================
def main() -> None:
    set_seed(CFG.random_seed)
    ensure_dir(CFG.output_dir)
    ensure_dir(CFG.processed_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Configuration:\n" + json.dumps(asdict(CFG), indent=2))
    print(f"Using device: {device}")

    # Ensure dataset (auto-scan folders, custom txt file, or auto-download)
    if CFG.auto_scan_folders:
        print("Auto-scanning video folders...")
        auto_scan_video_folders(CFG)
    elif CFG.use_custom_txt and os.path.isfile(CFG.custom_txt_path):
        print(f"Using custom testing videos file: {CFG.custom_txt_path}")
        create_celebf_labels_from_txt(CFG, CFG.custom_txt_path)
    else:
        maybe_kaggle_download(CFG)
        
        # Check if labels file exists after download attempt
        if not os.path.isfile(CFG.labels_csv):
            print(f"Labels file not found: {CFG.labels_csv}")
            print("Please ensure the dataset is downloaded and labels.csv is generated.")
            sys.exit(1)
    
    df = read_labels(CFG.labels_csv, CFG.dataset_root)
    if len(df) == 0:
        print("No labeled videos found. Ensure labels.csv has columns 'video_path,label' and files exist.")
        sys.exit(1)

    # Stratified split
    train_df, val_df, test_df = stratified_split(df, label_col="label", val_ratio=CFG.val_ratio, test_ratio=CFG.test_ratio, seed=CFG.random_seed)
    print(f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # Face detection (MTCNN or OpenCV fallback)
    if FACENET_AVAILABLE:
        mtcnn = MTCNN(keep_all=False, device=CFG.mtcnn_device, post_process=True)
        print("Using MTCNN for face detection")
    else:
        mtcnn = None
        print("Using OpenCV Haar cascades for face detection (MTCNN not available)")

    # Datasets
    train_items = list(zip(train_df["video_path"].tolist(), train_df["label"].astype(int).tolist()))
    val_items = list(zip(val_df["video_path"].tolist(), val_df["label"].astype(int).tolist()))
    test_items = list(zip(test_df["video_path"].tolist(), test_df["label"].astype(int).tolist()))

    train_ds = VideoFaceDataset(train_items, CFG, mtcnn, augment=True)
    val_ds = VideoFaceDataset(val_items, CFG, mtcnn, augment=False)
    test_ds = VideoFaceDataset(test_items, CFG, mtcnn, augment=False)

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=CFG.pin_memory,
        collate_fn=collate_skip_none,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=CFG.pin_memory,
        collate_fn=collate_skip_none,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=CFG.pin_memory,
        collate_fn=collate_skip_none,
        drop_last=False,
    )

    # Model
    model = EfficientNetV2Binary(CFG.backbone_name, dropout=CFG.dropout)
    model.to(device)

    # Loss
    if CFG.pos_weight is not None:
        pos_weight = torch.tensor([CFG.pos_weight], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG.reduce_on_plateau_factor, patience=CFG.reduce_on_plateau_patience, min_lr=CFG.reduce_on_plateau_min_lr)

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if (device.type == 'cuda' and CFG.mixed_precision) else None

    # Training loop
    best_f1 = -1.0
    best_path = os.path.join(CFG.output_dir, CFG.best_model_name)

    for epoch in range(1, CFG.epochs + 1):
        print(f"\nEpoch {epoch}/{CFG.epochs}")
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, CFG.log_interval, grad_accum_steps=CFG.grad_accum_steps)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])  # ReduceLROnPlateau reacts to validation loss

        msg = (
            f"Train - loss: {train_metrics['loss']:.4f}, acc: {train_metrics['accuracy']:.4f}, "
            f"prec: {train_metrics['precision']:.4f}, rec: {train_metrics['recall']:.4f}, f1: {train_metrics['f1']:.4f}\n"
            f"Val   - loss: {val_metrics['loss']:.4f}, acc: {val_metrics['accuracy']:.4f}, "
            f"prec: {val_metrics['precision']:.4f}, rec: {val_metrics['recall']:.4f}, f1: {val_metrics['f1']:.4f}"
        )
        print(msg)

        # Checkpointing on best F1
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'config': asdict(CFG),
            }, best_path)
            print(f"Saved improved model with F1={best_f1:.4f} -> {best_path}")

    # Load best model for final evaluation
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded best model from {best_path} (F1={ckpt.get('best_f1', float('nan')):.4f})")
    else:
        print("Warning: best model checkpoint not found; evaluating current weights.")

    # Final evaluation on test set
    test_metrics = evaluate(model, test_loader, criterion, device)
    print("\nFinal Test Performance:")
    print(
        f"Test  - loss: {test_metrics['loss']:.4f}, acc: {test_metrics['accuracy']:.4f}, "
        f"prec: {test_metrics['precision']:.4f}, rec: {test_metrics['recall']:.4f}, f1: {test_metrics['f1']:.4f}"
    )


if __name__ == "__main__":
    main()


