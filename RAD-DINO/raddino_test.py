"""Multi-label evaluation of RAD-DINO checkpoints on chest X-rays stored as NPZ files.

The script supports two common RAD-DINO training implementations:

1. Hugging Face Transformers
   AutoModel.from_pretrained("microsoft/rad-dino") + linear classifier.

2. Original DINOv2/PyTorch Hub implementation
   dinov2_vitb14 + compatible RAD-DINO weights + linear classifier.

The backend is detected automatically from state_dict keys, but can be
overridden using --backend transformers or --backend torch_hub.

Features:
- loads checkpoints with model_state_dict, state_dict, model or raw weights;
- supports wrappers with encoder, backbone, rad_dino or vision_encoder;
- reproduces checkpoint preprocessing, with RAD-DINO fallback settings;
- runs multi-label inference and computes overall and per-class metrics;
- saves metrics, predictions, and metadata to CSV/JSON.

IMPORTANT: test preprocessing must match training preprocessing.
If the checkpoint lacks these metadata, verify --image-size,
--resize-mode, --image-mean and --image-std.
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
import math
import random
from typing import Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

try:
    from transformers import AutoConfig, AutoModel
except ImportError:
    AutoConfig = None
    AutoModel = None


# ============================================================
# PATHS — UPDATE HERE IF NOT USING COMMAND-LINE ARGUMENTS
# ============================================================

BASE_DIR = Path("/path/to/vindr-cxr")
TEST_NPZ_DIR = BASE_DIR / "test"
TEST_CSV = BASE_DIR / "annotations/classification.csv"
CHECKPOINT_PATH = Path("/path/to/checkpoint.pt")
OUTPUT_DIR = Path("./outputs/rad_dino/test")

# ============================================================
# FALLBACK CONFIGURATION
# ============================================================

DEFAULT_MODEL_SOURCE = "microsoft/rad-dino"
DEFAULT_TORCH_HUB_REPOSITORY = "facebookresearch/dinov2"
DEFAULT_TORCH_HUB_MODEL = "dinov2_vitb14"

DEFAULT_LABEL_COLUMNS = [
    "Lung tumor",
    "Pneumonia",
    "Tuberculosis",
    "Other diseases",
    "No finding",
]

# Official RAD-DINO processor values. Used only when the
# checkpoint does not record the values used during training.
DEFAULT_IMAGE_SIZE = 518
DEFAULT_IMAGE_MEAN = [0.5307, 0.5307, 0.5307]
DEFAULT_IMAGE_STD = [0.2583, 0.2583, 0.2583]
DEFAULT_RESIZE_MODE = "resize_and_pad"
DEFAULT_APPLY_MIN_MAX_NORMALIZATION = True
DEFAULT_NPZ_IMAGE_KEY: str | None = None
DEFAULT_THRESHOLD = 0.5
DEFAULT_DROPOUT = 0.0

BATCH_SIZE = 16
NUM_WORKERS = 4
SEED = 42

TORCH_HUB_TRUST_REPO = True
TORCH_HUB_FORCE_RELOAD = False


# ============================================================
# REPRODUCIBILITY
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# NPZ LOADING AND PREPROCESSING
# ============================================================

def load_npz_image_array(
    image_path: Path,
    image_key: str | None,
) -> np.ndarray:
    """Read the image array from an NPZ file."""

    preferred_keys = (
        "image",
        "arr_0",
        "pixel_array",
        "pixels",
        "data",
    )

    try:
        with np.load(image_path, allow_pickle=False) as archive:
            available_keys = list(archive.files)

            if not available_keys:
                raise ValueError(
                    f"The NPZ file contains no arrays:\n{image_path}"
                )

            if image_key is not None:
                selected_key = str(image_key)
                if selected_key not in archive:
                    raise KeyError(
                        f"The NPZ key {selected_key!r} does not exist in "
                        f"{image_path}. Available keys: {available_keys}."
                    )
            else:
                selected_key = next(
                    (key for key in preferred_keys if key in archive),
                    None,
                )

                if selected_key is None:
                    if len(available_keys) == 1:
                        selected_key = available_keys[0]
                    else:
                        raise ValueError(
                            "The NPZ file contains multiple arrays and the image "
                            "could not be identified automatically. "
                            f"Path: {image_path} | keys: {available_keys}. "
                            "Use --npz-image-key with the correct key."
                        )

            return np.array(archive[selected_key], copy=True)

    except (OSError, ValueError, KeyError) as error:
        if isinstance(error, (ValueError, KeyError)):
            raise

        raise OSError(
            f"Could not open the NPZ file:\n{image_path}"
        ) from error


def read_npz(
    image_path: Path,
    image_key: str | None,
    apply_min_max_normalization: bool,
) -> torch.Tensor:
    """Read an NPZ file and return a float32 tensor of shape [1, H, W]."""

    image = load_npz_image_array(
        image_path=image_path,
        image_key=image_key,
    )
    image = np.asarray(image, dtype=np.float32)

    if image.ndim == 3:
        if image.shape[0] == 1:
            image = image[0]
        elif image.shape[-1] == 1:
            image = image[..., 0]
        else:
            raise ValueError(
                "The NPZ array has three dimensions but does not correspond "
                "to a single-channel image. "
                f"Path: {image_path} | shape: {image.shape}"
            )

    if image.ndim != 2:
        raise ValueError(
            "Expected an image of shape [H,W], [1,H,W], or [H,W,1]. "
            f"Path: {image_path} | shape: {image.shape}"
        )

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    image_min = float(image.min())
    image_max = float(image.max())

    if apply_min_max_normalization:
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        else:
            image = np.zeros_like(image, dtype=np.float32)
    elif image_min < 0.0 or image_max > 1.0:
        raise ValueError(
            "Min-max normalization is disabled, but the image is not "
            "within the [0,1] range. "
            f"Path: {image_path} | min={image_min:.6f} | "
            f"max={image_max:.6f}"
        )

    image = np.ascontiguousarray(image, dtype=np.float32)
    return torch.from_numpy(image).unsqueeze(0)


class ResizeAndPad:
    """Preserve aspect ratio and pad to output_size × output_size."""

    def __init__(self, output_size: int):
        self.output_size = int(output_size)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        _, height, width = image.shape

        scale = self.output_size / max(height, width)
        new_height = max(1, round(height * scale))
        new_width = max(1, round(width * scale))

        image = F.interpolate(
            image.unsqueeze(0),
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)

        pad_height = self.output_size - new_height
        pad_width = self.output_size - new_width

        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        return F.pad(
            image,
            (pad_left, pad_right, pad_top, pad_bottom),
            value=0.0,
        )


class RepeatGrayscaleToRGB:
    """Convert [1,H,W] to [3,H,W] by repeating the channel."""

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 3 or image.shape[0] != 1:
            raise ValueError(
                "RepeatGrayscaleToRGB expected [1,H,W], but received "
                f"{tuple(image.shape)}."
            )

        return image.repeat(3, 1, 1)




def resize_shortest_edge_and_center_crop(
    image: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    """Resize the shorter edge and apply a square center crop."""

    _, height, width = image.shape
    scale = output_size / min(height, width)
    new_height = max(output_size, round(height * scale))
    new_width = max(output_size, round(width * scale))

    image = F.interpolate(
        image.unsqueeze(0),
        size=(new_height, new_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).squeeze(0)

    top = max(0, (new_height - output_size) // 2)
    left = max(0, (new_width - output_size) // 2)
    return image[:, top:top + output_size, left:left + output_size]


class ShortestEdgeCenterCrop:
    def __init__(self, output_size: int):
        self.output_size = int(output_size)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return resize_shortest_edge_and_center_crop(
            image,
            self.output_size,
        )


class ResizeSquare:
    """Resize directly to a square without preserving aspect ratio."""

    def __init__(self, output_size: int):
        self.output_size = int(output_size)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            image.unsqueeze(0),
            size=(self.output_size, self.output_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).squeeze(0)


def build_test_transform(
    image_size: int,
    image_mean: list[float],
    image_std: list[float],
    resize_mode: str,
):
    if resize_mode == "resize_and_pad":
        resize_transform = ResizeAndPad(image_size)
    elif resize_mode == "shortest_edge_center_crop":
        resize_transform = ShortestEdgeCenterCrop(image_size)
    elif resize_mode == "square":
        resize_transform = ResizeSquare(image_size)
    else:
        raise ValueError(f"Unsupported resize_mode: {resize_mode!r}")

    return transforms.Compose([
        resize_transform,
        RepeatGrayscaleToRGB(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])


# ============================================================
# DATASET
# ============================================================


class CXRTestDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: Path,
        label_columns: list[str],
        transform,
        image_key: str | None,
        apply_min_max_normalization: bool,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.label_columns = label_columns
        self.transform = transform
        self.image_key = image_key
        self.apply_min_max_normalization = apply_min_max_normalization

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        image_id = str(row["image_id"])
        image_path = self.image_dir / f"{image_id}.npz"

        if not image_path.exists():
            raise FileNotFoundError(
                f"NPZ file not found:\n{image_path}"
            )

        image = read_npz(
            image_path=image_path,
            image_key=self.image_key,
            apply_min_max_normalization=self.apply_min_max_normalization,
        )

        if self.transform is not None:
            image = self.transform(image)

        labels = torch.tensor(
            row[self.label_columns].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

        return image, labels, image_id


def load_test_csv(
    csv_path: Path,
    label_columns: list[str],
) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"The test CSV does not exist:\n{csv_path}")

    dataframe = pd.read_csv(csv_path, sep=",")
    required_columns = ["image_id"] + label_columns

    missing_columns = [
        column for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns are missing from the test CSV: "
            f"{missing_columns}"
        )

    dataframe = dataframe[required_columns].copy()
    dataframe["image_id"] = (
        dataframe["image_id"]
        .astype(str)
        .map(lambda value: Path(value).stem)
    )
    dataframe = dataframe.reset_index(drop=True)

    if dataframe.empty:
        raise ValueError("The test CSV is empty.")

    if dataframe["image_id"].duplicated().any():
        duplicated = (
            dataframe.loc[
                dataframe["image_id"].duplicated(),
                "image_id",
            ]
            .head(5)
            .tolist()
        )
        raise ValueError(
            "Duplicate image_id values were found in the test CSV. "
            f"Examples: {duplicated}"
        )

    for column in label_columns:
        invalid_values = ~dataframe[column].isin([0, 1, 0.0, 1.0])
        if invalid_values.any():
            examples = dataframe.loc[invalid_values, column].head(5).tolist()
            raise ValueError(
                f"Column {column!r} contains values other than 0/1. "
                f"Examples: {examples}"
            )

    return dataframe


# ============================================================
# CHECKPOINT AND MODEL
# ============================================================


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"The checkpoint does not exist:\n{checkpoint_path}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "The checkpoint must be a dictionary or a state_dict."
        )

    return checkpoint


def extract_raw_state_dict(
    checkpoint: dict[str, Any],
) -> dict[str, torch.Tensor]:
    for key in (
        "model_state_dict",
        "state_dict",
        "model",
        "network_state_dict",
    ):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict) and candidate:
            state_dict = candidate
            break
    else:
        if checkpoint and all(
            isinstance(value, torch.Tensor)
            for value in checkpoint.values()
        ):
            state_dict = checkpoint
        else:
            raise KeyError(
                "No state_dict was found. Expected keys: "
                "'model_state_dict', 'state_dict', 'model', or "
                "'network_state_dict'."
            )

    return {
        str(key): value
        for key, value in state_dict.items()
        if isinstance(value, torch.Tensor)
    }


def normalize_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Normalize common key names to encoder.* and classifier.1.*."""

    cleaned: dict[str, torch.Tensor] = {}

    encoder_aliases = (
        "backbone.",
        "rad_dino.",
        "raddino.",
        "vision_encoder.",
        "image_encoder.",
    )
    classifier_aliases = (
        "head.",
        "fc.",
        "linear.",
        "classification_head.",
    )

    for original_key, value in state_dict.items():
        key = str(original_key)

        while key.startswith("module."):
            key = key[len("module."):]

        if key.startswith("model."):
            inner = key[len("model."):]
            known_wrapper_prefixes = (
                "encoder.",
                "classifier.",
                *encoder_aliases,
                *classifier_aliases,
            )
            if inner.startswith(known_wrapper_prefixes):
                key = inner
            else:
                # AutoModel saved directly as self.model.
                key = "encoder." + inner

        for prefix in encoder_aliases:
            if key.startswith(prefix):
                key = "encoder." + key[len(prefix):]
                break

        # State dict of an AutoModel saved without a wrapper.
        if key.startswith((
            "embeddings.",
            "layernorm.",
            "pooler.",
        )):
            key = "encoder." + key
        elif key.startswith("encoder.layer."):
            # Internal HF Dinov2Model: encoder.layer.*
            key = "encoder." + key

        for prefix in classifier_aliases:
            if key.startswith(prefix):
                key = "classifier." + key[len(prefix):]
                break

        if key in ("classifier.weight", "classifier.bias"):
            suffix = key.split(".")[-1]
            key = f"classifier.1.{suffix}"
        elif key.startswith("classifier.0."):
            suffix = key[len("classifier.0."):]
            if suffix in ("weight", "bias"):
                key = f"classifier.1.{suffix}"
        elif key.startswith("classifier.1."):
            pass

        cleaned[key] = value

    return cleaned


def detect_backend(
    state_dict: dict[str, torch.Tensor],
    checkpoint: dict[str, Any],
    requested_backend: str,
) -> str:
    if requested_backend != "auto":
        return requested_backend

    metadata_backend = str(
        checkpoint.get(
            "backend",
            checkpoint.get("model_backend", ""),
        )
    ).lower()

    if "transform" in metadata_backend or "hugging" in metadata_backend:
        return "transformers"
    if "hub" in metadata_backend or "dinov2_original" in metadata_backend:
        return "torch_hub"

    keys = tuple(state_dict.keys())

    if any(
        key.startswith("encoder.embeddings.")
        or key.startswith("encoder.encoder.layer.")
        or key.startswith("encoder.layernorm.")
        for key in keys
    ):
        return "transformers"

    if any(
        key.startswith("encoder.patch_embed.")
        or key.startswith("encoder.blocks.")
        or key in ("encoder.cls_token", "encoder.pos_embed")
        for key in keys
    ):
        return "torch_hub"

    # The official RAD-DINO package is distributed through Transformers.
    return "transformers"


def infer_classifier_dimensions(
    state_dict: dict[str, torch.Tensor],
    fallback_num_classes: int,
) -> tuple[int | None, int]:
    weight = state_dict.get("classifier.1.weight")

    if weight is None:
        return None, fallback_num_classes

    if weight.ndim != 2:
        raise ValueError(
            "Classifier weights should have shape [classes, features], "
            f"but received {tuple(weight.shape)}."
        )

    return int(weight.shape[1]), int(weight.shape[0])


def first_metadata_value(
    checkpoint: dict[str, Any],
    keys: tuple[str, ...],
    default: Any,
) -> Any:
    for key in keys:
        if key in checkpoint and checkpoint[key] is not None:
            return checkpoint[key]
    return default


def resolve_metadata(
    checkpoint: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, Any]:
    label_columns = checkpoint.get("label_columns", DEFAULT_LABEL_COLUMNS)
    if not isinstance(label_columns, (list, tuple)) or not label_columns:
        raise ValueError("'label_columns' in the checkpoint is invalid.")
    label_columns = [str(label) for label in label_columns]

    inferred_hidden_size, inferred_num_classes = infer_classifier_dimensions(
        state_dict,
        fallback_num_classes=len(label_columns),
    )

    if inferred_num_classes != len(label_columns):
        raise ValueError(
            "The number of classifier outputs does not match "
            f"label_columns: {inferred_num_classes} vs {len(label_columns)}."
        )

    backend = detect_backend(
        state_dict=state_dict,
        checkpoint=checkpoint,
        requested_backend=args.backend,
    )

    image_size = int(
        args.image_size
        if args.image_size is not None
        else first_metadata_value(
            checkpoint,
            ("image_size", "input_size", "crop_size"),
            DEFAULT_IMAGE_SIZE,
        )
    )
    if image_size <= 0:
        raise ValueError("image_size must be positive.")

    checkpoint_resize_mode = str(
        first_metadata_value(
            checkpoint,
            ("resize_mode", "preprocessing_resize_mode"),
            DEFAULT_RESIZE_MODE,
        )
    )
    resize_mode = (
        args.resize_mode
        if args.resize_mode != "auto"
        else checkpoint_resize_mode
    )
    resize_aliases = {
        "pad": "resize_and_pad",
        "resize_pad": "resize_and_pad",
        "resize-and-pad": "resize_and_pad",
        "center_crop": "shortest_edge_center_crop",
        "official": "shortest_edge_center_crop",
        "direct": "square",
    }
    resize_mode = resize_aliases.get(resize_mode, resize_mode)
    if resize_mode not in {
        "resize_and_pad",
        "shortest_edge_center_crop",
        "square",
    }:
        raise ValueError(
            f"Invalid resize_mode: {resize_mode!r}."
        )

    image_mean = (
        list(args.image_mean)
        if args.image_mean is not None
        else list(first_metadata_value(
            checkpoint,
            ("image_mean", "rad_dino_mean", "normalization_mean", "imagenet_mean"),
            DEFAULT_IMAGE_MEAN,
        ))
    )
    image_std = (
        list(args.image_std)
        if args.image_std is not None
        else list(first_metadata_value(
            checkpoint,
            ("image_std", "rad_dino_std", "normalization_std", "imagenet_std"),
            DEFAULT_IMAGE_STD,
        ))
    )
    image_mean = [float(value) for value in image_mean]
    image_std = [float(value) for value in image_std]

    if len(image_mean) != 3 or len(image_std) != 3:
        raise ValueError("image_mean and image_std must contain three values.")
    if any(value <= 0 for value in image_std):
        raise ValueError("All image_std values must be positive.")

    checkpoint_min_max = bool(first_metadata_value(
        checkpoint,
        ("apply_min_max_normalization", "min_max_normalization"),
        DEFAULT_APPLY_MIN_MAX_NORMALIZATION,
    ))
    apply_min_max_normalization = (
        args.apply_min_max_normalization
        if args.apply_min_max_normalization is not None
        else checkpoint_min_max
    )

    npz_image_key = (
        args.npz_image_key
        if args.npz_image_key is not None
        else checkpoint.get("npz_image_key", DEFAULT_NPZ_IMAGE_KEY)
    )

    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(first_metadata_value(
            checkpoint,
            ("threshold", "classification_threshold"),
            DEFAULT_THRESHOLD,
        ))
    )
    if not 0.0 < threshold < 1.0:
        raise ValueError("The threshold must be between 0 and 1.")

    dropout = float(first_metadata_value(
        checkpoint,
        ("dropout", "classifier_dropout"),
        DEFAULT_DROPOUT,
    ))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be within [0,1).")

    model_source = str(
        args.model_source
        if args.model_source is not None
        else first_metadata_value(
            checkpoint,
            (
                "pretrained_model_name_or_path",
                "model_id",
                "encoder_name",
                "backbone_name",
                "model_name",
                "model_source",
            ),
            DEFAULT_MODEL_SOURCE,
        )
    )
    if model_source.lower() in {
        "rad-dino",
        "raddino",
        "microsoft_rad_dino",
        "dinov2_vitb14",
        "rad_dino_vitb14",
        "huggingface",
        "hf",
        "transformers",
        "github_torch_hub",
    }:
        model_source = DEFAULT_MODEL_SOURCE

    torch_hub_repository = str(
        args.dinov2_repo
        if args.dinov2_repo is not None
        else first_metadata_value(
            checkpoint,
            ("dinov2_hub_repository", "torch_hub_repository"),
            DEFAULT_TORCH_HUB_REPOSITORY,
        )
    )
    torch_hub_model = str(first_metadata_value(
        checkpoint,
        ("torch_hub_model", "dinov2_model_name"),
        DEFAULT_TORCH_HUB_MODEL,
    ))

    return {
        "backend": backend,
        "model_source": model_source,
        "torch_hub_repository": torch_hub_repository,
        "torch_hub_model": torch_hub_model,
        "hub_source": args.hub_source,
        "label_columns": label_columns,
        "num_classes": len(label_columns),
        "hidden_size_from_checkpoint": inferred_hidden_size,
        "image_size": image_size,
        "resize_mode": resize_mode,
        "image_mean": image_mean,
        "image_std": image_std,
        "in_chans": int(checkpoint.get("in_chans", 3)),
        "npz_image_key": npz_image_key,
        "apply_min_max_normalization": apply_min_max_normalization,
        "classification_threshold": threshold,
        "dropout": dropout,
        "epoch": checkpoint.get("epoch"),
        "protocol": checkpoint.get(
            "protocol",
            checkpoint.get("fine_tuning", "unknown"),
        ),
        "validation_loss": checkpoint.get("validation_loss"),
        "validation_metrics": checkpoint.get("validation_metrics"),
        "best_validation_macro_auc": checkpoint.get(
            "best_validation_macro_auc"
        ),
        "model_selection_metric": checkpoint.get("model_selection_metric"),
    }


class TransformersRADDINOClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.encoder = encoder

        hidden_size = int(getattr(encoder.config, "hidden_size"))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(pixel_values=images)

        features = getattr(outputs, "pooler_output", None)
        if features is None:
            last_hidden_state = getattr(outputs, "last_hidden_state", None)
            if last_hidden_state is None or last_hidden_state.ndim != 3:
                raise RuntimeError(
                    "The RAD-DINO encoder did not return a valid last_hidden_state."
                )
            features = last_hidden_state[:, 0]

        return self.classifier(features)


class TorchHubRADDINOClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.encoder = encoder

        if not hasattr(encoder, "embed_dim"):
            raise AttributeError(
                "The DINOv2 encoder does not have the embed_dim attribute."
            )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(int(encoder.embed_dim), num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise RuntimeError(
                "The torch_hub encoder should return [batch, features], "
                f"but returned {type(features).__name__} "
                f"with shape={getattr(features, 'shape', None)}."
            )
        return self.classifier(features)


def state_dict_contains_encoder(
    state_dict: dict[str, torch.Tensor],
) -> bool:
    return any(key.startswith("encoder.") for key in state_dict)


def build_transformers_model(
    metadata: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    local_files_only: bool,
) -> nn.Module:
    if AutoConfig is None or AutoModel is None:
        raise ImportError(
            "O backend Transformers requer a biblioteca transformers. "
            "Instale-a com: pip install transformers"
        )

    has_encoder_weights = state_dict_contains_encoder(state_dict)

    if has_encoder_weights:
        config = AutoConfig.from_pretrained(
            metadata["model_source"],
            local_files_only=local_files_only,
        )
        encoder = AutoModel.from_config(config)
    else:
        # Support checkpoints that only store the classifier from a
        # linear probing protocol.
        encoder = AutoModel.from_pretrained(
            metadata["model_source"],
            local_files_only=local_files_only,
        )

    model = TransformersRADDINOClassifier(
        encoder=encoder,
        num_classes=metadata["num_classes"],
        dropout=metadata["dropout"],
    )

    hidden_size = int(model.encoder.config.hidden_size)
    expected_hidden_size = metadata["hidden_size_from_checkpoint"]
    if (
        expected_hidden_size is not None
        and hidden_size != expected_hidden_size
    ):
        raise ValueError(
            "Classifier dimension does not match the encoder: "
            f"checkpoint={expected_hidden_size}, encoder={hidden_size}."
        )

    return model


def build_torch_hub_model(
    metadata: dict[str, Any],
    force_reload: bool,
) -> nn.Module:
    kwargs: dict[str, Any] = {
        "repo_or_dir": metadata["torch_hub_repository"],
        "model": metadata["torch_hub_model"],
        "source": metadata["hub_source"],
        "pretrained": False,
        "force_reload": force_reload,
    }

    if metadata["hub_source"] == "github":
        kwargs["trust_repo"] = TORCH_HUB_TRUST_REPO

    try:
        encoder = torch.hub.load(**kwargs)
    except TypeError:
        kwargs.pop("trust_repo", None)
        encoder = torch.hub.load(**kwargs)
    except Exception as error:
        raise RuntimeError(
            "Could not construct the DINOv2 architecture through PyTorch Hub. "
            "Use an accessible cache or --hub-source local with "
            "--dinov2-repo /caminho/para/dinov2."
        ) from error

    model = TorchHubRADDINOClassifier(
        encoder=encoder,
        num_classes=metadata["num_classes"],
        dropout=metadata["dropout"],
    )

    expected_hidden_size = metadata["hidden_size_from_checkpoint"]
    if (
        expected_hidden_size is not None
        and int(encoder.embed_dim) != expected_hidden_size
    ):
        raise ValueError(
            "Classifier dimension does not match the encoder: "
            f"checkpoint={expected_hidden_size}, encoder={encoder.embed_dim}."
        )

    return model


def build_model_from_checkpoint(
    checkpoint: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    device: torch.device,
    local_files_only: bool,
    force_reload: bool,
) -> tuple[nn.Module, dict[str, Any]]:
    if metadata["backend"] == "transformers":
        model = build_transformers_model(
            metadata=metadata,
            state_dict=state_dict,
            local_files_only=local_files_only,
        )
    elif metadata["backend"] == "torch_hub":
        if not state_dict_contains_encoder(state_dict):
            raise RuntimeError(
                "The torch_hub backend requires the checkpoint to contain "
                "RAD-DINO encoder weights, but only "
                "classifier weights were found."
            )
        model = build_torch_hub_model(
            metadata=metadata,
            force_reload=force_reload,
        )
    else:
        raise ValueError(f"Invalid backend: {metadata['backend']}")

    incompatible = model.load_state_dict(state_dict, strict=False)

    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)

    has_encoder_weights = state_dict_contains_encoder(state_dict)
    allowed_missing = []
    if not has_encoder_weights and metadata["backend"] == "transformers":
        allowed_missing = [
            key for key in missing_keys
            if key.startswith("encoder.")
        ]

    disallowed_missing = [
        key for key in missing_keys
        if key not in allowed_missing
    ]

    if disallowed_missing or unexpected_keys:
        examples = list(state_dict.keys())[:15]
        raise RuntimeError(
            "The checkpoint is incompatible with the expected RAD-DINO wrapper. "
            f"Detected backend: {metadata['backend']}. "
            f"Missing keys: {disallowed_missing}. "
            f"Unexpected keys: {unexpected_keys}. "
            f"Checkpoint examples: {examples}"
        )

    model = model.to(device)
    model.eval()

    load_report = {
        "checkpoint_contains_encoder": has_encoder_weights,
        "allowed_missing_pretrained_encoder_keys": len(allowed_missing),
        "unexpected_keys": unexpected_keys,
    }
    return model, load_report


# ============================================================
# INFERENCE
# ============================================================


@torch.inference_mode()
def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    all_image_ids: list[str] = []
    all_targets: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    model.eval()

    progress_bar = tqdm(
        dataloader,
        desc="Test inference",
        leave=False,
        dynamic_ncols=True,
    )

    for images, labels, image_ids in progress_bar:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probabilities = torch.sigmoid(logits)

        all_image_ids.extend(list(image_ids))
        all_targets.append(labels.numpy())
        all_probabilities.append(probabilities.cpu().numpy())

    if not all_targets:
        raise RuntimeError("The test DataLoader returned no samples.")

    targets = np.concatenate(all_targets, axis=0).astype(np.int32)
    probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    ).astype(np.float64)

    return all_image_ids, targets, probabilities


# ============================================================
# METRICS
# ============================================================


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def calculate_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    label_columns: list[str],
    threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    """
    Metrics used in the benchmark:
      - mAP: unweighted mean of AP across pathologies
      - Micro-AP
      - Macro-AUROC
      - Macro-Sensitivity
      - Macro-Specificity
      - Macro-F1
      - Micro-F1
      - AP per class
    """
    if targets.shape != probabilities.shape:
        raise ValueError(
            "Targets and probabilities have different shapes: "
            f"{targets.shape} vs {probabilities.shape}"
        )

    if targets.ndim != 2:
        raise ValueError(
            "Targets and probabilities must have shape [num_samples, num_classes]."
        )

    if targets.shape[1] != len(label_columns):
        raise ValueError(
            "The number of classes in targets does not match label_columns."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities contain NaN or infinite values.")

    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("Probabilities must be within [0, 1].")

    targets = targets.astype(np.int32)

    if not np.all((targets == 0) | (targets == 1)):
        raise ValueError("Targets must contain only binary 0/1 values.")

    positive_supports = targets.sum(axis=0)
    negative_supports = targets.shape[0] - positive_supports

    invalid_classes = [
        label_columns[i]
        for i in range(len(label_columns))
        if positive_supports[i] == 0 or negative_supports[i] == 0
    ]

    if invalid_classes:
        raise ValueError(
            "Some classes have no positive or negative examples "
            "in the test set: "
            f"{invalid_classes}. Macro metrics would not be comparable."
        )

    predictions = (probabilities >= threshold).astype(np.int32)

    per_class_rows: list[dict[str, Any]] = []
    per_class_ap = []
    per_class_auroc = []
    per_class_sensitivity = []
    per_class_specificity = []

    for class_index, class_name in enumerate(label_columns):
        class_targets = targets[:, class_index]
        class_probabilities = probabilities[:, class_index]
        class_predictions = predictions[:, class_index]

        positive_support = int(class_targets.sum())
        negative_support = int(len(class_targets) - positive_support)

        tp = int(np.sum(
            (class_targets == 1) & (class_predictions == 1)
        ))
        tn = int(np.sum(
            (class_targets == 0) & (class_predictions == 0)
        ))
        fp = int(np.sum(
            (class_targets == 0) & (class_predictions == 1)
        ))
        fn = int(np.sum(
            (class_targets == 1) & (class_predictions == 0)
        ))

        ap = float(
            average_precision_score(
                class_targets,
                class_probabilities,
            )
        )

        auroc = float(
            roc_auc_score(
                class_targets,
                class_probabilities,
            )
        )

        sensitivity = safe_divide(tp, tp + fn)
        specificity = safe_divide(tn, tn + fp)

        class_f1 = float(
            f1_score(
                class_targets,
                class_predictions,
                zero_division=0,
            )
        )

        per_class_ap.append(ap)
        per_class_auroc.append(auroc)
        per_class_sensitivity.append(sensitivity)
        per_class_specificity.append(specificity)

        per_class_rows.append({
            "class": class_name,
            "threshold": float(threshold),
            "support_total": int(len(class_targets)),
            "support_positive": positive_support,
            "support_negative": negative_support,
            "prevalence": float(positive_support / len(class_targets)),
            "AP": ap,
            "AUROC": auroc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "F1": class_f1,
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
        })

    per_class_df = pd.DataFrame(per_class_rows)

    # mAP = unweighted arithmetic mean of AP across pathologies.
    mean_average_precision = float(
        np.mean(np.asarray(per_class_ap, dtype=np.float64))
    )

    # Micro-AP jointly considers all image-label pairs.
    micro_ap = float(
        average_precision_score(
            targets,
            probabilities,
            average="micro",
        )
    )

    # Macro-AUROC = unweighted mean of AUROC across pathologies.
    macro_auroc = float(
        np.mean(np.asarray(per_class_auroc, dtype=np.float64))
    )

    # Sensitivity and specificity are reported as macro averages.
    macro_sensitivity = float(
        np.mean(np.asarray(per_class_sensitivity, dtype=np.float64))
    )

    macro_specificity = float(
        np.mean(np.asarray(per_class_specificity, dtype=np.float64))
    )

    macro_f1 = float(
        f1_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    micro_f1 = float(
        f1_score(
            targets,
            predictions,
            average="micro",
            zero_division=0,
        )
    )

    summary: dict[str, Any] = {
        "number_of_images": int(targets.shape[0]),
        "number_of_classes": int(targets.shape[1]),
        "classification_threshold": float(threshold),
        "mAP": mean_average_precision,
        "micro_AP": micro_ap,
        "macro_AUROC": macro_auroc,
        "macro_sensitivity": macro_sensitivity,
        "macro_specificity": macro_specificity,
        "macro_F1": macro_f1,
        "micro_F1": micro_f1,
    }

    return summary, per_class_df, predictions


# ============================================================
# SAVE AND DISPLAY RESULTS
# ============================================================


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, (np.floating, float)):
        numeric_value = float(value)
        if math.isnan(numeric_value) or math.isinf(numeric_value):
            return None
        return numeric_value

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return json_safe(value.item())
        return json_safe(value.detach().cpu().tolist())

    return value


def save_predictions(
    output_path: Path,
    image_ids: list[str],
    targets: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    label_columns: list[str],
) -> None:
    data: dict[str, Any] = {"image_id": image_ids}

    for class_index, class_name in enumerate(label_columns):
        data[f"target__{class_name}"] = targets[:, class_index]
        data[f"probability__{class_name}"] = probabilities[:, class_index]
        data[f"prediction__{class_name}"] = predictions[:, class_index]

    pd.DataFrame(data).to_csv(output_path, index=False)


def format_metric(value: Any) -> str:
    if value is None:
        return "N/A"

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if not math.isfinite(numeric_value):
        return "N/A"

    return f"{numeric_value:.6f}"


def print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("OVERALL TEST SET RESULTS")
    print("=" * 70)

    preferred_order = [
        "number_of_images",
        "number_of_classes",
        "classification_threshold",
        "mAP",
        "micro_AP",
        "macro_AUROC",
        "macro_sensitivity",
        "macro_specificity",
        "macro_F1",
        "micro_F1",
    ]

    for key in preferred_order:
        value = summary[key]
        if isinstance(value, (int, np.integer)):
            formatted = str(value)
        else:
            formatted = f"{float(value):.6f}"
        print(f"{key}: {formatted}")


# ============================================================
# ARGUMENTS
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a multi-label RAD-DINO checkpoint on NPZ chest X-rays. "
            "Without arguments, use the paths defined at the top of the script."
        )
    )

    parser.add_argument("--test-npz-dir", type=Path, default=TEST_NPZ_DIR)
    parser.add_argument("--test-csv", type=Path, default=TEST_CSV)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            "Fixed threshold for converting probabilities into 0/1. "
            "The default is 0.5 for all benchmark models."
        ),
    )
    parser.add_argument("--npz-image-key", type=str, default=None)

    parser.add_argument(
        "--backend",
        choices=("auto", "transformers", "torch_hub"),
        default="auto",
        help="Implementation used during training.",
    )
    parser.add_argument(
        "--model-source",
        type=str,
        default=None,
        help=(
            "Hugging Face ID or local directory with config.json. "
            "Fallback: microsoft/rad-dino."
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Disable downloads from the Hugging Face Hub.",
    )

    parser.add_argument(
        "--dinov2-repo",
        type=str,
        default=None,
        help="GitHub repository or local facebookresearch/dinov2 directory.",
    )
    parser.add_argument(
        "--hub-source",
        choices=("github", "local"),
        default="github",
    )
    parser.add_argument("--force-reload", action="store_true")

    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument(
        "--resize-mode",
        choices=(
            "auto",
            "resize_and_pad",
            "shortest_edge_center_crop",
            "square",
        ),
        default="auto",
    )
    parser.add_argument(
        "--image-mean",
        type=float,
        nargs=3,
        default=None,
        metavar=("R", "G", "B"),
    )
    parser.add_argument(
        "--image-std",
        type=float,
        nargs=3,
        default=None,
        metavar=("R", "G", "B"),
    )

    normalization_group = parser.add_mutually_exclusive_group()
    normalization_group.add_argument(
        "--min-max-normalization",
        dest="apply_min_max_normalization",
        action="store_true",
    )
    normalization_group.add_argument(
        "--no-min-max-normalization",
        dest="apply_min_max_normalization",
        action="store_false",
    )
    parser.set_defaults(apply_min_max_normalization=None)

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1.")
    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0.")
    if args.threshold is not None and not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be between 0 and 1.")
    if args.image_size is not None and args.image_size < 1:
        parser.error("--image-size must be >= 1.")
    if args.image_std is not None and any(v <= 0 for v in args.image_std):
        parser.error("--image-std must contain positive values.")
    if args.hub_source == "local" and args.dinov2_repo is None:
        parser.error(
            "When using --hub-source local, also specify --dinov2-repo."
        )

    return args


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    args = parse_args()
    set_seed(SEED)

    if not args.test_npz_dir.exists():
        raise FileNotFoundError(
            f"The test NPZ directory does not exist:\n{args.test_npz_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("RAD-DINO — TEST SET EVALUATION")
    print("=" * 70)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, device)
    raw_state_dict = extract_raw_state_dict(checkpoint)
    state_dict = normalize_state_dict_keys(raw_state_dict)
    metadata = resolve_metadata(checkpoint, state_dict, args)

    if metadata["in_chans"] != 3:
        raise ValueError(
            "This script converts grayscale to RGB, but the checkpoint "
            f"specifies in_chans={metadata['in_chans']}."
        )

    print(f"Backend: {metadata['backend']}")
    if metadata["backend"] == "transformers":
        print(f"Model source: {metadata['model_source']}")
    else:
        print(
            "PyTorch Hub: "
            f"source={metadata['hub_source']} | "
            f"repo={metadata['torch_hub_repository']} | "
            f"model={metadata['torch_hub_model']}"
        )
    print(f"Training protocol: {metadata['protocol']}")
    print(f"Checkpoint epoch: {metadata['epoch']}")
    print(f"Image size: {metadata['image_size']}")
    print(f"Resize mode: {metadata['resize_mode']}")
    print(f"Image mean: {metadata['image_mean']}")
    print(f"Image std: {metadata['image_std']}")
    print(f"Classes: {metadata['label_columns']}")
    print(
        "Min-max normalization: "
        f"{metadata['apply_min_max_normalization']}"
    )
    print(f"NPZ key: {metadata['npz_image_key']}")
    print(f"Threshold: {metadata['classification_threshold']}")

    model, load_report = build_model_from_checkpoint(
        checkpoint=checkpoint,
        state_dict=state_dict,
        metadata=metadata,
        device=device,
        local_files_only=args.local_files_only,
        force_reload=(
            TORCH_HUB_FORCE_RELOAD or args.force_reload
        ),
    )
    metadata["checkpoint_load_report"] = load_report

    test_df = load_test_csv(
        csv_path=args.test_csv,
        label_columns=metadata["label_columns"],
    )

    print(f"\nTest CSV: {args.test_csv}")
    print(f"Test NPZ directory: {args.test_npz_dir}")
    print(f"Number of images: {len(test_df)}")

    print("\nPositive examples per test class:")
    for label in metadata["label_columns"]:
        print(f"  {label}: {int(test_df[label].sum())}")

    test_dataset = CXRTestDataset(
        dataframe=test_df,
        image_dir=args.test_npz_dir,
        label_columns=metadata["label_columns"],
        transform=build_test_transform(
            image_size=metadata["image_size"],
            image_mean=metadata["image_mean"],
            image_std=metadata["image_std"],
            resize_mode=metadata["resize_mode"],
        ),
        image_key=metadata["npz_image_key"],
        apply_min_max_normalization=(
            metadata["apply_min_max_normalization"]
        ),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    image_ids, targets, probabilities = run_inference(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    summary, per_class_df, predictions = calculate_metrics(
        targets=targets,
        probabilities=probabilities,
        label_columns=metadata["label_columns"],
        threshold=metadata["classification_threshold"],
    )

    summary_with_metadata = {
        "evaluation": summary,
        "checkpoint_metadata": metadata,
        "paths": {
            "test_npz_dir": str(args.test_npz_dir),
            "test_csv": str(args.test_csv),
            "checkpoint": str(args.checkpoint),
            "output_dir": str(args.output_dir),
        },
    }

    summary_json_path = args.output_dir / "test_summary_metrics.json"
    summary_csv_path = args.output_dir / "test_summary_metrics.csv"
    per_class_path = args.output_dir / "test_per_class_metrics.csv"
    predictions_path = args.output_dir / "test_predictions.csv"

    with summary_json_path.open("w", encoding="utf-8") as file:
        json.dump(
            json_safe(summary_with_metadata),
            file,
            ensure_ascii=False,
            indent=2,
        )

    pd.DataFrame([summary]).to_csv(summary_csv_path, index=False)
    per_class_df.to_csv(per_class_path, index=False)

    save_predictions(
        output_path=predictions_path,
        image_ids=image_ids,
        targets=targets,
        probabilities=probabilities,
        predictions=predictions,
        label_columns=metadata["label_columns"],
    )

    print_summary(summary)

    print("\n" + "=" * 70)
    print("METRICS PER CLASS")
    print("=" * 70)
    columns_to_print = [
        "class",
        "support_positive",
        "support_negative",
        "AP",
        "AUROC",
        "sensitivity",
        "specificity",
        "F1",
    ]

    print(
        per_class_df[
            columns_to_print
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nResults saved to:")
    print(f"  JSON summary: {summary_json_path}")
    print(f"  CSV summary: {summary_csv_path}")
    print(f"  Per-class metrics: {per_class_path}")
    print(f"  Per-image predictions: {predictions_path}")


if __name__ == "__main__":
    main()
