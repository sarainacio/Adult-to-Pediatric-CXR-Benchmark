from pathlib import Path
import argparse
import json
import math
import random

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

from medclip import MedCLIPVisionModelViT

# ============================================================
# PATH CONFIGURATION — UPDATE BEFORE RUNNING
# ============================================================

TEST_NPZ_DIR = Path("/path/to/vindr-cxr/test")
TEST_CSV = Path("/path/to/vindr-cxr/annotations/classification.csv")
CHECKPOINT_PATH = Path("/path/to/checkpoint.pt")
OUTPUT_DIR = Path("./outputs/medclip/test")

# ============================================================
# FALLBACK CONFIGURATION
# ============================================================
# These values are used only when missing from the checkpoint.

DEFAULT_MODEL_NAME = "medclip-vit-swin-tiny"

DEFAULT_LABEL_COLUMNS = [
    "Lung tumor",
    "Pneumonia",
    "Tuberculosis",
    "Other diseases",
    "No finding",
]

DEFAULT_IMAGE_SIZE = 224
DEFAULT_APPLY_MIN_MAX_NORMALIZATION = True

DEFAULT_MEDCLIP_MEAN = [0.5862785803043838] * 3
DEFAULT_MEDCLIP_STD = [0.27950088968644304] * 3

DEFAULT_CLASSIFICATION_THRESHOLD = 0.5

# None enables automatic detection of the image key in the NPZ archive.
DEFAULT_NPZ_IMAGE_KEY: str | None = None

BATCH_SIZE = 32
NUM_WORKERS = 4
SEED = 42


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
    """
    Extracts the image array from an NPZ file.

    When image_key is None, common keys are checked first. If none is found,
    an archive containing a single array is accepted automatically.
    """

    preferred_keys = (
        "image",
        "arr_0",
        "pixel_array",
        "pixels",
        "data",
    )

    requested_key = None
    if image_key is not None:
        requested_key = str(image_key).strip()
        if not requested_key:
            raise ValueError(
                "image_key cannot be an empty string."
            )

    with np.load(
            image_path,
            allow_pickle=False,
    ) as archive:
        available_keys = list(archive.files)

        if not available_keys:
            raise ValueError(
                f"The NPZ file contains no arrays:\n{image_path}"
            )

        if requested_key is not None:
            if requested_key not in archive:
                raise KeyError(
                    f"The NPZ key {requested_key!r} does not exist in "
                    f"{image_path}. Available keys: {available_keys}"
                )
            selected_key = requested_key
        else:
            selected_key = next(
                (key for key in preferred_keys if key in archive),
                None,
            )

            if selected_key is None and len(available_keys) == 1:
                selected_key = available_keys[0]

            if selected_key is None:
                raise ValueError(
                    "The NPZ file contains multiple arrays and it was not possible to "
                    "automatically identify the image array. "
                    f"Path: {image_path} | keys: {available_keys}. "
                    "Set --npz-image-key or save npz_image_key in the "
                    "checkpoint."
                )

        return np.array(
            archive[selected_key],
            copy=True,
        )


def load_npz_grayscale(
        image_path: Path,
        apply_min_max_normalization: bool,
        image_key: str | None,
) -> torch.Tensor:
    """
    Reads an NPZ image and returns float32 [1, H, W] in [0, 1].

    Accepted shapes:
        [H, W]
        [1, H, W]
        [H, W, 1]
    """

    image = load_npz_image_array(
        image_path=image_path,
        image_key=image_key,
    )

    image = np.asarray(
        image,
        dtype=np.float32,
    )

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
            "Expected an image with shape [H,W], [1,H,W] or [H,W,1]. "
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
            image = (
                    (image - image_min)
                    / (image_max - image_min)
            )
        else:
            image = np.zeros_like(
                image,
                dtype=np.float32,
            )

    elif image_min < 0.0 or image_max > 1.0:
        raise ValueError(
            "The checkpoint indicates that min-max normalization should not be applied, "
            "but the image is not within the [0,1] range. "
            f"Path: {image_path} | min={image_min:.6f} | "
            f"max={image_max:.6f}"
        )

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    image = np.ascontiguousarray(
        image,
        dtype=np.float32,
    )

    return torch.from_numpy(image).unsqueeze(0)

class ResizeAndPad:
    """Preserves the aspect ratio and pads to output_size × output_size."""

    def __init__(
            self,
            output_size: int,
    ):
        self.output_size = int(output_size)

    def __call__(
            self,
            image: torch.Tensor,
    ) -> torch.Tensor:
        if image.ndim != 3:
            raise ValueError(
                "ResizeAndPad expected a [C,H,W] tensor, "
                f"but received {tuple(image.shape)}."
            )

        _, height, width = image.shape

        scale = self.output_size / max(
            height,
            width,
        )

        new_height = max(
            1,
            round(height * scale),
        )

        new_width = max(
            1,
            round(width * scale),
        )

        image = F.interpolate(
            image.unsqueeze(0),
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).squeeze(0)

        pad_height = (
                self.output_size - new_height
        )

        pad_width = (
                self.output_size - new_width
        )

        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        return F.pad(
            image,
            (
                pad_left,
                pad_right,
                pad_top,
                pad_bottom,
            ),
            value=0.0,
        )


class RepeatGrayscaleToRGB:
    """Converts [1,H,W] to [3,H,W] by repeating the channel."""

    def __call__(
            self,
            image: torch.Tensor,
    ) -> torch.Tensor:
        if (
                image.ndim != 3
                or image.shape[0] != 1
        ):
            raise ValueError(
                "RepeatGrayscaleToRGB expected [1,H,W], "
                f"but received {tuple(image.shape)}."
            )

        return image.repeat(
            3,
            1,
            1,
        )


def build_test_transform(
        image_size: int,
        medclip_mean: list[float],
        medclip_std: list[float],
):
    if (
            len(medclip_mean) != 3
            or len(medclip_std) != 3
    ):
        raise ValueError(
            "medclip_mean and medclip_std must contain three values."
        )

    if any(
            float(value) <= 0.0
            for value in medclip_std
    ):
        raise ValueError(
            "All medclip_std values must be positive."
        )

    return transforms.Compose([
        ResizeAndPad(
            output_size=image_size,
        ),
        RepeatGrayscaleToRGB(),
        transforms.Normalize(
            mean=medclip_mean,
            std=medclip_std,
        ),
    ])


# ============================================================
# CSV AND DATASET
# ============================================================


def load_test_csv(
        csv_path: Path,
        label_columns: list[str],
) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"The test CSV does not exist:\n{csv_path}"
        )

    dataframe = pd.read_csv(
        csv_path,
        sep=",",
    )

    required_columns = [
        "image_id",
        *label_columns,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns are missing from the test CSV: "
            f"{missing_columns}"
        )

    dataframe = dataframe[
        required_columns
    ].copy()

    dataframe["image_id"] = (
        dataframe["image_id"]
        .astype(str)
        .map(lambda value: Path(value).stem)
    )

    dataframe = dataframe.reset_index(
        drop=True
    )

    if dataframe.empty:
        raise ValueError(
            "The test CSV is empty."
        )

    duplicated_mask = dataframe[
        "image_id"
    ].duplicated()

    if duplicated_mask.any():
        duplicated_examples = (
            dataframe.loc[
                duplicated_mask,
                "image_id",
            ]
            .head(5)
            .tolist()
        )

        raise ValueError(
            "Duplicate image_id values were found in the test CSV. "
            f"Examples: {duplicated_examples}"
        )

    for column in label_columns:
        numeric_values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        if numeric_values.isna().any():
            examples = (
                dataframe.loc[
                    numeric_values.isna(),
                    column,
                ]
                .head(5)
                .tolist()
            )

            raise ValueError(
                f"Column '{column}' contains non-numeric values. "
                f"Examples: {examples}"
            )

        invalid_values = ~numeric_values.isin(
            [0, 1]
        )

        if invalid_values.any():
            examples = (
                numeric_values.loc[
                    invalid_values
                ]
                .head(5)
                .tolist()
            )

            raise ValueError(
                f"Column '{column}' contains values other than 0/1. "
                f"Examples: {examples}"
            )

        dataframe[column] = (
            numeric_values.astype(
                np.float32
            )
        )

    return dataframe


class MedCLIPTestDataset(Dataset):
    def __init__(
            self,
            dataframe: pd.DataFrame,
            image_dir: Path,
            label_columns: list[str],
            transform,
            apply_min_max_normalization: bool,
            image_key: str | None,
    ):
        self.dataframe = dataframe.reset_index(
            drop=True
        )
        self.image_dir = Path(image_dir)
        self.label_columns = list(
            label_columns
        )
        self.transform = transform
        self.apply_min_max_normalization = bool(
            apply_min_max_normalization
        )
        self.image_key = image_key

    def __len__(self) -> int:
        return len(
            self.dataframe
        )

    def __getitem__(
            self,
            index: int,
    ):
        row = self.dataframe.iloc[
            index
        ]

        image_id = str(
            row["image_id"]
        )

        image_path = (
                self.image_dir
                / f"{image_id}.npz"
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"NPZ file not found:\n{image_path}"
            )

        image = load_npz_grayscale(
            image_path=image_path,
            apply_min_max_normalization=(
                self.apply_min_max_normalization
            ),
            image_key=self.image_key,
        )

        if self.transform is not None:
            image = self.transform(
                image
            )

        labels = torch.tensor(
            row[
                self.label_columns
            ].to_numpy(
                dtype=np.float32
            ),
            dtype=torch.float32,
        )

        return (
            image,
            labels,
            image_id,
        )


# ============================================================
# CHECKPOINT AND MODEL
# ============================================================


def load_checkpoint(
        checkpoint_path: Path,
        device: torch.device,
) -> dict:
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

    if not isinstance(
            checkpoint,
            dict,
    ):
        raise TypeError(
            "The checkpoint must be a dictionary containing "
            "'model_state_dict'."
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "The checkpoint does not contain the key 'model_state_dict'."
        )

    return checkpoint


class MedCLIPClassifier(nn.Module):
    """
    Multi-label classifier built on the pre-projection MedCLIP-ViT embedding.

    The architecture matches the training scripts:
        Swin-Tiny vision encoder
        dropout
        Linear classification layer
    """

    def __init__(
            self,
            encoder: nn.Module,
            num_classes: int,
            dropout: float = 0.0,
    ):
        super().__init__()

        self.encoder = encoder

        if not hasattr(
                encoder,
                "config",
        ):
            raise AttributeError(
                "The MedCLIP encoder does not have the attribute 'config'."
            )

        hidden_size = getattr(
            encoder.config,
            "hidden_size",
            None,
        )

        if hidden_size is None:
            raise AttributeError(
                "Could not determine encoder.config.hidden_size."
            )

        self.classifier = nn.Sequential(
            nn.Dropout(
                p=dropout
            ),
            nn.Linear(
                int(hidden_size),
                int(num_classes),
            ),
        )

    def forward(
            self,
            images: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(
            pixel_values=images,
        )

        if hasattr(
                outputs,
                "pooler_output",
        ):
            features = (
                outputs.pooler_output
            )

        elif (
                isinstance(
                    outputs,
                    dict,
                )
                and "pooler_output" in outputs
        ):
            features = (
                outputs[
                    "pooler_output"
                ]
            )

        else:
            raise TypeError(
                "The MedCLIP/Swin backbone did not return pooler_output."
            )

        if features.ndim != 2:
            raise ValueError(
                "Expected an embedding of shape [batch, dimension], "
                f"but received shape={tuple(features.shape)}."
            )

        return self.classifier(
            features
        )


def create_medclip_encoder() -> nn.Module:
    """
    Creates only the visual encoder used by MedCLIP-ViT.

    Loading the text model is unnecessary. The training checkpoint
    contains the complete state of the visual encoder and classifier.

    MedCLIPVisionModelViT constructs the Swin-Tiny visual backbone. Then,
    the checkpoint model_state_dict replaces its parameters.
    """

    try:
        vision_wrapper = (
            MedCLIPVisionModelViT()
        )
    except Exception as error:
        raise RuntimeError(
            "Could not construct the MedCLIP-ViT encoder. "
            "Verify that the MedCLIP package and the "
            "transformers dependencies are installed correctly."
        ) from error

    if not hasattr(
            vision_wrapper,
            "model",
    ):
        raise AttributeError(
            "MedCLIPVisionModelViT does not contain the internal backbone 'model'."
        )

    return vision_wrapper.model


def remove_prefix_from_state_dict(
        state_dict: dict,
        prefix: str,
) -> dict:
    if (
            state_dict
            and all(
        key.startswith(prefix)
        for key in state_dict
    )
    ):
        return {
            key[len(prefix):]: value
            for key, value in state_dict.items()
        }

    return state_dict


def load_model_state_strict(
        model: nn.Module,
        state_dict: dict,
) -> None:
    """
    Loads the model state with strict=True.

    Supports checkpoints saved using DataParallel, DDP,
    torch.compile, or an additional wrapper called model.
    """

    candidate_state_dicts = [
        state_dict,
        remove_prefix_from_state_dict(
            state_dict,
            "module.",
        ),
        remove_prefix_from_state_dict(
            state_dict,
            "_orig_mod.",
        ),
        remove_prefix_from_state_dict(
            state_dict,
            "model.",
        ),
    ]

    errors = []

    for candidate in candidate_state_dicts:
        try:
            model.load_state_dict(
                candidate,
                strict=True,
            )
            return
        except RuntimeError as error:
            errors.append(
                str(error)
            )

    raise RuntimeError(
        "Could not load model_state_dict with strict=True.\n"
        + "\n\n".join(errors)
    )


def build_model_from_checkpoint(
        checkpoint: dict,
        device: torch.device,
):
    model_name = str(
        checkpoint.get(
            "model_name",
            DEFAULT_MODEL_NAME,
        )
    )

    label_columns = checkpoint.get(
        "label_columns",
        DEFAULT_LABEL_COLUMNS,
    )

    if (
            not isinstance(
                label_columns,
                (list, tuple),
            )
            or not label_columns
    ):
        raise ValueError(
            "Invalid 'label_columns' in the checkpoint."
        )

    label_columns = list(
        label_columns
    )

    num_classes = len(
        label_columns
    )

    checkpoint_num_classes = checkpoint.get(
        "num_classes"
    )

    if (
            checkpoint_num_classes is not None
            and int(checkpoint_num_classes)
            != num_classes
    ):
        raise ValueError(
            "The checkpoint contains num_classes incompatible with "
            "label_columns: "
            f"{checkpoint_num_classes} vs {num_classes}."
        )

    encoder = create_medclip_encoder()

    model = MedCLIPClassifier(
        encoder=encoder,
        num_classes=num_classes,
        dropout=0.0,
    )

    load_model_state_strict(
        model=model,
        state_dict=checkpoint[
            "model_state_dict"
        ],
    )

    model = model.to(
        device
    )

    model.eval()

    image_size = int(
        checkpoint.get(
            "image_size",
            DEFAULT_IMAGE_SIZE,
        )
    )

    apply_min_max_normalization = bool(
        checkpoint.get(
            "apply_min_max_normalization",
            DEFAULT_APPLY_MIN_MAX_NORMALIZATION,
        )
    )

    medclip_mean = list(
        checkpoint.get(
            "medclip_mean",
            DEFAULT_MEDCLIP_MEAN,
        )
    )

    medclip_std = list(
        checkpoint.get(
            "medclip_std",
            DEFAULT_MEDCLIP_STD,
        )
    )

    checkpoint_threshold = float(
        checkpoint.get(
            "threshold",
            DEFAULT_CLASSIFICATION_THRESHOLD,
        )
    )

    preprocess = build_test_transform(
        image_size=image_size,
        medclip_mean=medclip_mean,
        medclip_std=medclip_std,
    )

    metadata = {
        "model_name": model_name,
        "vision_backbone": checkpoint.get(
            "vision_backbone",
            "microsoft/swin-tiny-patch4-window7-224",
        ),
        "label_columns": label_columns,
        "num_classes": num_classes,
        "protocol": checkpoint.get(
            "protocol",
            checkpoint.get(
                "fine_tuning",
                "unknown",
            ),
        ),
        "fine_tuning": checkpoint.get(
            "fine_tuning",
            "unknown",
        ),
        "training_epoch": checkpoint.get(
            "epoch"
        ),
        "checkpoint_threshold": (
            checkpoint_threshold
        ),
        "image_size": image_size,
        "input_format": checkpoint.get(
            "input_format",
            "npz",
        ),
        "npz_image_key": checkpoint.get(
            "npz_image_key",
            DEFAULT_NPZ_IMAGE_KEY,
        ),
        "dataset_implementation": checkpoint.get(
            "dataset_implementation",
            "standalone_npz",
        ),
        "apply_min_max_normalization": (
            apply_min_max_normalization
        ),
        "medclip_mean": medclip_mean,
        "medclip_std": medclip_std,
        "validation_loss": checkpoint.get(
            "validation_loss"
        ),
        "best_validation_loss": checkpoint.get(
            "best_validation_loss"
        ),
        "validation_metrics": checkpoint.get(
            "validation_metrics"
        ),
        "trainable_layers": checkpoint.get(
            "trainable_layers"
        ),
    }

    return (
        model,
        preprocess,
        metadata,
    )


# ============================================================
# INFERENCE
# ============================================================


def run_inference(
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
):
    all_image_ids = []
    all_targets = []
    all_probabilities = []

    model.eval()

    with torch.inference_mode():
        progress_bar = tqdm(
            dataloader,
            desc="Test inference",
            leave=False,
            dynamic_ncols=True,
        )

        for (
                images,
                labels,
                image_ids,
        ) in progress_bar:
            images = images.to(
                device,
                non_blocking=True,
            )

            logits = model(
                images
            )

            probabilities = torch.sigmoid(
                logits
            )

            all_image_ids.extend(
                list(image_ids)
            )

            all_targets.append(
                labels.numpy()
            )

            all_probabilities.append(
                probabilities
                .cpu()
                .numpy()
            )

    if not all_targets:
        raise RuntimeError(
            "The test DataLoader returned no samples."
        )

    targets = np.concatenate(
        all_targets,
        axis=0,
    ).astype(np.int32)

    probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    ).astype(np.float64)

    return (
        all_image_ids,
        targets,
        probabilities,
    )


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
):
    """
    Benchmark metrics:
      - mAP: unweighted mean of per-class average precision
      - Micro-AP
      - Macro-AUROC
      - Macro-Sensitivity
      - Macro-Specificity
      - Macro-F1
      - Micro-F1
      - Per-class AP
    """
    if targets.shape != probabilities.shape:
        raise ValueError(
            "Targets and probabilities have different shapes: "
            f"{targets.shape} vs {probabilities.shape}"
        )

    if targets.ndim != 2:
        raise ValueError(
            "Targets and probabilities must have shape [n_samples, n_classes]."
        )

    if targets.shape[1] != len(label_columns):
        raise ValueError(
            "The number of target classes does not match label_columns."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities contain NaN or infinite values.")

    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("Probabilities must be within [0,1].")

    targets = targets.astype(np.int32)

    if not np.all((targets == 0) | (targets == 1)):
        raise ValueError("Targets must contain only 0/1 values.")

    positive_supports = targets.sum(axis=0)
    negative_supports = targets.shape[0] - positive_supports

    invalid_classes = [
        label_columns[i]
        for i in range(len(label_columns))
        if positive_supports[i] == 0 or negative_supports[i] == 0
    ]

    if invalid_classes:
        raise ValueError(
            "Some test classes have no positive or negative examples: "
            f"{invalid_classes}. The macro metrics would not be comparable."
        )

    predictions = (probabilities >= threshold).astype(np.int32)

    per_class_rows = []
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

    # mAP = unweighted arithmetic mean of per-class APs.
    mean_average_precision = float(
        np.mean(np.asarray(per_class_ap, dtype=np.float64))
    )

    # Micro-AP pools all image-label pairs.
    micro_ap = float(
        average_precision_score(
            targets,
            probabilities,
            average="micro",
        )
    )

    # Macro-AUROC = unweighted mean of per-class AUROCs.
    macro_auroc = float(
        np.mean(np.asarray(per_class_auroc, dtype=np.float64))
    )

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

    summary = {
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


def json_safe(
        value,
):
    if isinstance(
            value,
            dict,
    ):
        return {
            str(key): json_safe(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
            value,
            (list, tuple),
    ):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(
            value,
            np.ndarray,
    ):
        return json_safe(
            value.tolist()
        )

    if isinstance(
            value,
            torch.Tensor,
    ):
        return json_safe(
            value
            .detach()
            .cpu()
            .tolist()
        )

    if isinstance(
            value,
            np.integer,
    ):
        return int(value)

    if isinstance(
            value,
            (np.floating, float),
    ):
        numeric_value = float(
            value
        )

        if (
                math.isnan(
                    numeric_value
                )
                or math.isinf(
            numeric_value
        )
        ):
            return None

        return numeric_value

    if isinstance(
            value,
            Path,
    ):
        return str(value)

    return value


def save_predictions(
        output_path: Path,
        image_ids: list[str],
        targets: np.ndarray,
        probabilities: np.ndarray,
        predictions: np.ndarray,
        label_columns: list[str],
) -> None:
    data = {
        "image_id": image_ids,
    }

    for (
            class_index,
            class_name,
    ) in enumerate(
        label_columns
    ):
        data[
            f"target__{class_name}"
        ] = targets[
            :,
            class_index,
        ]

        data[
            f"probability__{class_name}"
        ] = probabilities[
            :,
            class_index,
        ]

        data[
            f"prediction__{class_name}"
        ] = predictions[
            :,
            class_index,
        ]

    pd.DataFrame(
        data
    ).to_csv(
        output_path,
        index=False,
    )


def print_summary(summary: dict) -> None:
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
# COMMAND-LINE ARGUMENTS
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluates a multi-label MedCLIP-ViT checkpoint on NPZ images. "
            "Supports linear probing, partial fine-tuning, "
            "and full fine-tuning. Without arguments, it uses the paths "
            "defined at the beginning of the script."
        )
    )

    parser.add_argument(
        "--test-npz-dir",
        type=Path,
        default=TEST_NPZ_DIR,
        help=(
            "Directory containing test .npz files."
        ),
    )

    parser.add_argument(
        "--npz-image-key",
        type=str,
        default=None,
        help=(
            "Internal NPZ key containing the image. If omitted, "
            "uses the checkpoint key or automatic detection."
        ),
    )

    parser.add_argument(
        "--test-csv",
        type=Path,
        default=TEST_CSV,
        help=(
            "Test CSV with image_id and labels."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help=(
            "MedCLIP .pt checkpoint to evaluate."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=(
            "Directory where results will be saved."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=(
            "Inference batch size."
        ),
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help=(
            "Number of DataLoader workers."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CLASSIFICATION_THRESHOLD,
        help=(
            "Fixed threshold for converting probabilities to binary predictions. "
            "Defaults to 0.5 for all benchmark models."
        ),
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error(
            "--batch-size must be >= 1."
        )

    if args.num_workers < 0:
        parser.error(
            "--num-workers must be >= 0."
        )

    if (
            args.threshold is not None
            and not 0.0
                    < args.threshold
                    < 1.0
    ):
        parser.error(
            "--threshold must be between 0 and 1."
        )

    if args.npz_image_key is not None:
        args.npz_image_key = args.npz_image_key.strip()
        if not args.npz_image_key:
            parser.error(
                "--npz-image-key cannot be an empty string."
            )

    return args


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    args = parse_args()
    set_seed(
        SEED
    )

    if not args.test_npz_dir.exists():
        raise FileNotFoundError(
            "The test NPZ directory does not exist:\n"
            f"{args.test_npz_dir}"
        )

    if not args.test_csv.exists():
        raise FileNotFoundError(
            "The test CSV does not exist:\n"
            f"{args.test_csv}"
        )

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            "The checkpoint does not exist:\n"
            f"{args.checkpoint}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print(
        f"Checkpoint: {args.checkpoint}"
    )

    checkpoint = load_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    (
        model,
        preprocess,
        metadata,
    ) = build_model_from_checkpoint(
        checkpoint=checkpoint,
        device=device,
    )

    label_columns = metadata[
        "label_columns"
    ]

    if args.npz_image_key is not None:
        image_key = args.npz_image_key
        image_key_source = "command line"
    else:
        image_key = metadata[
            "npz_image_key"
        ]
        image_key_source = (
            "checkpoint"
            if image_key is not None
            else "automatic detection"
        )

    threshold = float(args.threshold)
    threshold_source = "evaluation (default=0.5 or --threshold)"

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "The selected threshold must be between 0 and 1, "
            f"but received {threshold}."
        )

    print(
        f"Model: {metadata['model_name']}"
    )
    print(
        f"Backbone: {metadata['vision_backbone']}"
    )
    print(
        f"Protocol: {metadata['protocol']}"
    )
    print(
        "Checkpoint epoch: "
        f"{metadata['training_epoch']}"
    )
    print(
        f"Classes: {label_columns}"
    )
    print(
        f"Image size: {metadata['image_size']}"
    )
    print(
        "NPZ min-max normalization: "
        f"{metadata['apply_min_max_normalization']}"
    )
    print(
        "Internal NPZ key: "
        f"{image_key!r} (source: {image_key_source})"
    )
    print(
        f"Threshold: {threshold} "
        f"(source: {threshold_source})"
    )

    test_df = load_test_csv(
        csv_path=args.test_csv,
        label_columns=label_columns,
    )

    print(
        f"\nTest CSV: {args.test_csv}"
    )
    print(
        "Test NPZ directory: "
        f"{args.test_npz_dir}"
    )
    print(
        f"Number of images: {len(test_df)}"
    )

    print(
        "\nPositive examples per test class:"
    )

    for label in label_columns:
        print(
            f"  {label}: "
            f"{int(test_df[label].sum())}"
        )

    test_dataset = MedCLIPTestDataset(
        dataframe=test_df,
        image_dir=args.test_npz_dir,
        label_columns=label_columns,
        transform=preprocess,
        apply_min_max_normalization=(
            metadata[
                "apply_min_max_normalization"
            ]
        ),
        image_key=image_key,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
                device.type == "cuda"
        ),
        persistent_workers=(
                args.num_workers > 0
        ),
    )

    (
        image_ids,
        targets,
        probabilities,
    ) = run_inference(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    (
        summary,
        per_class_df,
        predictions,
    ) = calculate_metrics(
        targets=targets,
        probabilities=probabilities,
        label_columns=label_columns,
        threshold=threshold,
    )

    summary_with_metadata = {
        "evaluation": summary,
        "checkpoint_metadata": metadata,
        "evaluation_config": {
            "npz_image_key": image_key,
            "npz_image_key_source": image_key_source,
        },
        "paths": {
            "test_npz_dir": str(
                args.test_npz_dir
            ),
            "test_csv": str(
                args.test_csv
            ),
            "checkpoint": str(
                args.checkpoint
            ),
            "output_dir": str(
                args.output_dir
            ),
        },
    }

    summary_json_path = (
            args.output_dir
            / "test_summary_metrics.json"
    )

    summary_csv_path = (
            args.output_dir
            / "test_summary_metrics.csv"
    )

    per_class_path = (
            args.output_dir
            / "test_per_class_metrics.csv"
    )

    predictions_path = (
            args.output_dir
            / "test_predictions.csv"
    )

    with summary_json_path.open(
            "w",
            encoding="utf-8",
    ) as file:
        json.dump(
            json_safe(
                summary_with_metadata
            ),
            file,
            ensure_ascii=False,
            indent=2,
        )

    pd.DataFrame([
        summary
    ]).to_csv(
        summary_csv_path,
        index=False,
    )

    per_class_df.to_csv(
        per_class_path,
        index=False,
    )

    save_predictions(
        output_path=predictions_path,
        image_ids=image_ids,
        targets=targets,
        probabilities=probabilities,
        predictions=predictions,
        label_columns=label_columns,
    )

    print_summary(
        summary
    )

    print(
        "\n" + "=" * 70
    )
    print(
        "PER-CLASS METRICS"
    )
    print(
        "=" * 70
    )

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
            float_format=lambda value: (
                "N/A"
                if not math.isfinite(
                    value
                )
                else f"{value:.6f}"
            ),
        )
    )

    print(
        "\nResults saved to:"
    )
    print(
        f"  JSON summary: {summary_json_path}"
    )
    print(
        f"  CSV summary: {summary_csv_path}"
    )
    print(
        "  Per-class metrics: "
        f"{per_class_path}"
    )
    print(
        f"  Predictions: {predictions_path}"
    )

    print(
        "\nNote: the benchmark uses a fixed threshold of 0.5 by default. "
        "Do not optimize the threshold on the test set."
    )


if __name__ == "__main__":
    main()
