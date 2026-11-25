"""Multi-label evaluation of DenseNet-121 on chest X-rays stored in NPZ files.

This script:
- reconstructs DenseNet-121 using timm;
- loads checkpoints in several common formats;
- runs inference on the test set;
- calculates overall and per-class metrics;
- saves predictions and results as CSV/JSON files.
"""

from pathlib import Path
import argparse
import json
import math
import random
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


# ============================================================
# PATH CONFIGURATION - UPDATE BEFORE RUNNING
# ============================================================

TEST_NPZ_DIR = Path("/path/to/vindr-cxr/test")
TEST_CSV = Path("/path/to/vindr-cxr/annotations/classification.csv")
CHECKPOINT_PATH = Path("/path/to/checkpoint.pt")
OUTPUT_DIR = Path("./outputs/densenet/test_results")
# ============================================================
# FALLBACK CONFIGURATION
# ============================================================
# These values are used only when the checkpoint lacks the corresponding metadata.

DEFAULT_MODEL_NAME = "densenet121.tv_in1k"

DEFAULT_LABEL_COLUMNS = [
    "Lung tumor",
    "Pneumonia",
    "Tuberculosis",
    "Other diseases",
    "No finding",
]

DEFAULT_IMAGE_SIZE = 224
DEFAULT_IN_CHANS = 1
DEFAULT_APPLY_MIN_MAX_NORMALIZATION = True
DEFAULT_NPZ_IMAGE_KEY: str | None = None

BATCH_SIZE = 32
NUM_WORKERS = 4
SEED = 42

# Do not select the threshold based on the test set.
# It must be fixed in advance or selected using validation data.
CLASSIFICATION_THRESHOLD = 0.5


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
    image_key: str | None = None,
) -> np.ndarray:
    """Reads and copies the image array stored in an NPZ file."""

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
                    (
                        key
                        for key in preferred_keys
                        if key in archive
                    ),
                    None,
                )

                if selected_key is None:
                    if len(available_keys) == 1:
                        selected_key = available_keys[0]
                    else:
                        raise ValueError(
                            "The NPZ file contains multiple arrays and it was not "
                            "possible to identify the image automatically. "
                            f"Path: {image_path} | keys: {available_keys}. "
                            "Use --npz-image-key to specify the correct key."
                        )

            return np.array(archive[selected_key], copy=True)

    except OSError as error:
        raise OSError(
            f"Could not open the NPZ file:\n{image_path}"
        ) from error


def read_npz(
    image_path: Path,
    apply_min_max_normalization: bool,
    in_chans: int,
    image_key: str | None = None,
) -> torch.Tensor:
    """
    Reads an NPZ file and returns a float32 tensor [C, H, W].

    Dataset images may be stored in grayscale, while the checkpoint may
    have been trained with one or three channels. When the checkpoint expects
    three channels, a grayscale image is repeated across channels.

    Accepted formats:
        [H, W]
        [1, H, W]
        [H, W, 1]
        [3, H, W]
        [H, W, 3]
    """

    if in_chans not in (1, 3):
        raise ValueError(
            f"This script supports checkpoints with in_chans=1 or 3, "
            f"but received in_chans={in_chans}."
        )

    image = load_npz_image_array(
        image_path=image_path,
        image_key=image_key,
    )
    image = np.asarray(image, dtype=np.float32)

    # Convert to CHW format.
    if image.ndim == 2:
        image = image[None, ...]
    elif image.ndim == 3:
        if image.shape[0] in (1, 3):
            pass
        elif image.shape[-1] in (1, 3):
            image = np.moveaxis(image, -1, 0)
        else:
            raise ValueError(
                "The NPZ file has three dimensions, but the channel axis "
                "could not be identified. "
                f"Path: {image_path} | shape: {image.shape}"
            )
    else:
        raise ValueError(
            "Expected an image with shape [H,W], [1,H,W], [H,W,1], "
            "[3,H,W] ou [H,W,3]. "
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
            "Min-max normalization is disabled, but the image "
            "is not within the [0,1] range. "
            f"Path: {image_path} | min={image_min:.6f} | "
            f"max={image_max:.6f}"
        )

    source_channels = int(image.shape[0])

    if source_channels == in_chans:
        pass
    elif source_channels == 1 and in_chans == 3:
        # Repeat grayscale channels to RGB: R = G = B.
        image = np.repeat(image, repeats=3, axis=0)
    else:
        raise ValueError(
            "The image and checkpoint channel counts are incompatible. "
            f"Image: {source_channels} channels | checkpoint: {in_chans} channels | "
            f"Path: {image_path}"
        )

    return torch.from_numpy(
        np.ascontiguousarray(image, dtype=np.float32)
    )


class ResizeAndPad:
    """Preserves aspect ratio and pads to output_size x output_size."""

    def __init__(self, output_size: int):
        self.output_size = output_size

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


def build_test_transform(image_size: int, in_chans: int):
    if in_chans not in (1, 3):
        raise ValueError(
            f"Normalization is not defined for in_chans={in_chans}."
        )

    # [0.5] and [0.5, 0.5, 0.5] apply the same per-channel
    # transformation, but the number of channels is kept explicit.
    mean = [0.5] * in_chans
    std = [0.5] * in_chans

    return transforms.Compose([
        ResizeAndPad(image_size),
        transforms.Normalize(mean=mean, std=std),
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
        apply_min_max_normalization: bool,
        in_chans: int,
        image_key: str | None = None,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.label_columns = label_columns
        self.transform = transform
        self.apply_min_max_normalization = apply_min_max_normalization
        self.in_chans = int(in_chans)
        self.image_key = image_key

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
            apply_min_max_normalization=self.apply_min_max_normalization,
            in_chans=self.in_chans,
            image_key=self.image_key,
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
        raise FileNotFoundError(f"The test CSV file does not exist:\n{csv_path}")

    dataframe = pd.read_csv(csv_path, sep=",")
    required_columns = ["image_id"] + label_columns

    missing_columns = [
        column for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "The test CSV is missing required columns: "
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
            dataframe.loc[dataframe["image_id"].duplicated(), "image_id"]
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
                f"Column '{column}' contains values other than 0/1. "
                f"Examples: {examples}"
            )

    return dataframe


# ============================================================
# CHECKPOINT AND MODEL
# ============================================================


def _looks_like_state_dict(value) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(isinstance(key, str) for key in value)
        and all(torch.is_tensor(item) for item in value.values())
    )


def _remove_state_dict_prefixes(state_dict: dict) -> dict:
    """Removes prefixes commonly added by training wrappers."""
    cleaned = dict(state_dict)

    for prefix in ("module.", "model."):
        if cleaned and all(key.startswith(prefix) for key in cleaned):
            cleaned = {
                key[len(prefix):]: value
                for key, value in cleaned.items()
            }

    return cleaned


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"The checkpoint does not exist:\n{checkpoint_path}"
        )

    # Supports PyTorch versions from before and after the introduction
    # of the weights_only argument in torch.load.
    try:
        loaded = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        loaded = torch.load(
            checkpoint_path,
            map_location=device,
        )

    if not isinstance(loaded, dict):
        raise TypeError(
            "The checkpoint must be a PyTorch dictionary or state_dict."
        )

    # Full checkpoint format used by the training script.
    if "model_state_dict" in loaded:
        checkpoint = dict(loaded)
        state_dict = checkpoint["model_state_dict"]

    # Other common field names.
    elif "state_dict" in loaded and _looks_like_state_dict(loaded["state_dict"]):
        checkpoint = dict(loaded)
        state_dict = checkpoint["state_dict"]

    elif "model" in loaded and _looks_like_state_dict(loaded["model"]):
        checkpoint = dict(loaded)
        state_dict = checkpoint["model"]

    # Checkpoint containing only model weights.
    elif _looks_like_state_dict(loaded):
        checkpoint = {}
        state_dict = loaded

    else:
        raise KeyError(
            "Could not find the model weights. Expected "
            "'model_state_dict', 'state_dict', 'model', or a plain state_dict."
        )

    checkpoint["model_state_dict"] = _remove_state_dict_prefixes(state_dict)
    return checkpoint


def build_model_from_checkpoint(
    checkpoint: dict,
    device: torch.device,
):
    model_name = checkpoint.get("model_name", DEFAULT_MODEL_NAME)
    label_columns = checkpoint.get("label_columns", DEFAULT_LABEL_COLUMNS)
    image_size = int(checkpoint.get("image_size", DEFAULT_IMAGE_SIZE))
    in_chans = int(checkpoint.get("in_chans", DEFAULT_IN_CHANS))
    apply_min_max_normalization = bool(
        checkpoint.get(
            "apply_min_max_normalization",
            DEFAULT_APPLY_MIN_MAX_NORMALIZATION,
        )
    )
    npz_image_key = checkpoint.get(
        "npz_image_key",
        DEFAULT_NPZ_IMAGE_KEY,
    )

    if not isinstance(label_columns, (list, tuple)) or not label_columns:
        raise ValueError("The checkpoint contains invalid 'label_columns'.")

    label_columns = list(label_columns)

    # pretrained=False avoids downloads during testing; all weights are
    # loaded from the checkpoint immediately afterward.
    model = timm.create_model(
        model_name,
        pretrained=False,
        in_chans=in_chans,
        num_classes=len(label_columns),
    )

    incompatible = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint loading failed due to incompatible weights. "
            f"Missing keys: {incompatible.missing_keys} | "
            f"Unexpected keys: {incompatible.unexpected_keys}"
        )

    model = model.to(device)
    model.eval()

    metadata = {
        "model_name": model_name,
        "label_columns": label_columns,
        "image_size": image_size,
        "in_chans": in_chans,
        "apply_min_max_normalization": apply_min_max_normalization,
        "input_format": checkpoint.get("input_format", "npz"),
        "npz_image_key": npz_image_key,
        "training_epoch": checkpoint.get("epoch"),
        "fine_tuning": checkpoint.get("fine_tuning", "unknown"),
        "validation_loss": checkpoint.get("validation_loss"),
        "best_validation_loss": checkpoint.get("best_validation_loss"),
    }

    return model, metadata


# ============================================================
# INFERENCE
# ============================================================


def run_inference(
    model,
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
        )

        for images, labels, image_ids in progress_bar:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probabilities = torch.sigmoid(logits)

            all_image_ids.extend(list(image_ids))
            all_targets.append(labels.numpy())
            all_probabilities.append(probabilities.cpu().numpy())

    targets = np.concatenate(all_targets, axis=0).astype(np.int32)
    probabilities = np.concatenate(all_probabilities, axis=0).astype(np.float64)

    return all_image_ids, targets, probabilities


# ============================================================
# METRICS
# ============================================================


def safe_divide(numerator: float, denominator: float) -> float:
    """Safe division for metrics derived from the confusion matrix."""
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def calculate_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    label_columns: list[str],
    threshold: float,
):
    """Calculates the metrics used in the multi-label benchmark.

    Threshold-independent metrics:
        - mAP: unweighted mean of the per-class AP scores;
        - Micro-AP: global AP across all image-label pairs;
        - Macro-AUROC: unweighted mean of the per-class AUROC scores;
        - Per-class AP.

    Threshold-dependent metrics:
        - Macro-Sensitivity;
        - Macro-Specificity;
        - Macro-F1;
        - Micro-F1.
    """
    if targets.shape != probabilities.shape:
        raise ValueError(
            "Targets and probabilities have different shapes: "
            f"{targets.shape} vs {probabilities.shape}"
        )

    if targets.ndim != 2:
        raise ValueError(
            "Expected targets/probabilities with shape "
            "[n_images, n_classes]."
        )

    if targets.shape[1] != len(label_columns):
        raise ValueError(
            "The number of target columns does not match label_columns: "
            f"{targets.shape[1]} vs {len(label_columns)}"
        )

    targets = targets.astype(np.int32)
    probabilities = probabilities.astype(np.float64)

    if not np.all(np.isin(targets, [0, 1])):
        raise ValueError("Targets must contain only 0/1 values.")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities contain NaN or infinite values.")

    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("Probabilities must be within [0, 1].")

    # Per-class AUROC requires at least one positive and one negative
    # example per class. Because these are the benchmark classes,
    # raise an explicit error instead of silently producing metrics
    # that cannot be compared across models.
    positive_support = targets.sum(axis=0).astype(int)
    negative_support = (targets.shape[0] - positive_support).astype(int)

    invalid_classes = [
        label_columns[i]
        for i in range(len(label_columns))
        if positive_support[i] == 0 or negative_support[i] == 0
    ]
    if invalid_classes:
        raise ValueError(
            "Some classes have no positive or negative test samples, "
            "making AUROC/sensitivity/specificity incomparable for "
            f"these classes: {invalid_classes}"
        )

    predictions = (probabilities >= threshold).astype(np.int32)

    # ------------------------------------------------------------
    # Threshold-independent metrics
    # ------------------------------------------------------------

    # AP for each condition; average=None returns one value per class.
    per_class_ap = np.asarray(
        average_precision_score(
            targets,
            probabilities,
            average=None,
        ),
        dtype=np.float64,
    )

    # mAP is the unweighted arithmetic mean of the per-class AP scores.
    mean_average_precision = float(np.mean(per_class_ap))

    # Micro-AP pools all image-label pairs.
    micro_ap = float(
        average_precision_score(
            targets,
            probabilities,
            average="micro",
        )
    )

    # AUROC for each condition and its macro average.
    per_class_auroc = np.asarray(
        [
            roc_auc_score(targets[:, i], probabilities[:, i])
            for i in range(targets.shape[1])
        ],
        dtype=np.float64,
    )
    macro_auroc = float(np.mean(per_class_auroc))

    # ------------------------------------------------------------
    # Threshold-dependent metrics
    # ------------------------------------------------------------

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

    # Positive-class recall is sensitivity. With average="macro",
    # calculate sensitivity per class and then take the unweighted mean.
    macro_sensitivity = float(
        recall_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    # sklearn has no dedicated multi-label specificity argument;
    # compute TN / (TN + FP) per class, then take the macro average.
    per_class_rows = []
    per_class_specificity = []

    for class_index, class_name in enumerate(label_columns):
        class_targets = targets[:, class_index]
        class_predictions = predictions[:, class_index]

        tp = int(np.sum((class_targets == 1) & (class_predictions == 1)))
        tn = int(np.sum((class_targets == 0) & (class_predictions == 0)))
        fp = int(np.sum((class_targets == 0) & (class_predictions == 1)))
        fn = int(np.sum((class_targets == 1) & (class_predictions == 0)))

        sensitivity = safe_divide(tp, tp + fn)
        specificity = safe_divide(tn, tn + fp)
        class_f1 = float(
            f1_score(
                class_targets,
                class_predictions,
                zero_division=0,
            )
        )

        per_class_specificity.append(specificity)

        per_class_rows.append({
            "class": class_name,
            "support_positive": int(positive_support[class_index]),
            "support_negative": int(negative_support[class_index]),
            "prevalence": float(
                positive_support[class_index] / targets.shape[0]
            ),
            "AP": float(per_class_ap[class_index]),
            "AUROC": float(per_class_auroc[class_index]),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "F1": class_f1,
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
        })

    macro_specificity = float(
        np.mean(np.asarray(per_class_specificity, dtype=np.float64))
    )

    per_class_df = pd.DataFrame(per_class_rows)

    summary = {
        "number_of_images": int(targets.shape[0]),
        "number_of_classes": int(targets.shape[1]),
        "classification_threshold": float(threshold),

        # Threshold-independent metrics
        "mAP": mean_average_precision,
        "micro_AP": micro_ap,
        "macro_AUROC": macro_auroc,

        # Threshold-dependent metrics
        "macro_sensitivity": macro_sensitivity,
        "macro_specificity": macro_specificity,
        "macro_F1": macro_f1,
        "micro_F1": micro_f1,
    }

    return summary, per_class_df, predictions


# ============================================================
# SAVE RESULTS
# ============================================================


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        numeric_value = float(value)
        if math.isnan(numeric_value) or math.isinf(numeric_value):
            return None
        return numeric_value

    return value


def save_predictions(
    output_path: Path,
    image_ids: list[str],
    targets: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    label_columns: list[str],
) -> None:
    data = {"image_id": image_ids}

    for class_index, class_name in enumerate(label_columns):
        data[f"target__{class_name}"] = targets[:, class_index]
        data[f"probability__{class_name}"] = probabilities[:, class_index]
        data[f"prediction__{class_name}"] = predictions[:, class_index]

    pd.DataFrame(data).to_csv(output_path, index=False)


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

        if isinstance(value, float):
            if math.isnan(value):
                formatted = "N/A"
            else:
                formatted = f"{value:.6f}"
        else:
            formatted = str(value)

        print(f"{key}: {formatted}")


# ============================================================
# OPTIONAL ARGUMENTS
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluates a DenseNet checkpoint on multi-label NPZ images. "
            "Without arguments, uses the default paths defined above."
        )
    )

    parser.add_argument(
        "--test-npz-dir",
        type=Path,
        default=TEST_NPZ_DIR,
        help="Directory containing test .npz files.",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=TEST_CSV,
        help="Test CSV containing image_id and class label columns.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help="Checkpoint .pt file to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where results will be saved.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--npz-image-key",
        type=str,
        default=None,
        help=(
            "Image array key inside the NPZ archive. If omitted, uses the "
            "key stored in the checkpoint or automatic detection."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=CLASSIFICATION_THRESHOLD,
        help="Fixed threshold for converting probabilities to 0/1.",
    )

    args = parser.parse_args()

    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be between 0 and 1.")

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1.")

    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0.")

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

    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, device)
    model, metadata = build_model_from_checkpoint(checkpoint, device)

    label_columns = metadata["label_columns"]
    image_size = metadata["image_size"]
    in_chans = int(metadata["in_chans"])
    apply_min_max_normalization = metadata[
        "apply_min_max_normalization"
    ]
    npz_image_key = (
        args.npz_image_key
        if args.npz_image_key is not None
        else metadata["npz_image_key"]
    )

    if in_chans not in (1, 3):
        raise ValueError(
            f"Checkpoint has in_chans={in_chans}; only 1 or 3 are supported."
        )

    print(f"Model: {metadata['model_name']}")
    print(f"Fine-tuning protocol: {metadata['fine_tuning']}")
    print(f"Checkpoint epoch: {metadata['training_epoch']}")
    print(f"Image size: {image_size}")
    print(f"Input channels: {in_chans}")
    print(f"Classes: {label_columns}")
    print(
        "Min-max normalization: "
        f"{apply_min_max_normalization}"
    )
    print(
        "NPZ key: "
        + (
            repr(npz_image_key)
            if npz_image_key is not None
            else "automatic detection"
        )
    )
    print(f"Threshold: {args.threshold}")

    test_df = load_test_csv(
        csv_path=args.test_csv,
        label_columns=label_columns,
    )

    print(f"\nTest CSV: {args.test_csv}")
    print(f"Test NPZ directory: {args.test_npz_dir}")
    print(f"Number of images: {len(test_df)}")

    print("\nPositive samples per class in the test set:")
    for label in label_columns:
        print(f"  {label}: {int(test_df[label].sum())}")

    test_dataset = CXRTestDataset(
        dataframe=test_df,
        image_dir=args.test_npz_dir,
        label_columns=label_columns,
        transform=build_test_transform(image_size, in_chans),
        apply_min_max_normalization=apply_min_max_normalization,
        in_chans=in_chans,
        image_key=npz_image_key,
    )

    loader_arguments = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }

    test_loader = DataLoader(
        test_dataset,
        **loader_arguments,
    )

    image_ids, targets, probabilities = run_inference(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    summary, per_class_df, predictions = calculate_metrics(
        targets=targets,
        probabilities=probabilities,
        label_columns=label_columns,
        threshold=args.threshold,
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
        label_columns=label_columns,
    )

    print_summary(summary)

    print("\n" + "=" * 70)
    print("PER-CLASS METRICS")
    print("=" * 70)
    print(
        per_class_df[
            [
                "class",
                "support_positive",
                "support_negative",
                "AP",
                "AUROC",
                "sensitivity",
                "specificity",
                "F1",
            ]
        ].to_string(index=False)
    )

    print("\nResults saved to:")
    print(f"  JSON summary: {summary_json_path}")
    print(f"  CSV summary: {summary_csv_path}")
    print(f"  Per-class metrics: {per_class_path}")
    print(f"  Per-image predictions: {predictions_path}")


if __name__ == "__main__":
    main()
