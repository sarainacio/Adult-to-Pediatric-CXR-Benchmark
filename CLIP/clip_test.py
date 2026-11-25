from pathlib import Path
import argparse
import json
import math
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from clip_dataset import (
    DEFAULT_LABEL_COLUMNS,
    load_npz_image_array,
)
from clip_model import load_clip_classifier


# ============================================================
# PATH CONFIGURATION - UPDATE BEFORE RUNNING
# ============================================================

TEST_NPZ_DIR = Path("/path/to/vindr-cxr/test_npz")
TEST_CSV = Path("/path/to/annotations/test.csv")
CHECKPOINT_PATH = Path("/path/to/checkpoint.pt")
OUTPUT_DIR = Path("./outputs/clip/test_results")


# ============================================================
# FALLBACK CONFIGURATION
# ============================================================
# These values are used only when the corresponding checkpoint metadata is missing.

DEFAULT_MODEL_NAME = "ViT-B/16"
DEFAULT_LABEL_COLUMNS_FALLBACK = list(DEFAULT_LABEL_COLUMNS)
DEFAULT_DATA_NORMALIZATION = "minmax"
DEFAULT_CLASSIFICATION_THRESHOLD = 0.5

BATCH_SIZE = 32
NUM_WORKERS = 4
SEED = 42

# The fine-tuning checkpoints were trained in float32.
EVALUATE_IN_FLOAT32 = True


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


def load_npz_grayscale(
    image_path: Path,
    normalization: str,
    image_key: str | None = None,
) -> np.ndarray:
    """
    Reads an NPZ file and returns a two-dimensional float32 image in [0, 1].

    Accepted shapes:
        [H, W]
        [1, H, W]
        [H, W, 1]

    Supported normalization modes:
        "minmax":
            applies per-image min-max normalization;

        "none":
            requires the image to already be in [0, 1].
    """

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
                "The NPZ array has three dimensions but does not represent "
                "a single-channel image. "
                f"Path: {image_path} | shape: {image.shape}"
            )

    if image.ndim != 2:
        raise ValueError(
            "Expected an image with shape [H,W], [1,H,W], or [H,W,1]. "
            f"Path: {image_path} | shape: {image.shape}"
        )

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    normalization_key = str(normalization).strip().lower()
    normalization_key = normalization_key.replace("-", "").replace("_", "")

    image_min = float(image.min())
    image_max = float(image.max())

    if normalization_key == "minmax":
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min)
        else:
            image = np.zeros_like(image, dtype=np.float32)

    elif normalization_key in {"none", "identity", "already01"}:
        if image_min < 0.0 or image_max > 1.0:
            raise ValueError(
                "DATA_NORMALIZATION indicates that no normalization should be applied, "
                "but the image is not in [0,1]. "
                f"Path: {image_path} | min={image_min:.6f} | "
                f"max={image_max:.6f}"
            )

    else:
        raise ValueError(
            "Unsupported normalization in this script: "
            f"{normalization!r}. Use 'minmax' or 'none', or adapt "
            "load_npz_grayscale to exactly reproduce training."
        )

    image = np.clip(image, 0.0, 1.0)
    return np.ascontiguousarray(image, dtype=np.float32)


def npz_to_rgb_pil(
    image_path: Path,
    normalization: str,
    image_key: str | None = None,
) -> Image.Image:
    """
    Converts a single-channel NPZ image to a PIL RGB image.

    The official CLIP preprocessing is subsequently applied by the Dataset.
    """

    image = load_npz_grayscale(
        image_path=image_path,
        normalization=normalization,
        image_key=image_key,
    )

    image_uint8 = np.round(image * 255.0).astype(np.uint8)
    return Image.fromarray(image_uint8, mode="L").convert("RGB")


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

    dataframe = pd.read_csv(csv_path, sep=",")
    required_columns = ["image_id"] + label_columns

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns in the test CSV: "
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

    duplicated_mask = dataframe["image_id"].duplicated()

    if duplicated_mask.any():
        duplicated_examples = (
            dataframe.loc[duplicated_mask, "image_id"]
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
                dataframe.loc[numeric_values.isna(), column]
                .head(5)
                .tolist()
            )
            raise ValueError(
                f"Column '{column}' contains non-numeric values. "
                f"Examples: {examples}"
            )

        invalid_values = ~numeric_values.isin([0, 1])

        if invalid_values.any():
            examples = numeric_values.loc[invalid_values].head(5).tolist()
            raise ValueError(
                f"Column '{column}' contains values other than 0/1. "
                f"Examples: {examples}"
            )

        dataframe[column] = numeric_values.astype(np.float32)

    return dataframe


class CLIPTestDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: Path,
        label_columns: list[str],
        preprocess,
        normalization: str,
        image_key: str | None = None,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.label_columns = list(label_columns)
        self.preprocess = preprocess
        self.normalization = normalization
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

        image = npz_to_rgb_pil(
            image_path=image_path,
            normalization=self.normalization,
            image_key=self.image_key,
        )

        if self.preprocess is not None:
            image = self.preprocess(image)

        labels = torch.tensor(
            row[self.label_columns].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

        return image, labels, image_id


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

    # weights_only=False is required for complete checkpoints that may
    # include metrics, configurations, and optimizer states.
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
            "The checkpoint must be a dictionary containing "
            "'model_state_dict'."
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "The checkpoint does not contain the 'model_state_dict' key."
        )

    return checkpoint


def remove_prefix_from_state_dict(
    state_dict: dict,
    prefix: str,
) -> dict:
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {
            key[len(prefix):]: value
            for key, value in state_dict.items()
        }

    return state_dict


def load_model_state_strict(
    model: torch.nn.Module,
    state_dict: dict,
) -> None:
    """
    Loads the model state with strict=True.

    Also supports checkpoints saved with DataParallel
    ("module.") or torch.compile ("_orig_mod.").
    """

    candidate_state_dicts = [
        state_dict,
        remove_prefix_from_state_dict(state_dict, "module."),
        remove_prefix_from_state_dict(state_dict, "_orig_mod."),
    ]

    errors = []

    for candidate in candidate_state_dicts:
        try:
            model.load_state_dict(candidate, strict=True)
            return
        except RuntimeError as error:
            errors.append(str(error))

    raise RuntimeError(
        "Could not load model_state_dict with strict=True.\n"
        + "\n\n".join(errors)
    )


def build_model_from_checkpoint(
    checkpoint: dict,
    device: torch.device,
):
    model_name = str(
        checkpoint.get("model_name", DEFAULT_MODEL_NAME)
    )

    label_columns = checkpoint.get(
        "label_columns",
        DEFAULT_LABEL_COLUMNS_FALLBACK,
    )

    if not isinstance(label_columns, (list, tuple)) or not label_columns:
        raise ValueError(
            "The checkpoint's 'label_columns' field is invalid."
        )

    label_columns = list(label_columns)
    num_classes = len(label_columns)

    checkpoint_num_classes = checkpoint.get("num_classes")

    if (
        checkpoint_num_classes is not None
        and int(checkpoint_num_classes) != num_classes
    ):
        raise ValueError(
            "The checkpoint contains a num_classes value incompatible with "
            "label_columns: "
            f"{checkpoint_num_classes} vs {num_classes}."
        )

    model, preprocess = load_clip_classifier(
        model_name=model_name,
        num_classes=num_classes,
        dropout=0.0,
        device=device,
    )

    if EVALUATE_IN_FLOAT32:
        model = model.float()

    load_model_state_strict(
        model=model,
        state_dict=checkpoint["model_state_dict"],
    )

    model = model.to(device)
    model.eval()

    normalization = checkpoint.get(
        "data_normalization",
        DEFAULT_DATA_NORMALIZATION,
    )

    checkpoint_threshold = float(
        checkpoint.get(
            "threshold",
            DEFAULT_CLASSIFICATION_THRESHOLD,
        )
    )

    metadata = {
        "model_name": model_name,
        "label_columns": label_columns,
        "num_classes": num_classes,
        "protocol": checkpoint.get(
            "protocol",
            checkpoint.get("fine_tuning", "unknown"),
        ),
        "fine_tuning": checkpoint.get("fine_tuning", "unknown"),
        "training_epoch": checkpoint.get("epoch"),
        "checkpoint_threshold": checkpoint_threshold,
        "data_normalization": normalization,
        "input_format": checkpoint.get("input_format"),
        "npz_image_key": checkpoint.get("npz_image_key"),
        "dataset_implementation": checkpoint.get(
            "dataset_implementation"
        ),
        "training_dtype": checkpoint.get("training_dtype"),
        "validation_loss": checkpoint.get("validation_loss"),
        "best_validation_loss": checkpoint.get(
            "best_validation_loss"
        ),
        "validation_metrics": checkpoint.get("validation_metrics"),
        "trainable_layers": checkpoint.get("trainable_layers"),
    }

    return model, preprocess, metadata


# ============================================================
# INFERENCE
# ============================================================


def run_inference(
    model: torch.nn.Module,
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

        for images, labels, image_ids in progress_bar:
            images = images.to(
                device,
                non_blocking=True,
            )

            logits = model(images)
            probabilities = torch.sigmoid(logits)

            all_image_ids.extend(list(image_ids))
            all_targets.append(labels.numpy())
            all_probabilities.append(
                probabilities.cpu().numpy()
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
    """Computes the metrics used in the multi-label benchmark.

    Threshold-independent metrics:
        - mAP: unweighted mean of per-class AP;
        - Micro-AP: AP computed over all image-label pairs;
        - Macro-AUROC: unweighted mean of per-class AUROC;
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
        raise ValueError("Targets must contain only binary values (0/1).")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities contain NaN or infinite values.")

    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("Probabilities must be in [0, 1].")

    # AUROC, sensitivity, and specificity require positive and
    # negative examples in each class. We explicitly stop if the
    # test set does not satisfy this condition, to avoid silently producing
    # metrics that are not comparable across models.
    positive_support = targets.sum(axis=0).astype(int)
    negative_support = (targets.shape[0] - positive_support).astype(int)

    invalid_classes = [
        label_columns[i]
        for i in range(len(label_columns))
        if positive_support[i] == 0 or negative_support[i] == 0
    ]
    if invalid_classes:
        raise ValueError(
            "Some classes have no positive or negative test examples, "
            "making AUROC/sensitivity/specificity non-comparable for "
            f"these classes: {invalid_classes}"
        )

    predictions = (probabilities >= threshold).astype(np.int32)

    # ------------------------------------------------------------
    # Threshold-independent metrics
    # ------------------------------------------------------------

    # Per-class AP: average=None returns one value for each class.
    per_class_ap = np.asarray(
        average_precision_score(
            targets,
            probabilities,
            average=None,
        ),
        dtype=np.float64,
    )

    # mAP is the unweighted arithmetic mean of the per-class AP values.
    mean_average_precision = float(np.mean(per_class_ap))

    # Micro-AP pools all image-label pairs.
    micro_ap = float(
        average_precision_score(
            targets,
            probabilities,
            average="micro",
        )
    )

    # Per-class AUROC and its macro average.
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

    # Positive-class recall equals sensitivity. With average="macro",
    # sensitivity is computed for each class and then averaged without weighting.
    macro_sensitivity = float(
        recall_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    # Specificity has no dedicated multilabel argument in sklearn;
    # we compute it per class as TN / (TN + FP), then take the macro average.
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
# SAVE AND DISPLAY RESULTS
# ============================================================


def json_safe(value):
    if isinstance(value, dict):
        return {
            key: json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return [
            json_safe(item)
            for item in value.tolist()
        ]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, (np.floating, float)):
        numeric_value = float(value)

        if (
            math.isnan(numeric_value)
            or math.isinf(numeric_value)
        ):
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
    data = {
        "image_id": image_ids,
    }

    for class_index, class_name in enumerate(label_columns):
        data[
            f"target__{class_name}"
        ] = targets[:, class_index]

        data[
            f"probability__{class_name}"
        ] = probabilities[:, class_index]

        data[
            f"prediction__{class_name}"
        ] = predictions[:, class_index]

    pd.DataFrame(data).to_csv(
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

        if isinstance(value, float):
            if not math.isfinite(value):
                formatted = "N/A"
            else:
                formatted = f"{value:.6f}"
        else:
            formatted = str(value)

        print(f"{key}: {formatted}")


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluates a multi-label CLIP checkpoint on NPZ images. "
            "Supports linear probing, partial fine-tuning, "
            "and full fine-tuning. Without arguments, uses the paths "
            "defined at the beginning of the script."
        )
    )

    parser.add_argument(
        "--test-npz-dir",
        type=Path,
        default=TEST_NPZ_DIR,
        help="Directory containing the test .npz files.",
    )

    parser.add_argument(
        "--npz-image-key",
        type=str,
        default=None,
        help=(
            "Image array key inside the NPZ file. "
            "If omitted, uses the checkpoint key or automatic detection."
        ),
    )

    parser.add_argument(
        "--test-csv",
        type=Path,
        default=TEST_CSV,
        help="Test CSV containing image_id and labels.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help=".pt checkpoint to evaluate.",
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
        "--threshold",
        type=float,
        default=DEFAULT_CLASSIFICATION_THRESHOLD,
        help=(
            "Fixed threshold for converting probabilities to 0/1. "
            "Defaults to 0.5, as in the other benchmark models."
        ),
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1.")

    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0.")

    if (
        args.threshold is not None
        and not 0.0 < args.threshold < 1.0
    ):
        parser.error("--threshold must be between 0 and 1.")

    return args


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    args = parse_args()
    set_seed(SEED)

    if not args.test_npz_dir.exists():
        raise FileNotFoundError(
            "The test NPZ directory does not exist:\n"
            f"{args.test_npz_dir}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Checkpoint: {args.checkpoint}")

    checkpoint = load_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    model, preprocess, metadata = build_model_from_checkpoint(
        checkpoint=checkpoint,
        device=device,
    )

    label_columns = metadata["label_columns"]
    normalization = metadata["data_normalization"]
    npz_image_key = (
        args.npz_image_key
        if args.npz_image_key is not None
        else metadata.get("npz_image_key")
    )

    threshold = float(args.threshold)

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "The selected threshold must be between 0 and 1, "
            f"but received {threshold}."
        )

    print(f"Model: {metadata['model_name']}")
    print(f"Protocol: {metadata['protocol']}")
    print(f"Checkpoint epoch: {metadata['training_epoch']}")
    print(f"Classes: {label_columns}")
    print(f"NPZ normalization: {normalization}")
    print(
        "NPZ image key: "
        + (repr(npz_image_key) if npz_image_key is not None else "automatic")
    )
    print(f"Threshold: {threshold}")
    print(f"Evaluation in float32: {EVALUATE_IN_FLOAT32}")

    test_df = load_test_csv(
        csv_path=args.test_csv,
        label_columns=label_columns,
    )

    print(f"\nTest CSV: {args.test_csv}")
    print(f"Test NPZ directory: {args.test_npz_dir}")
    print(f"Number of images: {len(test_df)}")

    print("\nPositive samples per class in the test set:")

    for label in label_columns:
        print(
            f"  {label}: "
            f"{int(test_df[label].sum())}"
        )

    test_dataset = CLIPTestDataset(
        dataframe=test_df,
        image_dir=args.test_npz_dir,
        label_columns=label_columns,
        preprocess=preprocess,
        normalization=normalization,
        image_key=npz_image_key,
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
        label_columns=label_columns,
        threshold=threshold,
    )

    summary_with_metadata = {
        "evaluation": summary,
        "checkpoint_metadata": metadata,
        "paths": {
            "test_npz_dir": str(args.test_npz_dir),
            "npz_image_key": npz_image_key,
            "test_csv": str(args.test_csv),
            "checkpoint": str(args.checkpoint),
            "output_dir": str(args.output_dir),
        },
    }

    summary_json_path = (
        args.output_dir / "test_summary_metrics.json"
    )
    summary_csv_path = (
        args.output_dir / "test_summary_metrics.csv"
    )
    per_class_path = (
        args.output_dir / "test_per_class_metrics.csv"
    )
    predictions_path = (
        args.output_dir / "test_predictions.csv"
    )

    with summary_json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_safe(summary_with_metadata),
            file,
            ensure_ascii=False,
            indent=2,
        )

    pd.DataFrame([summary]).to_csv(
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

    print_summary(summary)

    print("\n" + "=" * 70)
    print("PER-CLASS METRICS")
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
            float_format=lambda value: (
                "N/A"
                if not math.isfinite(value)
                else f"{value:.6f}"
            ),
        )
    )

    print("\nResults saved to:")
    print(f"  JSON summary: {summary_json_path}")
    print(f"  CSV summary: {summary_csv_path}")
    print(f"  Per-class metrics: {per_class_path}")
    print(f"  Predictions: {predictions_path}")

    print(
        "\nNote: do not optimize the threshold on the "
        "test set. Use a previously fixed threshold or one "
        "selected on the validation set."
    )


if __name__ == "__main__":
    main()
