from pathlib import Path
import argparse
import json
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from health_multimodal.image.model.pretrained import (
    get_biovil_t_image_encoder,
)


# ============================================================
# PATH CONFIGURATION - UPDATE BEFORE RUNNING
# ============================================================

TEST_NPZ_DIR = Path("/path/to/vindr-cxr/test_npz")
TEST_CSV = Path("/path/to/annotations/test.csv")
CHECKPOINT_PATH = Path("/path/to/checkpoint.pt")
OUTPUT_DIR = Path("./outputs/biovilt/test_results")

# ============================================================
# FALLBACK CONFIGURATION
# ============================================================
# These defaults are used only when the corresponding checkpoint metadata is missing.

DEFAULT_MODEL_NAME = "BioViL-T"
DEFAULT_MODEL_ID = "microsoft/BiomedVLP-BioViL-T"

DEFAULT_IMAGE_WEIGHTS_FILENAME = (
    "biovil_t_image_model_proj_size_128.pt"
)

DEFAULT_LABEL_COLUMNS = [
    "Lung tumor",
    "Pneumonia",
    "Tuberculosis",
    "Other diseases",
    "No finding",
]

DEFAULT_IMAGE_SIZE = 224
DEFAULT_FEATURE_SIZE = 512
DEFAULT_DATA_NORMALIZATION = "minmax"
DEFAULT_CLASSIFICATION_THRESHOLD = 0.5
DEFAULT_NPZ_IMAGE_KEY: str | None = None

PREFERRED_NPZ_IMAGE_KEYS = (
    "image",
    "arr_0",
    "pixel_array",
    "pixels",
    "data",
)

BATCH_SIZE = 32
NUM_WORKERS = 4
SEED = 42

# The checkpoints were trained in float32 without AMP.
EVALUATE_IN_FLOAT32 = True


# ============================================================
# REPRODUCIBILITY
# ============================================================


def set_seed(
    seed: int,
) -> None:
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
    Extracts the image from an NPZ archive.

    The array key can be specified explicitly. Otherwise, common keys are
    checked first; if the archive contains only one array, that key is
    selected automatically.
    """

    try:
        loaded = np.load(
            image_path,
            allow_pickle=False,
        )
    except Exception as error:
        raise RuntimeError(
            f"Could not open the NPZ file:\n{image_path}"
        ) from error

    if not isinstance(
        loaded,
        np.lib.npyio.NpzFile,
    ):
        raise TypeError(
            "Expected an NPZ archive, but np.load did not return "
            f"an NpzFile. Path: {image_path}"
        )

    with loaded as archive:
        available_keys = list(
            archive.files
        )

        if not available_keys:
            raise ValueError(
                f"The NPZ archive contains no arrays:\n{image_path}"
            )

        requested_key = (
            None
            if image_key is None
            else str(image_key).strip()
        )

        if requested_key:
            if requested_key not in available_keys:
                raise KeyError(
                    f"The NPZ key {requested_key!r} does not exist in "
                    f"{image_path}. Available keys: {available_keys}"
                )

            selected_key = requested_key

        else:
            selected_key = next(
                (
                    key
                    for key in PREFERRED_NPZ_IMAGE_KEYS
                    if key in available_keys
                ),
                None,
            )

            if selected_key is None:
                if len(available_keys) == 1:
                    selected_key = available_keys[0]
                else:
                    raise ValueError(
                        "Could not automatically determine which "
                        "NPZ array contains the image. "
                        f"Path: {image_path} | "
                        f"available keys: {available_keys}. "
                        "Use --npz-image-key to select the array key."
                    )

        image = np.array(
            archive[selected_key],
            copy=True,
        )

    return image


def normalize_npz_image(
    image: np.ndarray,
    normalization: str,
    image_path: Path,
) -> np.ndarray:
    """
    Normalizes a single-channel NPZ image and returns float32 values in [0, 1].

    Accepted formats:
        [H, W]
        [1, H, W]
        [H, W, 1]

    Supported modes:
        "minmax":
            applies per-image min-max normalization;

        "none":
            requires the image to already be in [0, 1].
    """

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
                "The NPZ array has three dimensions but does not represent "
                "a single-channel image. "
                f"Path: {image_path} | shape: {image.shape}"
            )

    if image.ndim != 2:
        raise ValueError(
            "Expected an NPZ image with shape [H,W], [1,H,W] "
            "or [H,W,1]. "
            f"Path: {image_path} | shape: {image.shape}"
        )

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    normalization_key = (
        str(normalization)
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
    )

    image_min = float(
        image.min()
    )

    image_max = float(
        image.max()
    )

    if normalization_key == "minmax":
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

    elif normalization_key in {
        "none",
        "identity",
        "already01",
    }:
        if (
            image_min < 0.0
            or image_max > 1.0
        ):
            raise ValueError(
                "The checkpoint indicates that the image should already be "
                "in [0,1], but this condition is not met. "
                f"Path: {image_path} | min={image_min:.6f} | "
                f"max={image_max:.6f}"
            )

    else:
        raise ValueError(
            "Unsupported normalization: "
            f"{normalization!r}. Use 'minmax' or 'none'."
        )

    return np.ascontiguousarray(
        np.clip(
            image,
            0.0,
            1.0,
        ),
        dtype=np.float32,
    )


def load_npz_as_grayscale_pil(
    image_path: Path,
    normalization: str,
    image_key: str | None,
) -> Image.Image:
    """
    Loads a chest X-ray from an NPZ file and converts it to a grayscale PIL image.
    """

    image = load_npz_image_array(
        image_path=image_path,
        image_key=image_key,
    )

    image = normalize_npz_image(
        image=image,
        normalization=normalization,
        image_path=image_path,
    )

    image_uint8 = np.round(
        image * 255.0
    ).astype(np.uint8)

    return Image.fromarray(
        image_uint8,
        mode="L",
    )


class ExpandChannels:
    """
    Repeats a single-channel chest X-ray along the channel axis.

    Input:
        [1, H, W]

    Output:
        [3, H, W]
    """

    def __call__(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        if (
            image.ndim != 3
            or image.shape[0] != 1
        ):
            raise ValueError(
                "ExpandChannels expected a [1,H,W] tensor, "
                f"but received {tuple(image.shape)}."
            )

        return image.repeat(
            3,
            1,
            1,
        )


def build_biovil_t_preprocess(
    image_size: int,
):
    """
    Reproduces the preprocessing used in the BioViL-T training scripts.

    Pipeline:
        Resize(image_size)
        CenterCrop(image_size)
        ToTensor()
        channel repetition to RGB

    No mean/std normalization is applied.
    """

    image_size = int(
        image_size
    )

    if image_size < 1:
        raise ValueError(
            "image_size must be a positive integer."
        )

    return transforms.Compose([
        transforms.Resize(
            image_size,
            interpolation=(
                transforms.InterpolationMode.BILINEAR
            ),
            antialias=True,
        ),
        transforms.CenterCrop(
            image_size
        ),
        transforms.ToTensor(),
        ExpandChannels(),
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
            f"The test CSV file does not exist:\n{csv_path}"
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


class BioViLTTestDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: Path,
        label_columns: list[str],
        transform,
        normalization: str,
        npz_image_key: str | None,
    ):
        self.dataframe = dataframe.reset_index(
            drop=True
        )
        self.image_dir = Path(
            image_dir
        )
        self.label_columns = list(
            label_columns
        )
        self.transform = transform
        self.normalization = normalization
        self.npz_image_key = npz_image_key

    def __len__(
        self,
    ) -> int:
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

        image = load_npz_as_grayscale_pil(
            image_path=image_path,
            normalization=self.normalization,
            image_key=self.npz_image_key,
        )

        if self.transform is not None:
            image = self.transform(
                image
            )

        if not isinstance(
            image,
            torch.Tensor,
        ):
            raise TypeError(
                "The BioViL-T transform must return "
                "a torch.Tensor."
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
            "The checkpoint does not contain the 'model_state_dict' key."
        )

    return checkpoint


class BioViLTClassifier(nn.Module):
    """
    BioViL-T visual encoder with a linear multi-label classifier.

    The encoder receives only the current chest X-ray:
        previous_image=None

    The classifier uses the global pre-projection embedding.
    """

    def __init__(
        self,
        encoder: nn.Module,
        feature_size: int,
        num_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.encoder = encoder
        self.feature_size = int(
            feature_size
        )

        if self.feature_size < 1:
            raise ValueError(
                "feature_size must be positive."
            )

        self.classifier = nn.Sequential(
            nn.Dropout(
                p=dropout
            ),
            nn.Linear(
                self.feature_size,
                int(num_classes),
            ),
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        features = self.encoder(
            current_image=images,
            previous_image=None,
            return_patch_embeddings=False,
        )

        if isinstance(
            features,
            tuple,
        ):
            raise TypeError(
                "The BioViL-T encoder returned a tuple when "
                "only the global embedding was expected."
            )

        if features.ndim != 2:
            raise ValueError(
                "Expected an embedding with shape [batch, dimension], "
                f"but received {tuple(features.shape)}."
            )

        if (
            features.shape[1]
            != self.feature_size
        ):
            raise ValueError(
                "The BioViL-T embedding dimension is incompatible "
                "with the classifier: "
                f"{features.shape[1]} vs {self.feature_size}."
            )

        return self.classifier(
            features
        )


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
    Loads the full model state using strict=True.

    Also supports checkpoints created with:
        DataParallel/DDP: "module."
        torch.compile: "_orig_mod."
        additional wrapper: "model."
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
            "The checkpoint's 'label_columns' field is invalid."
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
            "The checkpoint contains a num_classes value incompatible with "
            "label_columns: "
            f"{checkpoint_num_classes} vs {num_classes}."
        )

    try:
        pretrained_image_model = (
            get_biovil_t_image_encoder()
        )

    except Exception as error:
        raise RuntimeError(
            "Could not build the BioViL-T visual encoder. "
            "Ensure that 'health_multimodal' is installed and that "
            "the local cache or Hugging Face is accessible."
        ) from error

    if not hasattr(
        pretrained_image_model,
        "encoder",
    ):
        raise AttributeError(
            "The BioViL-T model does not have an 'encoder' attribute."
        )

    detected_feature_size = getattr(
        pretrained_image_model,
        "feature_size",
        None,
    )

    feature_size = int(
        checkpoint.get(
            "feature_size",
            (
                detected_feature_size
                if detected_feature_size is not None
                else DEFAULT_FEATURE_SIZE
            ),
        )
    )

    if (
        detected_feature_size is not None
        and int(detected_feature_size)
        != feature_size
    ):
        raise ValueError(
            "The loaded BioViL-T encoder dimension does not match "
            "the checkpoint: "
            f"{detected_feature_size} vs {feature_size}."
        )

    model = BioViLTClassifier(
        encoder=pretrained_image_model.encoder,
        feature_size=feature_size,
        num_classes=num_classes,
        dropout=0.0,
    )

    if EVALUATE_IN_FLOAT32:
        model = model.float()

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

    normalization = str(
        checkpoint.get(
            "data_normalization",
            DEFAULT_DATA_NORMALIZATION,
        )
    )

    checkpoint_threshold = float(
        checkpoint.get(
            "threshold",
            DEFAULT_CLASSIFICATION_THRESHOLD,
        )
    )

    preprocess = build_biovil_t_preprocess(
        image_size=image_size,
    )

    metadata = {
        "model_name": checkpoint.get(
            "model_name",
            DEFAULT_MODEL_NAME,
        ),
        "model_id": checkpoint.get(
            "model_id",
            DEFAULT_MODEL_ID,
        ),
        "image_weights_filename": checkpoint.get(
            "image_weights_filename",
            DEFAULT_IMAGE_WEIGHTS_FILENAME,
        ),
        "model_source": checkpoint.get(
            "model_source",
            "huggingface_health_multimodal",
        ),
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
        "label_columns": label_columns,
        "num_classes": num_classes,
        "feature_size": feature_size,
        "feature_source": checkpoint.get(
            "feature_source",
            "global_embedding_pre_projection",
        ),
        "single_image_mode": checkpoint.get(
            "single_image_mode",
            True,
        ),
        "previous_image": checkpoint.get(
            "previous_image"
        ),
        "temporal_transformer_active": checkpoint.get(
            "temporal_transformer_active",
            False,
        ),
        "image_size": image_size,
        "official_resize_size": checkpoint.get(
            "official_resize_size"
        ),
        "official_crop_size": checkpoint.get(
            "official_crop_size"
        ),
        "data_normalization": normalization,
        "npz_image_key": checkpoint.get(
            "npz_image_key",
            DEFAULT_NPZ_IMAGE_KEY,
        ),
        "dataset_implementation": checkpoint.get(
            "dataset_implementation",
            "standalone_npz",
        ),
        "checkpoint_threshold": (
            checkpoint_threshold
        ),
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
# EVALUATION METRICS
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
    Metrics used in the benchmark:
      - mAP: unweighted mean of the AP for each class
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
            "The number of target columns does not match label_columns."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities contain NaN or infinite values.")

    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("Probabilities must be in [0,1].")

    targets = targets.astype(np.int32)

    if not np.all((targets == 0) | (targets == 1)):
        raise ValueError("Targets must contain only binary values (0/1).")

    positive_supports = targets.sum(axis=0)
    negative_supports = targets.shape[0] - positive_supports

    invalid_classes = [
        label_columns[i]
        for i in range(len(label_columns))
        if positive_supports[i] == 0 or negative_supports[i] == 0
    ]

    if invalid_classes:
        raise ValueError(
            "Some test classes have no positive or no negative samples: "
            f"{invalid_classes}. Macro metrics would not be comparable."
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

        tp = int(np.sum((class_targets == 1) & (class_predictions == 1)))
        tn = int(np.sum((class_targets == 0) & (class_predictions == 0)))
        fp = int(np.sum((class_targets == 0) & (class_predictions == 1)))
        fn = int(np.sum((class_targets == 1) & (class_predictions == 0)))

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

    mean_average_precision = float(
        np.mean(np.asarray(per_class_ap, dtype=np.float64))
    )

    micro_ap = float(
        average_precision_score(
            targets,
            probabilities,
            average="micro",
        )
    )

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
            "Evaluates a multi-label BioViL-T checkpoint on NPZ images. "
            "Supports linear probing, partial fine-tuning, "
            "and full fine-tuning. Without arguments, uses the paths "
            "defined at the beginning of the script."
        )
    )

    parser.add_argument(
        "--test-npz-dir",
        "--test-npy-dir",
        dest="test_npz_dir",
        type=Path,
        default=TEST_NPZ_DIR,
        help=(
            "Directory containing the test .npz files. "
            "The legacy --test-npy-dir alias remains available."
        ),
    )

    parser.add_argument(
        "--test-csv",
        type=Path,
        default=TEST_CSV,
        help=(
            "Test CSV containing image_id and labels."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
        help=(
            "BioViL-T .pt checkpoint to evaluate."
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

    parser.add_argument(
        "--npz-image-key",
        type=str,
        default=None,
        help=(
            "Image array key inside each NPZ file. "
            "If omitted, uses the checkpoint key or automatic "
            "detection."
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

    if args.npz_image_key is not None:
        args.npz_image_key = args.npz_image_key.strip()

        if not args.npz_image_key:
            parser.error(
                "--npz-image-key cannot be an empty string."
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
            "The test CSV file does not exist:\n"
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

    normalization = metadata[
        "data_normalization"
    ]

    if args.npz_image_key is not None:
        npz_image_key = args.npz_image_key
        npz_image_key_source = "command line"
    else:
        checkpoint_npz_image_key = metadata.get(
            "npz_image_key"
        )

        npz_image_key = (
            None
            if checkpoint_npz_image_key is None
            else str(checkpoint_npz_image_key).strip() or None
        )

        npz_image_key_source = (
            "checkpoint"
            if npz_image_key is not None
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
        f"Hugging Face model: {metadata['model_id']}"
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
        "Global embedding dimension: "
        f"{metadata['feature_size']}"
    )
    print(
        "Feature source: "
        f"{metadata['feature_source']}"
    )
    print(
        "Single-image mode: "
        f"{metadata['single_image_mode']}"
    )
    print(
        f"Resolution: {metadata['image_size']} x "
        f"{metadata['image_size']}"
    )
    print(
        f"NPZ normalization: {normalization}"
    )
    print(
        "NPZ image key: "
        f"{npz_image_key!r} (source: {npz_image_key_source})"
    )
    print(
        f"Threshold: {threshold} "
        f"(source: {threshold_source})"
    )
    print(
        f"Evaluation in float32: {EVALUATE_IN_FLOAT32}"
    )

    test_dataframe = load_test_csv(
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
        f"Number of images: {len(test_dataframe)}"
    )

    print(
        "\nPositive samples per class in the test set:"
    )

    for label in label_columns:
        print(
            f"  {label}: "
            f"{int(test_dataframe[label].sum())}"
        )

    test_dataset = BioViLTTestDataset(
        dataframe=test_dataframe,
        image_dir=args.test_npz_dir,
        label_columns=label_columns,
        transform=preprocess,
        normalization=normalization,
        npz_image_key=npz_image_key,
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
        drop_last=False,
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
        per_class_dataframe,
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
        "npz": {
            "image_key": npz_image_key,
            "image_key_source": npz_image_key_source,
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

    per_class_dataframe.to_csv(
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
        per_class_dataframe[
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
