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
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
import open_clip


# ============================================================
# PATH CONFIGURATION - UPDATE BEFORE RUNNING
# ============================================================

TEST_NPZ_DIR = Path("/path/to/vindr-cxr/test")
TEST_CSV = Path( "/path/to/annotations/test.csv")
CHECKPOINT_PATH = Path("/path/to/checkpoint.pt")
OUTPUT_DIR = Path("/path/to/results")

# ============================================================
# FALLBACK CONFIGURATION
# ============================================================
# These defaults are used only when the corresponding checkpoint metadata is missing.

DEFAULT_MODEL_NAME = "BiomedCLIP"
DEFAULT_MODEL_ID = (
    "microsoft/"
    "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)

DEFAULT_OPEN_CLIP_MODEL_NAME = (
    f"hf-hub:{DEFAULT_MODEL_ID}"
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
DEFAULT_PATCH_SIZE = 16
DEFAULT_DATA_NORMALIZATION = "minmax"
DEFAULT_NPZ_IMAGE_KEY: str | None = None
DEFAULT_CLASSIFICATION_THRESHOLD = 0.5

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



NPZ_PREFERRED_IMAGE_KEYS = (
    "image",
    "arr_0",
    "pixel_array",
    "pixels",
    "data",
)


def load_npz_image_array(
    image_path: Path,
    image_key: str | None = None,
) -> np.ndarray:
    """
    Extracts the image from an NPZ archive without allowing serialized objects.

    If image_key is not specified, common keys are checked first. If the
    archive contains only one array, its key is selected automatically.
    For archives with multiple arrays and no recognized key,
    NPZ_IMAGE_KEY must be explicitly defined.
    """

    try:
        with np.load(
            image_path,
            allow_pickle=False,
        ) as archive:
            available_keys = list(archive.files)

            if not available_keys:
                raise ValueError(
                    f"The NPZ archive is empty: {image_path}"
                )

            if image_key is not None:
                selected_key = str(image_key)

                if selected_key not in archive:
                    raise KeyError(
                        f"The NPZ key {selected_key!r} does not exist in "
                        f"{image_path}. Available keys: {available_keys}"
                    )

            else:
                selected_key = next(
                    (
                        key
                        for key in NPZ_PREFERRED_IMAGE_KEYS
                        if key in archive
                    ),
                    None,
                )

                if selected_key is None:
                    if len(available_keys) == 1:
                        selected_key = available_keys[0]
                    else:
                        raise ValueError(
                            "Could not automatically identify "
                            f"the image array in {image_path}. "
                            f"Available keys: {available_keys}. "
                            "Set NPZ_IMAGE_KEY to the correct key."
                        )

            # Copy the array so it remains valid after the NPZ archive is closed.
            return np.array(
                archive[selected_key],
                copy=True,
            )

    except (OSError, ValueError, KeyError) as error:
        if isinstance(error, (ValueError, KeyError)):
            raise

        raise OSError(
            f"Could not open the NPZ archive: {image_path}"
        ) from error


def normalize_npz_image(
    image: np.ndarray,
    normalization: str,
    image_path: Path,
) -> np.ndarray:
    """
    Normalizes a single-channel NPZ image and returns a float32 array in [0, 1].

    Accepted formats:
        [H, W]
        [1, H, W]
        [H, W, 1]

    Modos:
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
            "Expected an NPZ array with shape [H,W], [1,H,W] "
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
            f"{normalization!r}. Use 'minmax' ou 'none'."
        )

    return np.ascontiguousarray(
        np.clip(
            image,
            0.0,
            1.0,
        ),
        dtype=np.float32,
    )


def load_npz_as_rgb_pil(
    image_path: Path,
    normalization: str,
    image_key: str | None = None,
) -> Image.Image:
    """
    Loads a chest X-ray from an NPZ file and converts it to a PIL grayscale image.
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
    ).convert("RGB")


def build_biomedclip_preprocess(
    open_clip_model_name: str,
):
    """
    Loads the official BiomedCLIP validation preprocessing pipeline.

    OpenCLIP retrieves from Hugging Face:
        - 224 x 224 resolution;
        - resize e center crop;
        - tensor conversion;
        - official normalization.
    """

    try:
        (
            clip_model,
            _preprocess_train,
            preprocess_validation,
        ) = open_clip.create_model_and_transforms(
            open_clip_model_name,
            precision="fp32",
            device="cpu",
        )

    except Exception as error:
        raise RuntimeError(
            "Could not load the BiomedCLIP preprocessing pipeline. "
            "Ensure that open_clip_torch, timm, transformers e "
            "huggingface_hub are installed."
        ) from error

    del clip_model

    return preprocess_validation


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


class BiomedCLIPTestDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: Path,
        label_columns: list[str],
        transform,
        normalization: str,
        image_key: str | None = None,
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
        self.image_key = image_key

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

        image = load_npz_as_rgb_pil(
            image_path=image_path,
            normalization=self.normalization,
            image_key=self.image_key,
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
                "The BiomedCLIP transform must return "
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


def infer_visual_output_dim(
    visual_encoder: nn.Module,
) -> int:
    """
    Determines the projected visual embedding dimension.
    """

    head = getattr(
        visual_encoder,
        "head",
        None,
    )

    if head is not None:
        linear_layers = [
            module
            for module in head.modules()
            if isinstance(module, nn.Linear)
        ]

        if linear_layers:
            return int(
                linear_layers[-1].out_features
            )

    output_dim = getattr(
        visual_encoder,
        "output_dim",
        None,
    )

    if output_dim is not None:
        return int(
            output_dim
        )

    raise AttributeError(
        "Could not determine the output dimension "
        "of the BiomedCLIP visual encoder."
    )


class BiomedCLIPClassifier(nn.Module):
    """
    BiomedCLIP visual encoder with a multi-label classifier.

    The classifier uses the projected visual embedding.
    The text encoder is not used during evaluation.
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
            images
        )

        if isinstance(
            features,
            dict,
        ):
            if "pooled" not in features:
                raise KeyError(
                    "The visual encoder returned a dictionary "
                    "without the 'pooled' key."
                )

            features = features[
                "pooled"
            ]

        elif isinstance(
            features,
            (tuple, list),
        ):
            if not features:
                raise ValueError(
                    "The visual encoder returned an empty sequence."
                )

            features = features[0]

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
                "The BiomedCLIP embedding dimension is incompatible "
                "with the classifier: "
                f"{features.shape[1]} vs {self.feature_size}."
            )

        return self.classifier(
            features
        )


def load_biomedclip_visual_encoder(
    open_clip_model_name: str,
):
    """
    Loads the official BiomedCLIP through OpenCLIP and retains only
    the visual encoder.

    The pretrained weights are used to reconstruct the architecture. Then,
    the training checkpoint model_state_dict replaces the full state
    of the visual encoder and classifier.
    """

    try:
        (
            clip_model,
            _preprocess_train,
            preprocess_validation,
        ) = open_clip.create_model_and_transforms(
            open_clip_model_name,
            precision="fp32",
            device="cpu",
        )

    except Exception as error:
        raise RuntimeError(
            "Could not load BiomedCLIP. Ensure "
            "que open_clip_torch, timm, transformers e "
            "huggingface_hub are installed."
        ) from error

    if not hasattr(
        clip_model,
        "visual",
    ):
        raise AttributeError(
            "The OpenCLIP model does not contain a 'visual' encoder."
        )

    visual_encoder = clip_model.visual
    feature_size = infer_visual_output_dim(
        visual_encoder
    )

    model_information = {
        "feature_size": feature_size,
        "visual_class": (
            visual_encoder.__class__.__name__
        ),
        "trunk_class": (
            visual_encoder.trunk.__class__.__name__
            if hasattr(
                visual_encoder,
                "trunk",
            )
            else None
        ),
    }

    del clip_model

    return (
        visual_encoder,
        preprocess_validation,
        model_information,
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
            "The checkpoint contains a num_classes value incompatible "
            "with label_columns: "
            f"{checkpoint_num_classes} vs {num_classes}."
        )

    model_id = str(
        checkpoint.get(
            "model_id",
            DEFAULT_MODEL_ID,
        )
    )

    open_clip_model_name = str(
        checkpoint.get(
            "open_clip_model_name",
            f"hf-hub:{model_id}",
        )
    )

    (
        visual_encoder,
        preprocess,
        model_information,
    ) = load_biomedclip_visual_encoder(
        open_clip_model_name=(
            open_clip_model_name
        ),
    )

    detected_feature_size = int(
        model_information[
            "feature_size"
        ]
    )

    feature_size = int(
        checkpoint.get(
            "feature_size",
            detected_feature_size,
        )
    )

    if (
        detected_feature_size
        != feature_size
    ):
        raise ValueError(
            "The loaded visual encoder dimension does not match "
            "the checkpoint: "
            f"{detected_feature_size} vs {feature_size}."
        )

    model = BiomedCLIPClassifier(
        encoder=visual_encoder,
        feature_size=feature_size,
        num_classes=num_classes,
        dropout=0.0,
    )

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

    image_size = int(
        checkpoint.get(
            "image_size",
            DEFAULT_IMAGE_SIZE,
        )
    )

    metadata = {
        "model_name": checkpoint.get(
            "model_name",
            DEFAULT_MODEL_NAME,
        ),
        "model_id": model_id,
        "open_clip_model_name": (
            open_clip_model_name
        ),
        "model_source": checkpoint.get(
            "model_source",
            "huggingface_open_clip",
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
        "image_size": image_size,
        "patch_size": int(
            checkpoint.get(
                "patch_size",
                DEFAULT_PATCH_SIZE,
            )
        ),
        "feature_size": feature_size,
        "feature_source": checkpoint.get(
            "feature_source",
            "projected_image_embedding",
        ),
        "visual_tower_only": checkpoint.get(
            "visual_tower_only",
            True,
        ),
        "text_tower_included": checkpoint.get(
            "text_tower_included",
            False,
        ),
        "visual_class": model_information.get(
            "visual_class"
        ),
        "trunk_class": model_information.get(
            "trunk_class"
        ),
        "biomedclip_mean": checkpoint.get(
            "biomedclip_mean"
        ),
        "biomedclip_std": checkpoint.get(
            "biomedclip_std"
        ),
        "data_normalization": normalization,
        "input_format": checkpoint.get(
            "input_format",
            "npz",
        ),
        "npz_image_key": checkpoint.get(
            "npz_image_key",
            DEFAULT_NPZ_IMAGE_KEY,
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


def calculate_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    label_columns: list[str],
    threshold: float,
):
    """Computes the metrics used in the multi-label benchmark.

    Threshold-independent metrics:
        - mAP: unweighted mean of per-class AP;
        - Micro-AP;
        - Macro-AUROC;
        - AP por classe.

    Threshold-dependent metrics:
        - Macro-Sensitivity;
        - Macro-Specificity;
        - Macro-F1;
        - Micro-F1.
    """

    targets = np.asarray(targets, dtype=np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float64)

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
            "The number of target columns does not match label_columns: "
            f"{targets.shape[1]} vs {len(label_columns)}."
        )

    if not np.all(np.isin(targets, [0, 1])):
        raise ValueError("Targets must contain only binary values (0/1).")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Probabilities contain NaN or infinite values.")

    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("Probabilities must be in [0,1].")

    if not 0.0 < threshold < 1.0:
        raise ValueError("The threshold must be strictly between 0 and 1.")

    positive_support = targets.sum(axis=0).astype(int)
    negative_support = (targets.shape[0] - positive_support).astype(int)

    # To ensure comparable macro metrics, every class must have
    # at least one positive and one negative example in the test set.
    invalid_classes = [
        label_columns[i]
        for i in range(len(label_columns))
        if positive_support[i] == 0 or negative_support[i] == 0
    ]

    if invalid_classes:
        raise ValueError(
            "Cannot compute all metrics consistently "
            "because some test classes lack positive or negative samples: "
            f"{invalid_classes}"
        )

    predictions = (probabilities >= threshold).astype(np.int32)

    # --------------------------------------------------------
    # Threshold-independent metrics
    # --------------------------------------------------------

    # Per-class AP; mAP is the unweighted arithmetic mean of class APs.
    per_class_ap = np.asarray(
        average_precision_score(
            targets,
            probabilities,
            average=None,
        ),
        dtype=np.float64,
    )
    mean_average_precision = float(np.mean(per_class_ap))

    # Micro-AP pools all image-label pairs.
    micro_ap = float(
        average_precision_score(
            targets,
            probabilities,
            average="micro",
        )
    )

    # Per-class AUROC and its unweighted macro average.
    per_class_auroc = np.asarray(
        [
            roc_auc_score(targets[:, class_index], probabilities[:, class_index])
            for class_index in range(targets.shape[1])
        ],
        dtype=np.float64,
    )
    macro_auroc = float(np.mean(per_class_auroc))

    # --------------------------------------------------------
    # Threshold-dependent metrics
    # --------------------------------------------------------

    macro_sensitivity = float(
        recall_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        )
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

    # Specificity = TN / (TN + FP), computed for each class.
    per_class_rows = []
    per_class_specificity = []

    for class_index, class_name in enumerate(label_columns):
        class_targets = targets[:, class_index]
        class_predictions = predictions[:, class_index]

        tp = int(np.sum((class_targets == 1) & (class_predictions == 1)))
        tn = int(np.sum((class_targets == 0) & (class_predictions == 0)))
        fp = int(np.sum((class_targets == 0) & (class_predictions == 1)))
        fn = int(np.sum((class_targets == 1) & (class_predictions == 0)))

        sensitivity = float(tp / (tp + fn))
        specificity = float(tn / (tn + fp))
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
            "threshold": float(threshold),
            "support_total": int(targets.shape[0]),
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

    macro_specificity = float(np.mean(per_class_specificity))
    per_class_df = pd.DataFrame(per_class_rows)

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

    return (
        summary,
        per_class_df,
        predictions,
    )


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


def print_summary(
    summary: dict,
) -> None:
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
        elif isinstance(value, (float, np.floating)):
            numeric_value = float(value)
            formatted = (
                "N/A"
                if not math.isfinite(numeric_value)
                else f"{numeric_value:.6f}"
            )
        else:
            formatted = str(value)

        print(f"{key}: {formatted}")


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluates a multi-label BiomedCLIP checkpoint on NPZ images. "
            "Supports linear probing, partial fine-tuning, "
            "and full fine-tuning. Without arguments, uses the paths "
            "defined at the beginning of the script."
        )
    )

    parser.add_argument(
        "--test-npz-dir",
        type=Path,
        default=TEST_NPZ_DIR,
        help=(
            "Directory containing the test .npz files."
        ),
    )

    parser.add_argument(
        "--npz-image-key",
        type=str,
        default=None,
        help=(
            "Image array key inside the NPZ file. If omitted, "
            "uses the checkpoint key or automatic detection."
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
            "BiomedCLIP .pt checkpoint to evaluate."
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
        default=None,
        help=(
            "Fixed threshold for threshold-dependent metrics. "
            "Defaults to 0.5 if omitted."
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
        image_key = args.npz_image_key
        image_key_source = "command line"
    else:
        image_key = metadata.get(
            "npz_image_key"
        )
        image_key_source = (
            "checkpoint"
            if image_key is not None
            else "automatic detection"
        )

    if args.threshold is None:
        threshold = float(DEFAULT_CLASSIFICATION_THRESHOLD)
        threshold_source = "fixed (default)"

    else:
        threshold = float(args.threshold)
        threshold_source = "command line"

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
        "OpenCLIP name: "
        f"{metadata['open_clip_model_name']}"
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
        "Visual architecture: "
        f"{metadata['trunk_class']}"
    )
    print(
        "Projected visual embedding dimension: "
        f"{metadata['feature_size']}"
    )
    print(
        "Feature source: "
        f"{metadata['feature_source']}"
    )
    print(
        "Visual encoder only: "
        f"{metadata['visual_tower_only']}"
    )
    print(
        "Text encoder included: "
        f"{metadata['text_tower_included']}"
    )
    print(
        f"Resolution: {metadata['image_size']} x "
        f"{metadata['image_size']}"
    )
    print(
        f"Patch size: {metadata['patch_size']} x "
        f"{metadata['patch_size']}"
    )
    print(
        f"NPZ normalization: {normalization}"
    )
    print(
        "NPZ image key: "
        f"{image_key!r} (source: {image_key_source})"
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

    test_dataset = BiomedCLIPTestDataset(
        dataframe=test_dataframe,
        image_dir=args.test_npz_dir,
        label_columns=label_columns,
        transform=preprocess,
        normalization=normalization,
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
        "npz_image_key": image_key,
        "npz_image_key_source": image_key_source,
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
        "prevalence",
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

    print("\nResults saved to:")
    print(f"  JSON summary: {summary_json_path}")
    print(f"  CSV summary: {summary_csv_path}")
    print(
        "  Per-class metrics: "
        f"{per_class_path}"
    )
    print(f"  Predictions: {predictions_path}")
    print("\nNote: threshold-dependent metrics use threshold=0.5 "
          "by default. Do not select the threshold based on the test set.")


if __name__ == "__main__":
    main()
