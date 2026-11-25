from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, roc_auc_score
from transformers import AutoImageProcessor, AutoModel
from tqdm.auto import tqdm


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("/path/to/vindr-cxr")

TRAIN_CSV = BASE_DIR / "annotations/train.csv"
VAL_CSV = BASE_DIR / "annotations/val.csv"
TRAIN_NPZ_DIR = BASE_DIR / "train"
VAL_NPZ_DIR = BASE_DIR / "train"
OUTPUT_DIR = Path("./outputs/rad_dino/linear_probing")


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "microsoft/rad-dino"
RAD_DINO_IMAGE_SIZE = 224
RAD_DINO_MEAN = [0.5307, 0.5307, 0.5307]
RAD_DINO_STD = [0.2583, 0.2583, 0.2583]
LABEL_COLUMNS = [
    "Lung tumor",
    "Pneumonia",
    "Tuberculosis",
    "Other diseases",
    "No finding",
]
NUM_CLASSES = len(LABEL_COLUMNS)

BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_EPOCHS = 100

# Only the classifier is trained with this learning rate.
CLASSIFIER_LEARNING_RATE = 1e-3

# Decoupled AdamW weight decay.
WEIGHT_DECAY = 1e-4

# Fixed scheduler shared by all models in the benchmark.
SCHEDULER_MILESTONES = [15, 30, 45, 60, 75, 90]
SCHEDULER_GAMMA = 0.7

CLASSIFICATION_THRESHOLD = 0.5
SEED = 2025

# Preserve the normalization used in the original code.
DATA_NORMALIZATION = "minmax"

# None enables automatic image-array detection in each NPZ archive.
# Set, for example, "image" when archives contain multiple arrays.
NPZ_IMAGE_KEY: str | None = None

# AMP is not used.
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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
# STANDALONE NPZ DATASET
# ============================================================


def normalize_npz_image(
    image: np.ndarray,
    normalization: str,
    image_path: Path,
) -> np.ndarray:
    """
    Normalizes a single-channel NPZ image and returns float32 values in [0, 1].

    Supported modes:
        "minmax": per-image min-max normalization;
        "none": the image must already be in [0, 1].
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

    image_min = float(image.min())
    image_max = float(image.max())

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
        if image_min < 0.0 or image_max > 1.0:
            raise ValueError(
                "DATA_NORMALIZATION indicates that the image should already "
                "be within [0,1], but this condition is not met. "
                f"Path: {image_path} | min={image_min:.6f} | "
                f"max={image_max:.6f}"
            )

    else:
        raise ValueError(
            "Unsupported DATA_NORMALIZATION: "
            f"{normalization!r}. Use 'minmax' ou 'none'."
        )

    return np.ascontiguousarray(
        np.clip(image, 0.0, 1.0),
        dtype=np.float32,
    )


def load_npz_image_array(
    image_path: Path,
    image_key: str | None = None,
) -> np.ndarray:
    """
    Extracts the image from an NPZ archive without allowing pickle objects.

    When ``image_key`` is None, common keys are checked first; if the
    archive contains only one array, that key is selected automatically.
    """

    preferred_keys = (
        "image",
        "arr_0",
        "pixel_array",
        "pixels",
        "data",
    )

    try:
        with np.load(
            image_path,
            allow_pickle=False,
        ) as archive:
            available_keys = list(
                archive.files
            )

            if not available_keys:
                raise ValueError(
                    f"The NPZ archive contains no arrays:\n{image_path}"
                )

            if image_key is not None:
                selected_key = str(
                    image_key
                )

                if selected_key not in archive:
                    raise KeyError(
                        f"The NPZ key {selected_key!r} does not exist in "
                        f"{image_path}. Chaves disponíveis: "
                        f"{available_keys}"
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
                            "Could not automatically identify "
                            "the image in the NPZ archive. Set "
                            "NPZ_IMAGE_KEY to one of the available keys. "
                            f"Path: {image_path} | keys: {available_keys}"
                        )

            # Create an independent copy before closing the NPZ archive.
            image = np.array(
                archive[selected_key],
                copy=True,
            )

    except (OSError, ValueError, KeyError) as error:
        if isinstance(error, (ValueError, KeyError)):
            raise

        raise OSError(
            f"Could not open the NPZ archive:\n{image_path}"
        ) from error

    return image


def load_npz_as_rgb_pil(
    image_path: Path,
    normalization: str,
    image_key: str | None = None,
) -> Image.Image:
    """
    Loads a single-channel chest X-ray from an NPZ archive and converts it to PIL RGB.
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
        image_uint8
    ).convert("RGB")


def load_split_dataframe(
    csv_path: Path,
    label_columns: list[str],
) -> pd.DataFrame:
    """
    Reads and validates a CSV containing:
        image_id + one binary column per class.
    """

    if not csv_path.exists():
        raise FileNotFoundError(
            f"The CSV was not found:\n{csv_path}"
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
            "Missing required columns in CSV "
            f"{csv_path.name}: {missing_columns}"
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
            f"The CSV is empty:\n{csv_path}"
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
            "Duplicate image_id values were found in the CSV. "
            f"Examples: {duplicated_examples}"
        )

    for column in label_columns:
        numeric_values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        if numeric_values.isna().any():
            invalid_examples = (
                dataframe.loc[
                    numeric_values.isna(),
                    column,
                ]
                .head(5)
                .tolist()
            )

            raise ValueError(
                f"Column '{column}' contains non-numeric values. "
                f"Examples: {invalid_examples}"
            )

        invalid_mask = ~numeric_values.isin(
            [0, 1]
        )

        if invalid_mask.any():
            invalid_examples = (
                numeric_values.loc[
                    invalid_mask
                ]
                .head(5)
                .tolist()
            )

            raise ValueError(
                f"Column '{column}' contains values other than 0/1. "
                f"Examples: {invalid_examples}"
            )

        dataframe[column] = numeric_values.astype(
            np.float32
        )

    return dataframe


class RADDINONPZDataset(Dataset):
    """
    Standalone multi-label dataset for chest X-rays stored in NPZ files.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: Path,
        label_columns: list[str],
        transform,
        normalization: str,
        npz_image_key: str | None = None,
    ):
        self.dataframe = dataframe.reset_index(
            drop=True
        )
        self.image_dir = Path(image_dir)
        self.label_columns = list(
            label_columns
        )
        self.transform = transform
        self.normalization = normalization
        self.npz_image_key = npz_image_key

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

        image = load_npz_as_rgb_pil(
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
                "The RAD-DINO transform must return "
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

        return image, labels


def create_train_val_dataloaders(
    train_csv: Path,
    val_csv: Path,
    train_npz_dir: Path,
    val_npz_dir: Path,
    train_transform,
    val_transform,
    batch_size: int,
    num_workers: int,
    normalization: str,
    label_columns: list[str],
    npz_image_key: str | None = None,
) -> dict[str, DataLoader]:
    """
    Creates DataLoaders without relying on any external project module.
    """

    train_dataframe = load_split_dataframe(
        csv_path=train_csv,
        label_columns=label_columns,
    )

    validation_dataframe = load_split_dataframe(
        csv_path=val_csv,
        label_columns=label_columns,
    )

    train_dataset = RADDINONPZDataset(
        dataframe=train_dataframe,
        image_dir=train_npz_dir,
        label_columns=label_columns,
        transform=train_transform,
        normalization=normalization,
        npz_image_key=npz_image_key,
    )

    validation_dataset = RADDINONPZDataset(
        dataframe=validation_dataframe,
        image_dir=val_npz_dir,
        label_columns=label_columns,
        transform=val_transform,
        normalization=normalization,
        npz_image_key=npz_image_key,
    )

    common_loader_arguments = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "pin_memory": DEVICE.type == "cuda",
        "persistent_workers": int(num_workers) > 0,
        "drop_last": False,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **common_loader_arguments,
    )

    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **common_loader_arguments,
    )

    return {
        "train": train_loader,
        "val": validation_loader,
    }



# ============================================================
# RAD-DINO MODEL
# ============================================================


class RADDINOTransform:
    """
    Adapter between the dataset and the official AutoImageProcessor.

    The processor preserves official RAD-DINO normalization while enforcing
    resizing and center cropping to the benchmark resolution
    defined by ``RAD_DINO_IMAGE_SIZE``.
    """

    def __init__(
        self,
        image_processor,
        image_size: int,
    ):
        self.image_processor = image_processor
        self.image_size = int(image_size)

        if self.image_size < 1:
            raise ValueError(
                "image_size must be a positive integer."
            )

    def __call__(self, image) -> torch.Tensor:
        processed = self.image_processor(
            images=image,
            do_resize=True,
            size={"shortest_edge": self.image_size},
            do_center_crop=True,
            crop_size={
                "height": self.image_size,
                "width": self.image_size,
            },
            return_tensors="pt",
        )

        if "pixel_values" not in processed:
            raise KeyError(
                "The AutoImageProcessor did not return 'pixel_values'."
            )

        pixel_values = processed["pixel_values"]

        if pixel_values.ndim != 4 or pixel_values.shape[0] != 1:
            raise ValueError(
                "Expected pixel_values with shape [1,C,H,W], "
                f"but received {tuple(pixel_values.shape)}."
            )

        expected_spatial_shape = (
            self.image_size,
            self.image_size,
        )

        if tuple(pixel_values.shape[-2:]) != expected_spatial_shape:
            raise ValueError(
                "The AutoImageProcessor did not produce the resolution "
                f"{self.image_size} x {self.image_size}: "
                f"shape={tuple(pixel_values.shape)}."
            )

        return pixel_values.squeeze(0)


class RADDINOClassifier(nn.Module):
    """
    RAD-DINO encoder with a linear multi-label classifier.

    The classifier uses the normalized class token returned in
    pooler_output. If this field is unavailable, it uses the first token in
    last_hidden_state.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        if not hasattr(encoder, "config"):
            raise AttributeError(
                "The RAD-DINO encoder does not have the attribute 'config'."
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

        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
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

        features = getattr(
            outputs,
            "pooler_output",
            None,
        )

        if features is None:
            last_hidden_state = getattr(
                outputs,
                "last_hidden_state",
                None,
            )

            if last_hidden_state is None:
                raise TypeError(
                    "RAD-DINO returned neither 'pooler_output' nor "
                    "'last_hidden_state'."
                )

            features = last_hidden_state[:, 0, :]

        if features.ndim != 2:
            raise ValueError(
                "Expected a [batch, dimension] embedding, "
                f"but received {tuple(features.shape)}."
            )

        return self.classifier(features)


def load_rad_dino_classifier(
    model_id: str,
    num_classes: int,
    dropout: float,
):
    """
    Automatically loads the model and image processor from Hugging Face.

    An internet connection is required on the first run. On subsequent
    runs, files are reused from the local cache.
    """

    try:
        image_processor = AutoImageProcessor.from_pretrained(
            model_id,
            use_fast=False,
        )
    except TypeError:
        image_processor = AutoImageProcessor.from_pretrained(
            model_id,
        )
    except Exception as error:
        raise RuntimeError(
            "Could not load the AutoImageProcessor for "
            f"'{model_id}'."
        ) from error

    try:
        encoder = AutoModel.from_pretrained(
            model_id,
        )
    except Exception as error:
        raise RuntimeError(
            f"Could not load model '{model_id}'."
        ) from error

    model = RADDINOClassifier(
        encoder=encoder,
        num_classes=num_classes,
        dropout=dropout,
    )

    preprocess = RADDINOTransform(
        image_processor=image_processor,
        image_size=RAD_DINO_IMAGE_SIZE,
    )

    return model, preprocess, image_processor


# ============================================================
# TRAINING PROTOCOL
# ============================================================


def configure_linear_probing(
    model: nn.Module,
) -> None:
    """
    Linear probing:
        - freezes the entire RAD-DINO encoder;
        - trains only the multi-label classifier.
    """

    if not hasattr(model, "encoder"):
        raise AttributeError(
            "The model does not have the attribute 'encoder'."
        )

    if not hasattr(model, "classifier"):
        raise AttributeError(
            "The model does not have the attribute 'classifier'."
        )

    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


# ============================================================
# CLASS WEIGHTS
# ============================================================


def calculate_pos_weight(
    train_loader,
    device: torch.device,
) -> torch.Tensor:
    """Computes negative-to-positive ratios using only the training set."""

    dataset = train_loader.dataset

    labels = dataset.dataframe[
        dataset.label_columns
    ].to_numpy(dtype=np.float32)

    positive_counts = labels.sum(axis=0)
    negative_counts = len(labels) - positive_counts

    classes_without_positives = [
        dataset.label_columns[index]
        for index, count in enumerate(positive_counts)
        if count == 0
    ]

    if classes_without_positives:
        raise ValueError(
            "No positive training examples for: "
            + ", ".join(classes_without_positives)
        )

    weights = negative_counts / positive_counts

    print("\npos_weight:")
    for class_name, weight in zip(
        dataset.label_columns,
        weights,
    ):
        print(f"  {class_name}: {weight:.4f}")

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )


# ============================================================
# METRICS
# ============================================================


def calculate_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    predictions = (
        probabilities >= CLASSIFICATION_THRESHOLD
    ).astype(np.int32)

    targets = targets.astype(np.int32)

    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    micro_f1 = f1_score(
        targets,
        predictions,
        average="micro",
        zero_division=0,
    )

    element_accuracy = float(
        np.mean(predictions == targets)
    )

    per_class_auc = {}
    valid_auc_values = []

    for class_index, class_name in enumerate(LABEL_COLUMNS):
        class_targets = targets[:, class_index]
        class_probabilities = probabilities[:, class_index]

        # ROC-AUC requires at least one positive and one negative example.
        if np.unique(class_targets).size < 2:
            per_class_auc[class_name] = np.nan
            continue

        auc = roc_auc_score(
            class_targets,
            class_probabilities,
        )

        per_class_auc[class_name] = float(auc)
        valid_auc_values.append(float(auc))

    macro_auc = (
        float(np.mean(valid_auc_values))
        if valid_auc_values
        else np.nan
    )

    return {
        "macro_auc": macro_auc,
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "element_accuracy": element_accuracy,
        "per_class_auc": per_class_auc,
    }


# ============================================================
# TRAINING EPOCH
# ============================================================


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, dict]:
    model.train()

    # Keep the frozen encoder in evaluation mode.
    model.encoder.eval()
    model.classifier.train()

    running_loss = 0.0
    number_of_samples = 0
    all_targets = []
    all_probabilities = []

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False,
        dynamic_ncols=True,
    )

    for images, labels in progress_bar:
        images = images.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        probabilities = torch.sigmoid(logits)
        batch_size = images.size(0)

        running_loss += loss.item() * batch_size
        number_of_samples += batch_size

        all_targets.append(labels.detach().cpu().numpy())
        all_probabilities.append(
            probabilities.detach().cpu().numpy()
        )

        progress_bar.set_postfix(
            loss=f"{running_loss / number_of_samples:.4f}"
        )

    if number_of_samples == 0:
        raise RuntimeError(
            "The training DataLoader returned no samples."
        )

    targets = np.concatenate(all_targets, axis=0)
    probabilities = np.concatenate(all_probabilities, axis=0)

    metrics = calculate_metrics(
        targets=targets,
        probabilities=probabilities,
    )

    average_loss = running_loss / number_of_samples
    return average_loss, metrics


# ============================================================
# VALIDATION
# ============================================================


def validate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict]:
    model.eval()

    running_loss = 0.0
    number_of_samples = 0
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        progress_bar = tqdm(
            dataloader,
            desc="Validation",
            leave=False,
            dynamic_ncols=True,
        )

        for images, labels in progress_bar:
            images = images.to(
                device,
                non_blocking=True,
            )
            labels = labels.to(
                device,
                non_blocking=True,
            )

            logits = model(images)
            loss = criterion(logits, labels)
            probabilities = torch.sigmoid(logits)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            number_of_samples += batch_size

            all_targets.append(labels.cpu().numpy())
            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            progress_bar.set_postfix(
                loss=f"{running_loss / number_of_samples:.4f}"
            )

    if number_of_samples == 0:
        raise RuntimeError(
            "The validation DataLoader returned no samples."
        )

    targets = np.concatenate(all_targets, axis=0)
    probabilities = np.concatenate(all_probabilities, axis=0)

    metrics = calculate_metrics(
        targets=targets,
        probabilities=probabilities,
    )

    average_loss = running_loss / number_of_samples
    return average_loss, metrics


# ============================================================
# METRIC FORMATTING
# ============================================================


def format_metric(value: float) -> str:
    if np.isfinite(value):
        return f"{value:.4f}"
    return "N/A"


def print_metrics(
    name: str,
    loss: float,
    metrics: dict,
    show_per_class: bool = False,
) -> None:
    print(
        f"{name}: "
        f"loss={loss:.4f} | "
        f"macro AUC={format_metric(metrics['macro_auc'])} | "
        f"macro F1={metrics['macro_f1']:.4f} | "
        f"micro F1={metrics['micro_f1']:.4f} | "
        f"accuracy={metrics['element_accuracy']:.4f}"
    )

    if show_per_class:
        print("AUC per class:")
        for class_name, auc in metrics["per_class_auc"].items():
            print(f"  {class_name}: {format_metric(auc)}")


# ============================================================
# CHECKPOINT
# ============================================================


def build_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.MultiStepLR,
    train_loss: float,
    validation_loss: float,
    validation_metrics: dict,
    best_validation_macro_auc: float,
) -> dict:
    return {
        "epoch": epoch,
        "protocol": "linear_probing",
        "fine_tuning": "linear_probing",
        "model_name": MODEL_ID,
        "model_id": MODEL_ID,
        "model_source": "huggingface_transformers",
        "image_size": RAD_DINO_IMAGE_SIZE,
        "rad_dino_mean": RAD_DINO_MEAN,
        "rad_dino_std": RAD_DINO_STD,
        "image_processor": "AutoImageProcessor",
        "pretrained": True,
        "num_classes": NUM_CLASSES,
        "label_columns": LABEL_COLUMNS,
        "threshold": CLASSIFICATION_THRESHOLD,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "validation_metrics": validation_metrics,
        "best_validation_macro_auc": best_validation_macro_auc,
        "model_selection_metric": "validation_macro_auc",
        "classifier_learning_rate": CLASSIFIER_LEARNING_RATE,
        "current_classifier_learning_rate": (
            optimizer.param_groups[0]["lr"]
        ),
        "weight_decay": WEIGHT_DECAY,
        "scheduler": {
            "name": "MultiStepLR",
            "applied_to": "classifier",
            "milestones": SCHEDULER_MILESTONES,
            "gamma": SCHEDULER_GAMMA,
        },
        "trainable_layers": ["classifier"],
        "data_normalization": DATA_NORMALIZATION,
        "dataset_implementation": "standalone_npz",
        "npz_image_key": NPZ_IMAGE_KEY,
        "seed": SEED,
    }

# ============================================================
# MAIN
# ============================================================


def main() -> None:
    set_seed(SEED)

    for path, description in (
        (TRAIN_CSV, "training CSV"),
        (VAL_CSV, "validation CSV"),
        (TRAIN_NPZ_DIR, "training NPZ directory"),
        (VAL_NPZ_DIR, "validation NPZ directory"),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find {description}:\n{path}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_checkpoint_path = (
        OUTPUT_DIR / "best_rad_dino_linear_probing.pt"
    )
    last_checkpoint_path = (
        OUTPUT_DIR / "last_rad_dino_linear_probing.pt"
    )
    history_path = OUTPUT_DIR / "training_history.csv"
    validation_results_path = (
        OUTPUT_DIR / "validation_results.csv"
    )

    print("=" * 60)
    print("RAD-DINO — Linear probing")
    print("=" * 60)
    print(f"Dispositivo: {DEVICE}")
    print("AMP: not used")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # RAD-DINO model and official preprocessing
    # --------------------------------------------------------

    model, preprocess, image_processor = load_rad_dino_classifier(
        model_id=MODEL_ID,
        num_classes=NUM_CLASSES,
        dropout=0.0,
    )

    model = model.float()

    configure_linear_probing(model)
    model = model.to(DEVICE)

    classifier_parameters = [
        parameter
        for parameter in model.classifier.parameters()
        if parameter.requires_grad
    ]

    if not classifier_parameters:
        raise RuntimeError(
            "The classifier has no trainable parameters."
        )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen_parameters = total_parameters - trainable_parameters

    print(f"\nModel: {MODEL_ID}")
    print(f"Resolução oficial: {model.encoder.config.image_size}")
    print(f"Resolução do benchmark: {RAD_DINO_IMAGE_SIZE}")
    print("Training protocol: linear probing")
    print("Trainable layers: classifier")
    print(f"Classes: {NUM_CLASSES}")
    print(f"Batch size: {BATCH_SIZE}")
    print(
        "NPZ image key: "
        f"{NPZ_IMAGE_KEY if NPZ_IMAGE_KEY is not None else 'automatic'}"
    )
    print(
        "Initial classifier learning rate: "
        f"{CLASSIFIER_LEARNING_RATE}"
    )
    print(f"Weight decay: {WEIGHT_DECAY}")
    print("Scheduler: MultiStepLR for the classifier")
    print(f"Parâmetros totais: {total_parameters:,}")
    print(f"Parâmetros treináveis: {trainable_parameters:,}")
    print(f"Parâmetros congelados: {frozen_parameters:,}")

    # --------------------------------------------------------
    # Standalone dataset and DataLoaders
    # --------------------------------------------------------

    dataloaders = create_train_val_dataloaders(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        train_npz_dir=TRAIN_NPZ_DIR,
        val_npz_dir=VAL_NPZ_DIR,
        train_transform=preprocess,
        val_transform=preprocess,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        normalization=DATA_NORMALIZATION,
        label_columns=LABEL_COLUMNS,
        npz_image_key=NPZ_IMAGE_KEY,
    )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")

    # --------------------------------------------------------
    # Loss, optimizer, and scheduler
    # --------------------------------------------------------

    pos_weight = calculate_pos_weight(
        train_loader=train_loader,
        device=DEVICE,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
    )

    optimizer = torch.optim.AdamW(
        classifier_parameters,
        lr=CLASSIFIER_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=SCHEDULER_MILESTONES,
        gamma=SCHEDULER_GAMMA,
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    best_validation_macro_auc = float("-inf")
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        print(
            f"\n{'=' * 60}\n"
            f"Epoch {epoch}/{NUM_EPOCHS}\n"
            f"{'=' * 60}"
        )

        learning_rate_used = optimizer.param_groups[0]["lr"]

        train_loss, train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
        )

        validation_loss, validation_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=DEVICE,
        )

        elapsed_time = time.time() - start_time

        print_metrics(
            name="Train",
            loss=train_loss,
            metrics=train_metrics,
        )
        print_metrics(
            name="Validation",
            loss=validation_loss,
            metrics=validation_metrics,
        )
        print(f"Tempo da época: {elapsed_time:.1f} s")
        print(
            "Learning rate used: "
            f"{learning_rate_used:.8f}"
        )

        scheduler.step()
        next_learning_rate = optimizer.param_groups[0]["lr"]

        if next_learning_rate < learning_rate_used:
            print(
                "Learning rate reduced by scheduler: "
                f"{learning_rate_used:.8f} -> "
                f"{next_learning_rate:.8f}"
            )
        else:
            print(
                "Next learning rate: "
                f"{next_learning_rate:.8f}"
            )

        validation_macro_auc = float(
            validation_metrics["macro_auc"]
        )

        improved = (
            np.isfinite(validation_macro_auc)
            and validation_macro_auc
            > best_validation_macro_auc
        )

        if improved:
            best_validation_macro_auc = validation_macro_auc

        epoch_record = {
            "epoch": epoch,
            "elapsed_seconds": elapsed_time,
            "classifier_learning_rate_used": learning_rate_used,
            "classifier_learning_rate_next": next_learning_rate,
            "train_loss": train_loss,
            "train_macro_auc": train_metrics["macro_auc"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_micro_f1": train_metrics["micro_f1"],
            "train_element_accuracy": (
                train_metrics["element_accuracy"]
            ),
            "validation_loss": validation_loss,
            "validation_macro_auc": (
                validation_metrics["macro_auc"]
            ),
            "validation_macro_f1": (
                validation_metrics["macro_f1"]
            ),
            "validation_micro_f1": (
                validation_metrics["micro_f1"]
            ),
            "validation_element_accuracy": (
                validation_metrics["element_accuracy"]
            ),
            "best_validation_macro_auc": (
                best_validation_macro_auc
            ),
        }

        for class_name in LABEL_COLUMNS:
            epoch_record[
                f"train_auc_{class_name}"
            ] = train_metrics["per_class_auc"][class_name]

            epoch_record[
                f"validation_auc_{class_name}"
            ] = validation_metrics["per_class_auc"][class_name]

        history.append(epoch_record)

        pd.DataFrame(history).to_csv(
            history_path,
            index=False,
        )

        checkpoint = build_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loss=train_loss,
            validation_loss=validation_loss,
            validation_metrics=validation_metrics,
            best_validation_macro_auc=best_validation_macro_auc,
        )

        torch.save(checkpoint, last_checkpoint_path)
        print(
            "Last checkpoint saved to:\n"
            f"{last_checkpoint_path}"
        )

        if improved:
            torch.save(checkpoint, best_checkpoint_path)
            print(
                "New best checkpoint by macro-AUC saved to:\n"
                f"{best_checkpoint_path}"
            )
        else:
            print(
                "Best validation macro-AUC so far: "
                f"{format_metric(best_validation_macro_auc)}"
            )

    # --------------------------------------------------------
    # Final evaluation of the best checkpoint
    # --------------------------------------------------------

    if not best_checkpoint_path.exists():
        raise RuntimeError(
            "No best checkpoint was saved."
        )

    best_checkpoint = torch.load(
        best_checkpoint_path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        best_checkpoint["model_state_dict"]
    )

    final_validation_loss, final_validation_metrics = validate(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=DEVICE,
    )

    print(f"\nMelhor época: {best_checkpoint['epoch']}")
    print("Final results from the best checkpoint:")

    print_metrics(
        name="Validation",
        loss=final_validation_loss,
        metrics=final_validation_metrics,
        show_per_class=True,
    )

    validation_results = [{
        "split": "validation",
        "best_epoch": best_checkpoint["epoch"],
        "loss": final_validation_loss,
        "macro_auc": final_validation_metrics["macro_auc"],
        "macro_f1": final_validation_metrics["macro_f1"],
        "micro_f1": final_validation_metrics["micro_f1"],
        "element_accuracy": (
            final_validation_metrics["element_accuracy"]
        ),
        **{
            f"auc_{class_name}": auc
            for class_name, auc in
            final_validation_metrics["per_class_auc"].items()
        },
    }]

    pd.DataFrame(validation_results).to_csv(
        validation_results_path,
        index=False,
    )

    print("\nRAD-DINO linear probing training completed.")
    print(f"Training history saved to:\n{history_path}")
    print(
        "Validation results saved to:\n"
        f"{validation_results_path}"
    )
    print(f"Melhor checkpoint:\n{best_checkpoint_path}")
    print(f"Último checkpoint:\n{last_checkpoint_path}")


if __name__ == "__main__":
    main()
