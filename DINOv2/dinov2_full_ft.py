from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

# ============================================================
# PATH CONFIGURATION — UPDATE BEFORE RUNNING
# ============================================================

# The model code and weights are obtained directly through PyTorch Hub.
# An internet connection is required on the first run. Subsequently, PyTorch
# automatically reuses the files stored in its local cache.
DINOV2_HUB_REPOSITORY = "facebookresearch/dinov2"

# Prevents an interactive confirmation prompt in recent PyTorch versions.
# Set to True only for repositories you trust.
TORCH_HUB_TRUST_REPO = True

# False reuses the cache. Set to True only to force a fresh download.
TORCH_HUB_FORCE_RELOAD = False

BASE_DIR = Path("/path/to/vindr-cxr")
TRAIN_CSV = BASE_DIR / "annotations/train.csv"
VAL_CSV = BASE_DIR / "annotations/val.csv"
TRAIN_NPZ_DIR = BASE_DIR / "train"
VAL_NPZ_DIR = BASE_DIR / "train"
OUTPUT_DIR = Path("./outputs/dinov2/full_ft")

# ============================================================
# CONFIGURATION
# ============================================================

# Common official alternatives:
#   dinov2_vits14
#   dinov2_vitb14
#   dinov2_vitl14
#   dinov2_vitg14
MODEL_NAME = "dinov2_vitb14"

LABEL_COLUMNS = [
    "Lung tumor",
    "Pneumonia",
    "Tuberculosis",
    "Other diseases",
    "No finding",
]

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_EPOCHS = 100

ENCODER_LEARNING_RATE = 1e-5
CLASSIFIER_LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-4

# Fixed learning-rate schedule shared by all benchmark models.
SCHEDULER_MILESTONES = [15, 30, 45, 60, 75, 90]
SCHEDULER_GAMMA = 0.7

CLASSIFICATION_THRESHOLD = 0.5
SEED = 2025

# Normalization applied before ImageNet normalization.
APPLY_MIN_MAX_NORMALIZATION = True

# Image array key inside each NPZ file.
# None enables automatic detection. For archives containing multiple arrays,
# explicitly specify the correct key, for example: "image".
NPZ_IMAGE_KEY: str | None = None

# DINOv2 was pretrained on RGB images.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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
# NPZ LOADING
# ============================================================


def load_npz_image_array(
        image_path: Path,
        image_key: str | None = None,
) -> np.ndarray:
    """
    Reads the image array from an NPZ file.

    If image_key is None, common keys are tried first. If the archive
    contains only one array, that key is selected automatically.
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
                        f"{image_path}. Available keys: "
                        f"{available_keys}."
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
                            "Set NPZ_IMAGE_KEY to the correct key."
                        )

            # Copying keeps the array valid after closing the NPZ archive.
            return np.array(
                archive[selected_key],
                copy=True,
            )

    except (OSError, ValueError, KeyError) as error:
        if isinstance(error, (ValueError, KeyError)):
            raise

        raise OSError(
            f"Could not open the NPZ file:\n{image_path}"
        ) from error


def read_npz(
        image_path: Path,
        image_key: str | None = None,
) -> torch.Tensor:
    """
    Reads an NPZ file and returns a float32 tensor [1, H, W].

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
                "The NPZ array has three dimensions but does not "
                "represent a single-channel image. "
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

    image_min = float(image.min())
    image_max = float(image.max())

    if APPLY_MIN_MAX_NORMALIZATION:
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
            "APPLY_MIN_MAX_NORMALIZATION=False, but the image "
            "is not within the [0,1] range. "
            f"Path: {image_path} | min={image_min:.6f} | "
            f"max={image_max:.6f}"
        )

    image = np.ascontiguousarray(
        image,
        dtype=np.float32,
    )

    return torch.from_numpy(image).unsqueeze(0)


# ============================================================
# ASPECT-RATIO-PRESERVING RESIZING
# ============================================================


class ResizeAndPad:
    """Preserves the aspect ratio and pads to output_size x output_size."""

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


class RepeatGrayscaleToRGB:
    """Converts [1,H,W] to [3,H,W] by repeating the channel."""

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 3 or image.shape[0] != 1:
            raise ValueError(
                "RepeatGrayscaleToRGB expected a [1,H,W] tensor, "
                f"but received {tuple(image.shape)}."
            )

        return image.repeat(3, 1, 1)


# ============================================================
# TRANSFORMS
# ============================================================


train_transform = transforms.Compose([
    ResizeAndPad(IMAGE_SIZE),
    RepeatGrayscaleToRGB(),
    transforms.RandomRotation(degrees=5, fill=0.0),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])

validation_transform = transforms.Compose([
    ResizeAndPad(IMAGE_SIZE),
    RepeatGrayscaleToRGB(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ),
])


# ============================================================
# DATASET
# ============================================================


class CXRDataset(Dataset):
    def __init__(
            self,
            dataframe: pd.DataFrame,
            image_dir: Path,
            transform=None,
            image_key: str | None = None,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
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
            image_key=self.image_key,
        )

        if self.transform is not None:
            image = self.transform(image)

        labels = torch.tensor(
            row[LABEL_COLUMNS].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

        return image, labels


# ============================================================
# LOAD DATA SPLITS
# ============================================================


def load_split_csv(
        csv_path: Path,
        split_name: str,
) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"The {split_name} CSV does not exist:\n{csv_path}"
        )

    dataframe = pd.read_csv(csv_path, sep=",")
    required_columns = ["image_id"] + LABEL_COLUMNS

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing from the {split_name} CSV: "
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
        raise ValueError(
            f"The {split_name} CSV is empty."
        )

    return dataframe


# ============================================================
# MODEL
# ============================================================


def load_dinov2_encoder() -> nn.Module:
    """
    Loads the pretrained DINOv2 backbone directly from PyTorch Hub.

    On the first run, PyTorch Hub downloads the official repository code
    and pretrained weights. Subsequent runs reuse the local PyTorch cache.

    There is no need to clone the DINOv2 repository or manually specify
    a weights file.
    """

    try:
        encoder = torch.hub.load(
            repo_or_dir=DINOV2_HUB_REPOSITORY,
            model=MODEL_NAME,
            source="github",
            pretrained=True,
            trust_repo=TORCH_HUB_TRUST_REPO,
            force_reload=TORCH_HUB_FORCE_RELOAD,
        )
    except TypeError:
        # Compatibility with older PyTorch versions that do not accept
        # the trust_repo argument.
        encoder = torch.hub.load(
            repo_or_dir=DINOV2_HUB_REPOSITORY,
            model=MODEL_NAME,
            source="github",
            pretrained=True,
            force_reload=TORCH_HUB_FORCE_RELOAD,
        )
    except Exception as error:
        raise RuntimeError(
            "Could not load DINOv2 through PyTorch Hub. "
            "An internet connection is required on the first run. "
            "Also check access to GitHub and the weights URL."
        ) from error

    if not hasattr(encoder, "embed_dim"):
        raise AttributeError(
            "The loaded backbone does not have the 'embed_dim' attribute."
        )

    return encoder


class DINOv2Classifier(nn.Module):
    def __init__(
            self,
            encoder: nn.Module,
            num_classes: int,
            dropout: float = 0.0,
    ):
        super().__init__()

        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(
                int(encoder.embed_dim),
                num_classes,
            ),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)

        if not isinstance(features, torch.Tensor):
            raise TypeError(
                "The DINOv2 encoder forward pass should return a tensor, "
                f"but returned {type(features).__name__}."
            )

        if features.ndim != 2:
            raise ValueError(
                "Expected an embedding [batch, dimension], but received "
                f"received shape={tuple(features.shape)}."
            )

        return self.classifier(features)


# ============================================================
# PROTOCOL — FULL FINE-TUNING
# ============================================================


def configure_protocol(model: DINOv2Classifier) -> list[str]:
    """
    Full fine-tuning:
        - initially freezes the entire wrapper;
        - makes the entire DINOv2 encoder trainable;
        - makes the multi-label classifier trainable.
    """

    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.encoder.parameters():
        parameter.requires_grad = True

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    return [
        "encoder (full DINOv2)",
        "classifier",
    ]


def configure_training_mode(model: DINOv2Classifier) -> None:
    model.train()
    model.encoder.train()
    model.classifier.train()


# ============================================================
# CLASS WEIGHTS
# ============================================================


def calculate_pos_weight(train_df: pd.DataFrame) -> torch.Tensor:
    positive = torch.tensor(
        train_df[LABEL_COLUMNS].sum(axis=0).to_numpy(),
        dtype=torch.float32,
    )

    negative = len(train_df) - positive

    if torch.any(positive == 0):
        missing = [
            LABEL_COLUMNS[index]
            for index, value in enumerate(positive.tolist())
            if value == 0
        ]
        raise ValueError(
            "No positive training samples for: "
            + ", ".join(missing)
        )

    return negative / positive


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
# TRAINING AND VALIDATION
# ============================================================


def train_one_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        encoder_optimizer: torch.optim.Optimizer,
        classifier_optimizer: torch.optim.Optimizer,
        device: torch.device,
) -> tuple[float, dict]:
    configure_training_mode(model)

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

        encoder_optimizer.zero_grad(set_to_none=True)
        classifier_optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()

        encoder_optimizer.step()
        classifier_optimizer.step()

        probabilities = torch.sigmoid(logits)
        current_batch_size = images.size(0)

        running_loss += loss.item() * current_batch_size
        number_of_samples += current_batch_size

        all_targets.append(
            labels.detach().cpu().numpy()
        )
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

    return running_loss / number_of_samples, metrics


@torch.inference_mode()
def validate(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
) -> tuple[float, dict]:
    model.eval()

    running_loss = 0.0
    number_of_samples = 0
    all_targets = []
    all_probabilities = []

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

        current_batch_size = images.size(0)
        running_loss += loss.item() * current_batch_size
        number_of_samples += current_batch_size

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

    return running_loss / number_of_samples, metrics


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
        print("Per-class AUC:")

        for class_name, auc in metrics["per_class_auc"].items():
            print(f"  {class_name}: {format_metric(auc)}")


# ============================================================
# CHECKPOINT
# ============================================================


def build_checkpoint(
        epoch: int,
        model: nn.Module,
        encoder_optimizer: torch.optim.Optimizer,
        classifier_optimizer: torch.optim.Optimizer,
        encoder_scheduler: torch.optim.lr_scheduler.MultiStepLR,
        classifier_scheduler: torch.optim.lr_scheduler.MultiStepLR,
        train_loss: float,
        validation_loss: float,
        validation_metrics: dict,
        best_validation_macro_auc: float,
        trainable_layers: list[str],
) -> dict:
    return {
        "epoch": epoch,
        "protocol": "full_fine_tuning",
        "fine_tuning": "full_fine_tuning",
        "model_name": MODEL_NAME,
        "model_source": "github_torch_hub",
        "dinov2_hub_repository": DINOV2_HUB_REPOSITORY,
        "pretrained": True,
        "torch_hub_force_reload": TORCH_HUB_FORCE_RELOAD,
        "num_classes": len(LABEL_COLUMNS),
        "label_columns": LABEL_COLUMNS,
        "image_size": IMAGE_SIZE,
        "in_chans": 3,
        "input_format": "npz",
        "npz_image_key": NPZ_IMAGE_KEY,
        "dataset_implementation": "standalone_npz",
        "apply_min_max_normalization": (
            APPLY_MIN_MAX_NORMALIZATION
        ),
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "threshold": CLASSIFICATION_THRESHOLD,
        "model_state_dict": model.state_dict(),
        "encoder_optimizer_state_dict": (
            encoder_optimizer.state_dict()
        ),
        "classifier_optimizer_state_dict": (
            classifier_optimizer.state_dict()
        ),
        "encoder_scheduler_state_dict": (
            encoder_scheduler.state_dict()
        ),
        "classifier_scheduler_state_dict": (
            classifier_scheduler.state_dict()
        ),
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "validation_metrics": validation_metrics,
        "best_validation_macro_auc": best_validation_macro_auc,
        "model_selection_metric": "validation_macro_auc",
        "encoder_learning_rate": ENCODER_LEARNING_RATE,
        "classifier_learning_rate": CLASSIFIER_LEARNING_RATE,
        "current_encoder_learning_rate": (
            encoder_optimizer.param_groups[0]["lr"]
        ),
        "current_classifier_learning_rate": (
            classifier_optimizer.param_groups[0]["lr"]
        ),
        "weight_decay": WEIGHT_DECAY,
        "scheduler": {
            "name": "MultiStepLR",
            "applied_to": ["encoder", "classifier"],
            "milestones": SCHEDULER_MILESTONES,
            "gamma": SCHEDULER_GAMMA,
        },
        "trainable_layers": trainable_layers,

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
                f"The {description} was not found:\n{path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("DINOv2 — Full fine-tuning")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print("AMP: not used")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_df = load_split_csv(
        TRAIN_CSV,
        "training",
    )
    validation_df = load_split_csv(
        VAL_CSV,
        "validation",
    )

    train_ids = set(train_df["image_id"])
    validation_ids = set(validation_df["image_id"])
    overlapping_ids = train_ids.intersection(validation_ids)

    if overlapping_ids:
        examples = sorted(overlapping_ids)[:5]
        raise ValueError(
            "Duplicate image_id values were found between training and "
            f"validation. Examples: {examples}"
        )

    train_dataset = CXRDataset(
        dataframe=train_df,
        image_dir=TRAIN_NPZ_DIR,
        transform=train_transform,
        image_key=NPZ_IMAGE_KEY,
    )

    validation_dataset = CXRDataset(
        dataframe=validation_df,
        image_dir=VAL_NPZ_DIR,
        transform=validation_transform,
        image_key=NPZ_IMAGE_KEY,
    )

    loader_arguments = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE.type == "cuda",
        "persistent_workers": NUM_WORKERS > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_arguments,
    )

    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_arguments,
    )

    encoder = load_dinov2_encoder()
    model = DINOv2Classifier(
        encoder=encoder,
        num_classes=len(LABEL_COLUMNS),
        dropout=0.0,
    )

    trainable_layers = configure_protocol(model)
    model = model.to(DEVICE)

    encoder_parameters = [
        parameter
        for parameter in model.encoder.parameters()
        if parameter.requires_grad
    ]

    classifier_parameters = [
        parameter
        for parameter in model.classifier.parameters()
        if parameter.requires_grad
    ]

    if not encoder_parameters:
        raise RuntimeError(
            "The encoder has no trainable parameters."
        )

    if not classifier_parameters:
        raise RuntimeError(
            "The classifier has no trainable parameters."
        )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen_parameter_count = (
            total_parameters - trainable_parameter_count
    )

    print(f"\nModel: {MODEL_NAME}")
    print(f"PyTorch Hub: {DINOV2_HUB_REPOSITORY}")
    print("Pretrained weights: automatic download/cache")
    print("Training protocol: full fine-tuning")
    print("Trainable layers:")

    for layer_name in trainable_layers:
        print(f"  {layer_name}")

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(validation_dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(
        "Encoder learning rate: "
        f"{ENCODER_LEARNING_RATE}"
    )
    print(
        "Initial classifier learning rate: "
        f"{CLASSIFIER_LEARNING_RATE}"
    )
    print(f"Weight decay: {WEIGHT_DECAY}")
    print("Scheduler: MultiStepLR for encoder and classifier")
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameter_count:,}")
    print(f"Frozen parameters: {frozen_parameter_count:,}")

    pos_weight = calculate_pos_weight(train_df).to(DEVICE)

    print("\npos_weight:")
    for label, weight in zip(
            LABEL_COLUMNS,
            pos_weight.cpu().tolist(),
    ):
        print(f"  {label}: {weight:.4f}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
    )

    encoder_optimizer = torch.optim.AdamW(
        encoder_parameters,
        lr=ENCODER_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    classifier_optimizer = torch.optim.AdamW(
        classifier_parameters,
        lr=CLASSIFIER_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    encoder_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        encoder_optimizer,
        milestones=SCHEDULER_MILESTONES,
        gamma=SCHEDULER_GAMMA,
    )

    classifier_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        classifier_optimizer,
        milestones=SCHEDULER_MILESTONES,
        gamma=SCHEDULER_GAMMA,
    )

    best_checkpoint_path = (
            OUTPUT_DIR / "best_dinov2_vitb14_full_fine_tuning.pt"
    )
    last_checkpoint_path = (
            OUTPUT_DIR / "last_dinov2_vitb14_full_fine_tuning.pt"
    )
    history_path = OUTPUT_DIR / "training_history.csv"
    validation_results_path = (
            OUTPUT_DIR / "validation_results.csv"
    )

    best_validation_macro_auc = float("-inf")
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        print(
            f"\n{'=' * 70}\n"
            f"Epoch {epoch}/{NUM_EPOCHS}\n"
            f"{'=' * 70}"
        )

        encoder_lr_used = (
            encoder_optimizer.param_groups[0]["lr"]
        )
        classifier_lr_used = (
            classifier_optimizer.param_groups[0]["lr"]
        )

        train_loss, train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            encoder_optimizer=encoder_optimizer,
            classifier_optimizer=classifier_optimizer,
            device=DEVICE,
        )

        validation_loss, validation_metrics = validate(
            model=model,
            dataloader=validation_loader,
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
        print(f"Epoch time: {elapsed_time:.1f} s")
        print(
            "Encoder learning rate used: "
            f"{encoder_lr_used:.8f}"
        )
        print(
            "Classifier learning rate used: "
            f"{classifier_lr_used:.8f}"
        )

        encoder_scheduler.step()
        classifier_scheduler.step()

        next_encoder_lr = (
            encoder_optimizer.param_groups[0]["lr"]
        )
        next_classifier_lr = (
            classifier_optimizer.param_groups[0]["lr"]
        )

        if next_encoder_lr < encoder_lr_used:
            print(
                "Encoder learning rate reduced by scheduler: "
                f"{encoder_lr_used:.8f} -> "
                f"{next_encoder_lr:.8f}"
            )
        else:
            print(
                "Next encoder learning rate: "
                f"{next_encoder_lr:.8f}"
            )

        if next_classifier_lr < classifier_lr_used:
            print(
                "Classifier learning rate reduced by scheduler: "
                f"{classifier_lr_used:.8f} -> "
                f"{next_classifier_lr:.8f}"
            )
        else:
            print(
                "Next classifier learning rate: "
                f"{next_classifier_lr:.8f}"
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
            "encoder_learning_rate_used": encoder_lr_used,
            "encoder_learning_rate_next": next_encoder_lr,
            "classifier_learning_rate_used": classifier_lr_used,
            "classifier_learning_rate_next": next_classifier_lr,

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
            encoder_optimizer=encoder_optimizer,
            classifier_optimizer=classifier_optimizer,
            encoder_scheduler=encoder_scheduler,
            classifier_scheduler=classifier_scheduler,
            train_loss=train_loss,
            validation_loss=validation_loss,
            validation_metrics=validation_metrics,
            best_validation_macro_auc=best_validation_macro_auc,
            trainable_layers=trainable_layers,
        )

        torch.save(checkpoint, last_checkpoint_path)
        print(
            "Latest checkpoint saved to:\n"
            f"{last_checkpoint_path}"
        )

        if improved:
            torch.save(checkpoint, best_checkpoint_path)
            print(
                "New best macro-AUC checkpoint saved to:\n"
                f"{best_checkpoint_path}"
            )
        else:
            print(
                "Best validation macro-AUC so far: "
                f"{format_metric(best_validation_macro_auc)}"
            )

    if not best_checkpoint_path.exists():
        raise RuntimeError(
            "No best checkpoint was saved."
        )

    try:
        best_checkpoint = torch.load(
            best_checkpoint_path,
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        best_checkpoint = torch.load(
            best_checkpoint_path,
            map_location=DEVICE,
        )

    model.load_state_dict(
        best_checkpoint["model_state_dict"]
    )

    final_validation_loss, final_validation_metrics = validate(
        model=model,
        dataloader=validation_loader,
        criterion=criterion,
        device=DEVICE,
    )

    print(f"\nBest epoch: {best_checkpoint['epoch']}")
    print("Final results for the best checkpoint:")

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

    print("\nFull fine-tuning completed.")
    print(f"Training history saved to:\n{history_path}")
    print(
        "Validation results saved to:\n"
        f"{validation_results_path}"
    )
    print(f"Best checkpoint:\n{best_checkpoint_path}")
    print(f"Latest checkpoint:\n{last_checkpoint_path}")


if __name__ == "__main__":
    main()
