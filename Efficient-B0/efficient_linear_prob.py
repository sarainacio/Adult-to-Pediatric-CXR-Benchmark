from pathlib import Path
import random

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

TRAIN_NPZ_DIR = Path("/path/to/vindr-cxr/train_npz")

# These CSV files must be created beforehand by the data-splitting script.
# This script does not split the dataset.
TRAIN_CSV = Path("/path/to/vindr-cxr/annotations/train_split_new.csv")

VALIDATION_CSV = Path("/path/to/vindr-cxr/annotations/val_split_new.csv")

OUTPUT_DIR = Path("./outputs/efficientnet/linear_probing")

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "efficientnet_b0.ra_in1k"

LABEL_COLUMNS = [
    "Lung tumor",
    "Pneumonia",
    "Tuberculosis",
    "Other diseases",
    "No finding",
]

IMAGE_SIZE = 224

# Input configuration associated with the efficientnet_b0.ra_in1k weights.
# The X-ray remains grayscale through resizing and augmentation;
# the single channel is then replicated to RGB to preserve the
# original pretrained three-channel conv_stem.
MODEL_INPUT_CHANNELS = 3
MODEL_INPUT_MEAN = (0.485, 0.456, 0.406)
MODEL_INPUT_STD = (0.229, 0.224, 0.225)

BATCH_SIZE = 32

# Fixed number of epochs; early stopping is not used.
NUM_EPOCHS = 100

# Only the classifier is trained with this learning rate.
CLASSIFIER_LEARNING_RATE = 1e-3

# Decoupled AdamW weight decay.
WEIGHT_DECAY = 1e-4

# Fixed scheduler applied to the classifier.
SCHEDULER_MILESTONES = [15, 30, 45, 60, 75, 90]
SCHEDULER_GAMMA = 0.7

NUM_WORKERS = 4
SEED = 2025

# Convert probabilities to binary predictions for F1 computation.
CLASSIFICATION_THRESHOLD = 0.5

# Set to False only if the NPZ images are already in the [0, 1] range.
APPLY_MIN_MAX_NORMALIZATION = True

# None enables automatic detection of the NPZ image array key.
NPZ_IMAGE_KEY = None


# ============================================================
# RANDOM SEED
# ============================================================


def set_seed(seed):
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



PREFERRED_NPZ_IMAGE_KEYS = (
    "image",
    "arr_0",
    "pixel_array",
    "pixels",
    "data",
)


def load_npz_image_array(image_path, image_key=None):
    """
    Loads the image array from an NPZ file.

    When image_key is None, common keys are checked first. If none
    is found, an archive containing just one array is accepted automatically.
    """

    try:
        with np.load(image_path, allow_pickle=False) as archive:
            available_keys = list(archive.files)

            if not available_keys:
                raise ValueError(
                    f"The NPZ file contains no arrays: {image_path}"
                )

            if image_key is not None:
                if image_key not in archive:
                    raise KeyError(
                        f"The NPZ key {image_key!r} does not exist in "
                        f"{image_path}. Available keys: {available_keys}"
                    )

                selected_key = image_key

            else:
                selected_key = next(
                    (
                        candidate
                        for candidate in PREFERRED_NPZ_IMAGE_KEYS
                        if candidate in archive
                    ),
                    None,
                )

                if selected_key is None and len(available_keys) == 1:
                    selected_key = available_keys[0]

                if selected_key is None:
                    raise ValueError(
                        "Could not determine which array contains the "
                        f"image in {image_path}. Available keys: "
                        f"{available_keys}. Set NPZ_IMAGE_KEY."
                    )

            return np.array(
                archive[selected_key],
                copy=True,
            )

    except OSError as error:
        raise OSError(
            f"Could not open the NPZ file: {image_path}"
        ) from error


def read_npz(image_path, image_key=None):
    """
    Reads an .npz file and returns a float32 tensor [1, H, W].

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

    # Remove a singleton channel dimension, if present.
    if image.ndim == 3:
        if image.shape[0] == 1:
            image = image[0]
        elif image.shape[-1] == 1:
            image = image[..., 0]
        else:
            raise ValueError(
                "The NPZ file has three dimensions but does not "
                "represent a single-channel image. "
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

    if APPLY_MIN_MAX_NORMALIZATION:
        image_min = float(image.min())
        image_max = float(image.max())

        if image_max > image_min:
            image = (
                (image - image_min) /
                (image_max - image_min)
            )
        else:
            image = np.zeros_like(
                image,
                dtype=np.float32,
            )
    else:
        image_min = float(image.min())
        image_max = float(image.max())

        if image_min < 0.0 or image_max > 1.0:
            raise ValueError(
                "APPLY_MIN_MAX_NORMALIZATION=False, but the image "
                "is not within the [0,1] range. "
                f"Path: {image_path} | "
                f"min={image_min:.6f} | max={image_max:.6f}"
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

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image):
        _, height, width = image.shape

        scale = self.output_size / max(height, width)
        new_height = max(1, round(height * scale))
        new_width = max(1, round(width * scale))

        image = F.interpolate(
            image.unsqueeze(0),
            size=(new_height, new_width),
            mode="bicubic",
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


# ============================================================
# GRAYSCALE -> RGB
# ============================================================


class GrayscaleToRGB:
    """Replicates the grayscale channel across RGB without adding color information."""

    def __call__(self, image):
        if image.ndim != 3 or image.shape[0] != 1:
            raise ValueError(
                "Expected a grayscale [1,H,W] tensor before "
                f"RGB conversion, but received {tuple(image.shape)}."
            )

        return image.repeat(3, 1, 1)


# ============================================================
# TRANSFORMS
# ============================================================


train_transform = transforms.Compose([
    # Preserve the entire anatomical field of view using interpolation
    # corresponding to the EfficientNet-B0 pretrained configuration.
    ResizeAndPad(IMAGE_SIZE),
    transforms.RandomRotation(degrees=5, fill=0.0),

    # Preserve the original pretrained RGB conv_stem.
    GrayscaleToRGB(),

    # Normalization associated with the model ImageNet-1k weights.
    transforms.Normalize(
        mean=MODEL_INPUT_MEAN,
        std=MODEL_INPUT_STD,
    ),
])

validation_transform = transforms.Compose([
    ResizeAndPad(IMAGE_SIZE),
    GrayscaleToRGB(),
    transforms.Normalize(
        mean=MODEL_INPUT_MEAN,
        std=MODEL_INPUT_STD,
    ),
])


# ============================================================
# DATASET
# ============================================================


class CXRDataset(Dataset):
    def __init__(
        self,
        dataframe,
        image_dir,
        transform=None,
        image_key=None,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_key = image_key

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_id = str(row["image_id"])
        image_path = self.image_dir / f"{image_id}.npz"

        if not image_path.exists():
            raise FileNotFoundError(
                "NPZ file not found:\n"
                f"{image_path}"
            )

        image = read_npz(
            image_path,
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


def load_split_csv(csv_path, split_name):
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
            f"Missing required columns in the {split_name}: "
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
        raise ValueError(f"The {split_name} CSV is empty.")

    return dataframe


# ============================================================
# CLASS WEIGHTS
# ============================================================


def calculate_pos_weight(train_df):
    """Computes negative-to-positive ratios using the training set only."""

    positive = torch.tensor(
        train_df[LABEL_COLUMNS].sum(axis=0).to_numpy(),
        dtype=torch.float32,
    )

    negative = len(train_df) - positive
    positive = positive.clamp(min=1.0)

    return negative / positive


# ============================================================
# VALIDATION METRICS
# ============================================================


def calculate_metrics(targets, probabilities):
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

    per_class_auc = {}
    valid_auc_values = []

    for class_index, class_name in enumerate(LABEL_COLUMNS):
        class_targets = targets[:, class_index]
        class_probabilities = probabilities[:, class_index]

        # AUROC requires at least one positive and one negative example.
        if np.unique(class_targets).size < 2:
            per_class_auc[class_name] = np.nan
            continue

        auc = roc_auc_score(
            class_targets,
            class_probabilities,
        )

        per_class_auc[class_name] = float(auc)
        valid_auc_values.append(auc)

    macro_auc = (
        float(np.mean(valid_auc_values))
        if valid_auc_values
        else np.nan
    )

    return {
        "macro_auc": macro_auc,
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "per_class_auc": per_class_auc,
    }


# ============================================================
# TRAIN ONE EPOCH
# ============================================================


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Runs one epoch over all training batches."""

    model.train()

    # model.train() also switches frozen BatchNorm layers to training mode.
    # Keep frozen BatchNorm layers in evaluation mode to preserve running statistics.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module_parameters = list(module.parameters())

            if (
                module_parameters
                and not any(
                    parameter.requires_grad
                    for parameter in module_parameters
                )
            ):
                module.eval()

    total_loss = 0.0

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False,
    )

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return total_loss / len(dataloader.dataset)


# ============================================================
# VALIDATION
# ============================================================


def validate(model, dataloader, criterion, device):
    """Evaluates the model without updating its parameters."""

    model.eval()

    total_loss = 0.0
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        progress_bar = tqdm(
            dataloader,
            desc="Validation",
            leave=False,
        )

        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)
            probabilities = torch.sigmoid(logits)

            total_loss += loss.item() * images.size(0)
            all_targets.append(labels.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())

    average_loss = total_loss / len(dataloader.dataset)

    all_targets = np.concatenate(all_targets, axis=0)
    all_probabilities = np.concatenate(all_probabilities, axis=0)

    metrics = calculate_metrics(
        targets=all_targets,
        probabilities=all_probabilities,
    )

    return average_loss, metrics


# ============================================================
# DISPLAY METRICS
# ============================================================


def print_metrics(name, loss, metrics, show_per_class=False):
    print(
        f"{name}: "
        f"loss={loss:.4f} | "
        f"macro AUC={metrics['macro_auc']:.4f} | "
        f"macro F1={metrics['macro_f1']:.4f} | "
        f"micro F1={metrics['micro_f1']:.4f}"
    )

    if show_per_class:
        print("Per-class AUROC:")

        for class_name, auc in metrics["per_class_auc"].items():
            if np.isnan(auc):
                print(f"  {class_name}: N/A")
            else:
                print(f"  {class_name}: {auc:.4f}")


# ============================================================
# CHECKPOINT
# ============================================================



def build_checkpoint(
    epoch,
    model,
    optimizer,
    scheduler,
    train_loss,
    validation_loss,
    validation_metrics,
    best_validation_macro_auc,
):
    return {
        "epoch": epoch,
        "model_name": MODEL_NAME,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "validation_metrics": validation_metrics,
        "best_validation_macro_auc": best_validation_macro_auc,
        "selection_metric": "validation_macro_auc",
        "label_columns": LABEL_COLUMNS,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "in_chans": MODEL_INPUT_CHANNELS,
        "input_representation": "grayscale_replicated_to_rgb",
        "model_input_mean": list(MODEL_INPUT_MEAN),
        "model_input_std": list(MODEL_INPUT_STD),
        "resize_interpolation": "bicubic",
        "input_format": "npz",
        "npz_image_key": NPZ_IMAGE_KEY,
        "dataset_implementation": "standalone_npz",
        "apply_min_max_normalization": APPLY_MIN_MAX_NORMALIZATION,
        "classifier_learning_rate": CLASSIFIER_LEARNING_RATE,
        "current_learning_rate": optimizer.param_groups[0]["lr"],
        "weight_decay": WEIGHT_DECAY,
        "scheduler": {
            "name": "MultiStepLR",
            "applied_to": "classifier",
            "milestones": list(SCHEDULER_MILESTONES),
            "gamma": SCHEDULER_GAMMA,
        },
        "early_stopping": False,
        "fine_tuning": "linear_probing",
        "trainable_layers": ["classifier."],
    }


# ============================================================
# MAIN
# ============================================================


def main():
    set_seed(SEED)

    if not TRAIN_NPZ_DIR.exists():
        raise FileNotFoundError(
            "The NPZ image directory does not exist:\n"
            f"{TRAIN_NPZ_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")
    print("AMP: not used")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Load previously created data splits
    # --------------------------------------------------------

    train_df = load_split_csv(TRAIN_CSV, "training")
    val_df = load_split_csv(VALIDATION_CSV, "validation")

    train_ids = set(train_df["image_id"])
    validation_ids = set(val_df["image_id"])
    overlapping_ids = train_ids.intersection(validation_ids)

    if overlapping_ids:
        examples = sorted(overlapping_ids)[:5]
        raise ValueError(
            "Overlapping image_id values were found between training and validation. "
            f"Examples: {examples}"
        )

    print(f"\nTraining CSV: {TRAIN_CSV}")
    print(f"Validation CSV: {VALIDATION_CSV}")
    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")

    print("\nPositive samples per class in training:")
    for label in LABEL_COLUMNS:
        print(f"  {label}: {int(train_df[label].sum())}")

    print("\nPositive samples per class in validation:")
    for label in LABEL_COLUMNS:
        print(f"  {label}: {int(val_df[label].sum())}")

    # --------------------------------------------------------
    # Dataset and DataLoader
    # --------------------------------------------------------

    train_dataset = CXRDataset(
        dataframe=train_df,
        image_dir=TRAIN_NPZ_DIR,
        transform=train_transform,
        image_key=NPZ_IMAGE_KEY,
    )

    val_dataset = CXRDataset(
        dataframe=val_df,
        image_dir=TRAIN_NPZ_DIR,
        transform=validation_transform,
        image_key=NPZ_IMAGE_KEY,
    )

    loader_arguments = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
        "persistent_workers": NUM_WORKERS > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_arguments,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_arguments,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        # Omitting in_chans preserves the original three-channel RGB conv_stem.
        num_classes=len(LABEL_COLUMNS),
    )

    # --------------------------------------------------------
    # Linear probing
    # --------------------------------------------------------

    # Freeze the entire EfficientNet-B0 encoder.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Train only the final classifier.
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    model = model.to(device)

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

    print(f"\nModel: {MODEL_NAME}")
    print("Backbone input channels: 3 (grayscale replicated to RGB)")
    print(f"NPZ directory: {TRAIN_NPZ_DIR}")
    print(
        "Min-max normalization during loading: "
        f"{APPLY_MIN_MAX_NORMALIZATION}"
    )
    print("Training protocol: linear probing")
    print("Trainable layers: classifier")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Fixed epochs: {NUM_EPOCHS}")
    print(
        "Initial classifier learning rate: "
        f"{CLASSIFIER_LEARNING_RATE}"
    )
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(
        "Scheduler: MultiStepLR for the classifier "
        f"(milestones={SCHEDULER_MILESTONES}, "
        f"gamma={SCHEDULER_GAMMA})"
    )
    print("Early stopping: not used")
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(f"Frozen parameters: {frozen_parameters:,}")

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    pos_weight = calculate_pos_weight(train_df).to(device)

    print("\npos_weight:")
    for label, weight in zip(
        LABEL_COLUMNS,
        pos_weight.cpu().tolist(),
    ):
        print(f"  {label}: {weight:.4f}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    # --------------------------------------------------------
    # Optimizer and scheduler
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=CLASSIFIER_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=SCHEDULER_MILESTONES,
        gamma=SCHEDULER_GAMMA,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_model_path = (
        OUTPUT_DIR /
        "best_efficientnet_b0_linear_probing_rgb_native.pt"
    )

    last_model_path = (
        OUTPUT_DIR /
        "last_efficientnet_b0_linear_probing_rgb_native.pt"
    )

    history_path = OUTPUT_DIR / "training_history.csv"

    best_val_macro_auc = float("-inf")
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        print(
            f"\n{'=' * 60}\n"
            f"Epoch {epoch}/{NUM_EPOCHS}\n"
            f"{'=' * 60}"
        )

        learning_rate_used = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(f"Train loss: {train_loss:.4f}")
        print_metrics(
            name="Validation",
            loss=val_loss,
            metrics=val_metrics,
        )
        print(f"Learning rate used: {learning_rate_used:.8f}")

        current_macro_auc = float(val_metrics["macro_auc"])

        if not np.isfinite(current_macro_auc):
            raise RuntimeError(
                "Validation macro-AUROC is not finite; "
                "cannot select the best checkpoint."
            )

        improved = current_macro_auc > best_val_macro_auc

        if improved:
            best_val_macro_auc = current_macro_auc

        # The scheduler steps at the end of each epoch and sets the next learning rate.
        scheduler.step()
        next_learning_rate = optimizer.param_groups[0]["lr"]

        print(f"Next learning rate: {next_learning_rate:.8f}")

        history.append({
            "epoch": epoch,
            "learning_rate_used": learning_rate_used,
            "learning_rate_next": next_learning_rate,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_macro_auc": current_macro_auc,
            "validation_macro_f1": val_metrics["macro_f1"],
            "validation_micro_f1": val_metrics["micro_f1"],
            "best_validation_macro_auc": best_val_macro_auc,
        })

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
            validation_loss=val_loss,
            validation_metrics=val_metrics,
            best_validation_macro_auc=best_val_macro_auc,
        )

        # Always save the most recent epoch checkpoint.
        torch.save(checkpoint, last_model_path)
        print(f"Latest checkpoint saved to:\n{last_model_path}")

        if improved:
            torch.save(checkpoint, best_model_path)
            print(
                "New best checkpoint by macro-AUROC saved to:\n"
                f"{best_model_path}"
            )
        else:
            print(
                "No improvement in validation macro-AUROC during this epoch. "
                "Training continues until the fixed number of epochs."
            )

    # --------------------------------------------------------
    # Final evaluation of the best checkpoint
    # --------------------------------------------------------

    best_checkpoint = torch.load(
        best_model_path,
        map_location=device,
    )

    model.load_state_dict(
        best_checkpoint["model_state_dict"]
    )

    final_val_loss, final_val_metrics = validate(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=device,
    )

    print(f"\nBest epoch: {best_checkpoint['epoch']}")
    print("Final results for the best checkpoint:")

    print_metrics(
        name="Validation",
        loss=final_val_loss,
        metrics=final_val_metrics,
        show_per_class=True,
    )

    validation_results = [{
        "split": "validation",
        "best_epoch": best_checkpoint["epoch"],
        "loss": final_val_loss,
        "macro_auc": final_val_metrics["macro_auc"],
        "macro_f1": final_val_metrics["macro_f1"],
        "micro_f1": final_val_metrics["micro_f1"],
        **{
            f"auc_{class_name}": auc
            for class_name, auc in
            final_val_metrics["per_class_auc"].items()
        },
    }]

    results_path = OUTPUT_DIR / "validation_results.csv"

    pd.DataFrame(validation_results).to_csv(
        results_path,
        index=False,
    )

    print("\nLinear probing training completed after the fixed number of epochs.")
    print(f"Training history saved to:\n{history_path}")
    print(f"Results saved to:\n{results_path}")
    print(f"Best checkpoint:\n{best_model_path}")
    print(f"Latest checkpoint:\n{last_model_path}")
    print(
        "The final test set must not be used in this script."
    )


if __name__ == "__main__":
    main()
