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
TRAIN_NPZ_DIR = Path("/path/to/vindr-cxr/train")
TRAIN_CSV = Path("/path/to/vindr-cxr/annotations/train.csv")
VALIDATION_CSV = Path("/path/to/vindr-cxr/annotations/val.csv")
OUTPUT_DIR = Path("./outputs/densenet/linear_probing")

# ============================================================
# MODEL AND TRAINING CONFIGURATION
# ============================================================

MODEL_NAME = "densenet121.tv_in1k"

LABEL_COLUMNS = [
    "Lung tumor",
    "Pneumonia",
    "Tuberculosis",
    "Other diseases",
    "No finding",
]

IMAGE_SIZE = 224

# Input configuration for the pretrained densenet121.tv_in1k weights.
# Keep the original three-channel backbone and repeat the grayscale X-ray
# across RGB channels before applying this normalization.
PRETRAINED_MEAN = (0.485, 0.456, 0.406)
PRETRAINED_STD = (0.229, 0.224, 0.225)

BATCH_SIZE = 32
NUM_EPOCHS = 100

# Only the classifier is trained with this learning rate.
CLASSIFIER_LEARNING_RATE = 1e-3

# Decoupled weight decay for AdamW.
WEIGHT_DECAY = 1e-4

# Fixed learning-rate schedule for the classifier.
SCHEDULER_MILESTONES = [15, 30, 45, 60, 75, 90]
SCHEDULER_GAMMA = 0.7

NUM_WORKERS = 4
SEED = 42 # Set the seed used for the experiment. #42, 123, 2025

# Convert probabilities to binary predictions for F1 calculation.
CLASSIFICATION_THRESHOLD = 0.5

# Use False only if the NPZ images are already stored in [0, 1].
APPLY_MIN_MAX_NORMALIZATION = True

# Image array key within the NPZ archive. Use None for automatic detection.
NPZ_IMAGE_KEY: str | None = None


# ============================================================
# REPRODUCIBILITY
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


def load_npz_image_array(
    image_path: Path,
    image_key: str | None,
) -> np.ndarray:
    """Reads and copies the image array stored in an NPZ archive."""

    try:
        with np.load(image_path, allow_pickle=False) as archive:
            available_keys = list(archive.files)

            if not available_keys:
                raise ValueError(
                    f"The NPZ archive contains no arrays: {image_path}"
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
                        for candidate in (
                            "image",
                            "arr_0",
                            "pixel_array",
                            "pixels",
                            "data",
                        )
                        if candidate in archive
                    ),
                    None,
                )

                if selected_key is None and len(available_keys) == 1:
                    selected_key = available_keys[0]

                if selected_key is None:
                    raise ValueError(
                        "Could not automatically identify the image array "
                        f"in {image_path}. Available keys: "
                        f"{available_keys}. Set NPZ_IMAGE_KEY explicitly."
                    )

            return np.array(
                archive[selected_key],
                copy=True,
            )
    except OSError as error:
        raise OSError(
            f"Could not open the NPZ archive: {image_path}"
        ) from error


def read_npz(
    image_path: Path,
    image_key: str | None = None,
) -> torch.Tensor:
    """
    Reads an .npz file and returns a float32 tensor of shape [1, H, W].

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
                "The NPZ array has three dimensions but does not "
                "represent a single-channel image. "
                f"Path: {image_path} | shape: {image.shape}"
            )

    if image.ndim != 2:
        raise ValueError(
            "Expected image shape [H,W], [1,H,W], or [H,W,1]. "
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
                "is not in the [0,1] range. "
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
    """Preserves aspect ratio and pads to output_size x output_size."""

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
# TRANSFORMS
# ============================================================


class GrayscaleToRGB:
    """Repeats the grayscale channel across RGB without introducing color information."""

    def __call__(self, image):
        if image.ndim != 3 or image.shape[0] != 1:
            raise ValueError(
                "Expected a grayscale [1,H,W] tensor before "
                f"RGB conversion, but received {tuple(image.shape)}."
            )

        return image.repeat(3, 1, 1)


train_transform = transforms.Compose([
    # CXR-specific geometry: preserve the complete field of view.
    ResizeAndPad(IMAGE_SIZE),
    transforms.RandomRotation(degrees=5, fill=0.0),
    # Preserve the original 3-channel ImageNet-pretrained DenseNet stem.
    GrayscaleToRGB(),
    # Native normalization of densenet121.tv_in1k.
    transforms.Normalize(mean=PRETRAINED_MEAN, std=PRETRAINED_STD),
])

validation_transform = transforms.Compose([
    ResizeAndPad(IMAGE_SIZE),
    GrayscaleToRGB(),
    transforms.Normalize(mean=PRETRAINED_MEAN, std=PRETRAINED_STD),
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
        image_key: str | None = None,
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

        # AUC requires at least one positive and one negative example.
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
# TRAINING EPOCH
# ============================================================


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Processes all training batches once."""

    model.train()

    # model.train() also switches frozen BatchNorm layers to training mode.
    # Keep them in eval mode to avoid updating running_mean/running_var.
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
        print("Per-class AUC:")

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
        "in_chans": 3,
        "grayscale_to_rgb": True,
        "pretrained_mean": list(PRETRAINED_MEAN),
        "pretrained_std": list(PRETRAINED_STD),
        "resize_interpolation": "bicubic",
        "resize_strategy": "aspect_ratio_preserving_pad",
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
        "trainable_layers": ["classifier"],
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
    print("AMP: disabled")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Load the predefined train and validation CSV files
    # --------------------------------------------------------

    train_df = load_split_csv(TRAIN_CSV, "training")
    val_df = load_split_csv(VALIDATION_CSV, "validation")

    train_ids = set(train_df["image_id"])
    validation_ids = set(val_df["image_id"])
    overlapping_ids = train_ids.intersection(validation_ids)

    if overlapping_ids:
        examples = sorted(overlapping_ids)[:5]
        raise ValueError(
            "Duplicate image_id values were found across training and validation. "
            f"Exemplos: {examples}"
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
    # Datasets and DataLoaders
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
        # Do not set in_chans: preserve the original RGB stem from tv_in1k weights.
        num_classes=len(LABEL_COLUMNS),
    )

    # --------------------------------------------------------
    # Linear probing
    # --------------------------------------------------------

    # Freeze the entire DenseNet.
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

    print(f"\nModelo: {MODEL_NAME}")
    print("Backbone input channels: 3 (grayscale repeated across RGB)")
    print(f"Pasta NPZ: {TRAIN_NPZ_DIR}")
    print(
        "NPZ image key: "
        + (repr(NPZ_IMAGE_KEY) if NPZ_IMAGE_KEY is not None else "automatic detection")
    )
    print(
        "Min-max normalization on load: "
        f"{APPLY_MIN_MAX_NORMALIZATION}"
    )
    print(f"Pretrained normalization: mean={PRETRAINED_MEAN}, std={PRETRAINED_STD}")
    print("Resize interpolation: bicubic | strategy: ResizeAndPad")
    print("Training protocol: linear probing")
    print("Trainable layers: classifier")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Initial learning rate: {CLASSIFIER_LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(
        "Scheduler: MultiStepLR "
        f"(milestones={SCHEDULER_MILESTONES}, gamma={SCHEDULER_GAMMA})"
    )
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
        "best_densenet121_linear_probing.pt"
    )

    last_model_path = (
        OUTPUT_DIR /
        "last_densenet121_linear_probing.pt"
    )

    history_path = OUTPUT_DIR / "training_history.csv"

    best_val_auc = float("-inf")
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
        print(f"Learning rate usada: {learning_rate_used:.8f}")

        # Update the learning rate at the end of the epoch for the next epoch.
        scheduler.step()
        next_learning_rate = optimizer.param_groups[0]["lr"]
        print(f"Learning rate seguinte: {next_learning_rate:.8f}")

        current_val_auc = float(val_metrics["macro_auc"])
        improved = (
            np.isfinite(current_val_auc)
            and current_val_auc > best_val_auc
        )

        if improved:
            best_val_auc = current_val_auc

        history.append({
            "epoch": epoch,
            "learning_rate_used": learning_rate_used,
            "learning_rate_next": next_learning_rate,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_macro_auc": current_val_auc,
            "validation_macro_f1": val_metrics["macro_f1"],
            "validation_micro_f1": val_metrics["micro_f1"],
            "best_validation_macro_auc": best_val_auc,
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
            best_validation_macro_auc=best_val_auc,
        )

        # Always save the most recent epoch checkpoint.
        torch.save(checkpoint, last_model_path)
        print(f"Latest checkpoint saved to:\n{last_model_path}")

        # Select the best checkpoint exclusively using validation macro-AUC.
        if improved:
            torch.save(checkpoint, best_model_path)
            print(
                "New best checkpoint saved to:\n"
                f"{best_model_path}"
            )

    if not best_model_path.exists():
        raise RuntimeError(
            "Could not select a checkpoint: "
            "validation macro-AUC was non-finite for every epoch."
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

    print("\nLinear probing training completed.")
    print(f"Training history saved to:\n{history_path}")
    print(f"Results saved to:\n{results_path}")
    print(f"Melhor checkpoint:\n{best_model_path}")
    print(f"Last checkpoint:\n{last_model_path}")
    print(
        "The held-out test set must not be used in this script."
    )


if __name__ == "__main__":
    main()
