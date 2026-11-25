from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from clip_dataset import (
    DEFAULT_LABEL_COLUMNS,
    create_train_val_dataloaders,
)
from clip_model import load_clip_classifier
from tqdm.auto import tqdm

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("/path/to/vindr-cxr")
TRAIN_CSV = BASE_DIR / "annotations/train.csv"
VAL_CSV = BASE_DIR / "annotations/val.csv"
TRAIN_NPZ_DIR = BASE_DIR / "train"
VAL_NPZ_DIR = BASE_DIR / "train"
OUTPUT_DIR = Path("./outputs/clip/full_ft")

# ============================================================
# MODEL AND TRAINING CONFIGURATION
# ============================================================

MODEL_NAME = "ViT-B/16"
IMAGE_SIZE = 224
LABEL_COLUMNS = list(DEFAULT_LABEL_COLUMNS)
NUM_CLASSES = len(LABEL_COLUMNS)

BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_EPOCHS = 100

# Initial learning rates for the ViT visual encoder and the classifier.
ENCODER_LEARNING_RATE = 1e-5
CLASSIFIER_LEARNING_RATE = 1e-3

# Decoupled weight decay used by AdamW.
WEIGHT_DECAY = 1e-4

# Fixed scheduler: reduces both learning rates at the specified milestones (every 15 epochs).
SCHEDULER_MILESTONES = [15, 30, 45, 60, 75, 90]
SCHEDULER_GAMMA = 0.7

CLASSIFICATION_THRESHOLD = 0.5
SEED = 42 # Set the seed used for the experiment. #42, 123, 2025

# Preserves the image normalization used in the original experiments.
DATA_NORMALIZATION = "minmax"

# Image array key inside each NPZ archive.
# None enables automatic detection: image, arr_0, pixel_array, pixels, data
# or the only key in the archive.
NPZ_IMAGE_KEY: str | None = None

# The CLIP visual weights are converted to float32 to allow
# stable fine-tuning without AMP/GradScaler.
TRAIN_IN_FLOAT32 = True

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
# TRAINING PROTOCOL
# ============================================================

def configure_protocol(
    model: nn.Module,
) -> None:
    """
    Full fine-tuning of the CLIP visual encoder:

        - first freezes the entire model;
        - makes the entire visual encoder trainable;
        - makes the multi-label classifier trainable;
        - keeps any text branch or auxiliary components frozen.
    """

    if not hasattr(model, "visual"):
        raise AttributeError(
            "The model does not have the attribute 'visual'."
        )

    if not hasattr(model, "classifier"):
        raise AttributeError(
            "The model does not have the attribute 'classifier'."
        )

    # First freeze the entire wrapper, including any text encoder.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Full fine-tuning of the entire visual encoder.
    for parameter in model.visual.parameters():
        parameter.requires_grad = True

    # Train the multi-label classifier.
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True



def configure_training_mode(model: nn.Module) -> None:
    """Sets the entire visual encoder and classifier to training mode."""

    model.train()
    model.visual.train()
    model.classifier.train()


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
            "No positive training samples for: "
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
# EVALUATION METRICS
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

        # ROC-AUC requires at least one positive and one negative sample.
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
        "image_size": IMAGE_SIZE,
        "num_classes": NUM_CLASSES,
        "label_columns": LABEL_COLUMNS,
        "threshold": CLASSIFICATION_THRESHOLD,
        "input_format": "npz",
        "npz_image_key": NPZ_IMAGE_KEY,
        "dataset_implementation": "clip_dataset_npz",
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
        "model_selection_metric": "validation_macro_auc",
        "fixed_num_epochs": NUM_EPOCHS,
        "early_stopping": False,
        "trainable_layers": trainable_layers,
        "data_normalization": DATA_NORMALIZATION,
        "training_dtype": "float32",
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
            OUTPUT_DIR / "best_clip_vit_b16_full_fine_tuning.pt"
    )
    last_checkpoint_path = (
            OUTPUT_DIR / "last_clip_vit_b16_full_fine_tuning.pt"
    )
    history_path = OUTPUT_DIR / "training_history.csv"
    validation_results_path = (
            OUTPUT_DIR / "validation_results.csv"
    )

    print("=" * 60)
    print("CLIP — Full fine-tuning")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print("AMP: disabled")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # CLIP model and official preprocessing
    # --------------------------------------------------------

    model, preprocess = load_clip_classifier(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        dropout=0.0,
        device=DEVICE,
    )

    if TRAIN_IN_FLOAT32:
        model = model.float()

    configure_protocol(model)

    trainable_layers = [
        "visual (encoder visual completo)",
        "classifier",
    ]
    model = model.to(DEVICE)

    encoder_parameters = [
        parameter
        for parameter in model.visual.parameters()
        if parameter.requires_grad
    ]

    classifier_parameters = [
        parameter
        for parameter in model.classifier.parameters()
        if parameter.requires_grad
    ]

    if not encoder_parameters:
        raise RuntimeError(
            "The visual encoder has no trainable parameters."
        )

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

    print(f"\nModel: {MODEL_NAME}")
    print("Training protocol: full fine-tuning")
    print("Trainable layers:")
    for layer_name in trainable_layers:
        print(f"  {layer_name}")
    print(f"Classes: {NUM_CLASSES}")
    print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Fixed epochs: {NUM_EPOCHS}")
    print(
        "Encoder learning rate: "
        f"{ENCODER_LEARNING_RATE}"
    )
    print(
        "Initial classifier learning rate: "
        f"{CLASSIFIER_LEARNING_RATE}"
    )
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(
        "Scheduler: MultiStepLR for encoder and classifier | "
        f"milestones={SCHEDULER_MILESTONES} | "
        f"gamma={SCHEDULER_GAMMA}"
    )
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(f"Frozen parameters: {frozen_parameters:,}")

    # --------------------------------------------------------
    # Dataset and DataLoaders
    # --------------------------------------------------------

    print(
        "NPZ image key: "
        + (repr(NPZ_IMAGE_KEY) if NPZ_IMAGE_KEY is not None else "automatic")
    )

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
        npz_image_key=NPZ_IMAGE_KEY,
        label_columns=LABEL_COLUMNS,
    )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")

    # --------------------------------------------------------
    # Loss, optimizers and scheduler
    # --------------------------------------------------------

    pos_weight = calculate_pos_weight(
        train_loader=train_loader,
        device=DEVICE,
    )

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

    # --------------------------------------------------------
    # Fixed-budget training loop
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
        print(f"Epoch duration: {elapsed_time:.1f} s")
        print(
            "Encoder LR used: "
            f"{encoder_lr_used:.8f}"
        )
        print(
            "Classifier LR used: "
            f"{classifier_lr_used:.8f}"
        )

        # The same multiplicative learning-rate schedule is applied to both groups.
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
                "Encoder LR reduced by scheduler: "
                f"{encoder_lr_used:.8f} -> "
                f"{next_encoder_lr:.8f}"
            )
        else:
            print(
                "Next encoder LR: "
                f"{next_encoder_lr:.8f}"
            )

        if next_classifier_lr < classifier_lr_used:
            print(
                "Classifier LR reduced by scheduler: "
                f"{classifier_lr_used:.8f} -> "
                f"{next_classifier_lr:.8f}"
            )
        else:
            print(
                "Next classifier LR: "
                f"{next_classifier_lr:.8f}"
            )

        validation_macro_auc = validation_metrics["macro_auc"]
        improved = (
            np.isfinite(validation_macro_auc)
            and validation_macro_auc > best_validation_macro_auc
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
            "validation_macro_auc": validation_macro_auc,
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

        # Always save the state of the latest epoch.
        torch.save(checkpoint, last_checkpoint_path)
        print(
            "Latest checkpoint saved to:\n"
            f"{last_checkpoint_path}"
        )

        # Select the best checkpoint exclusively using validation macro-AUC.
        if improved:
            torch.save(checkpoint, best_checkpoint_path)
            print(
                "New best macro-AUC checkpoint saved to:\n"
                f"{best_checkpoint_path}"
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
