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
from torchvision import transforms
from health_multimodal.image.model.pretrained import (
    get_biovil_t_image_encoder,
)
from tqdm.auto import tqdm


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path("/path/to/vindr-cxr")
TRAIN_CSV = BASE_DIR / "annotations/train.csv"
VAL_CSV = BASE_DIR / "annotations/val.csv"
TRAIN_NPZ_DIR = BASE_DIR / "train"
VAL_NPZ_DIR = BASE_DIR / "train"
OUTPUT_DIR = Path("./outputs/biovilt/partial_ft")

# ============================================================
# MODEL AND TRAINING CONFIGURATION
# ============================================================

BIOVIL_T_MODEL_ID = "microsoft/BiomedVLP-BioViL-T"
BIOVIL_T_IMAGE_WEIGHTS_FILENAME = (
    "biovil_t_image_model_proj_size_128.pt"
)

# Resolution used in this study to ensure comparability with the other models.
BIOVIL_T_IMAGE_SIZE = 224

# Official preprocessing settings published for the image encoder.
BIOVIL_T_OFFICIAL_RESIZE_SIZE = 512
BIOVIL_T_OFFICIAL_CROP_SIZE = 448
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

# BioViL-T uses a ResNet-50 backbone and is treated as a CNN.
ENCODER_LEARNING_RATE = 1e-4

# The newly initialized classifier uses a higher learning rate.
CLASSIFIER_LEARNING_RATE = 1e-3

# Decoupled weight decay used by AdamW.
WEIGHT_DECAY = 1e-4

# Fixed scheduler: decreases all learning rates by 30% every 15 epochs.
SCHEDULER_MILESTONES = [15, 30, 45, 60, 75, 90]
SCHEDULER_GAMMA = 0.7

CLASSIFICATION_THRESHOLD = 0.5
SEED = 42 # Set the seed used for the experiment. #42, 123, 2025

# Preserves the image normalization used in the original experiments.
DATA_NORMALIZATION = "minmax"

# Image array key inside each NPZ file.
# Use None for automatic detection. Keys are checked in this order:
# "image", "arr_0", "pixel_array", "pixels", and "data". If the archive
# contains only one array, it is selected automatically.
NPZ_IMAGE_KEY: str | None = None

# The BioViL-T encoder is trained in float32.
TRAIN_IN_FLOAT32 = True

# Number of final ResNet-50 residual stages to unfreeze.
NUM_UNFROZEN_RESNET_STAGES = 1

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
                "be in [0,1], but this condition is not met. "
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
    image_key: str | None,
) -> np.ndarray:
    """
    Extracts the image array from an NPZ archive.

    When ``image_key`` is None, common keys are checked; if the archive
    contains only one array, it is selected automatically.
    """

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
                if image_key not in archive:
                    raise KeyError(
                        f"The configured key {image_key!r} does not exist "
                        f"in {image_path}. Available keys: "
                        f"{available_keys}"
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
                        "Could not determine which array contains "
                        "the image. Set NPZ_IMAGE_KEY to one of the "
                        f"available keys: {available_keys}. Path: {image_path}"
                    )

            # Copy the array because the NPZ archive is closed after leaving the context manager.
            return np.array(
                archive[selected_key],
                copy=True,
            )

    except (OSError, ValueError, KeyError) as error:
        raise type(error)(
            f"Error reading NPZ file {image_path}: {error}"
        ) from error


def load_npz_as_grayscale_pil(
    image_path: Path,
    normalization: str,
    image_key: str | None,
) -> Image.Image:
    """
    Loads a single-channel chest X-ray from an NPZ archive and converts it to a grayscale PIL image.
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

def load_split_dataframe(
    csv_path: Path,
    label_columns: list[str],
) -> pd.DataFrame:
    """
    Reads and validates a CSV file containing:
        image_id and one binary column per class.
    """

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found:\n{csv_path}"
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
            f"The CSV file is empty:\n{csv_path}"
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
            f"Exemplos: {duplicated_examples}"
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
                f"Exemplos: {invalid_examples}"
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
                f"Exemplos: {invalid_examples}"
            )

        dataframe[column] = numeric_values.astype(
            np.float32
        )

    return dataframe


class BioViLTNPZDataset(Dataset):
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
        self.normalization = normalization
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

        image = load_npz_as_grayscale_pil(
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
    image_key: str | None,
    label_columns: list[str],
) -> dict[str, DataLoader]:
    """
    Creates DataLoaders without relying on external project modules.
    """

    train_dataframe = load_split_dataframe(
        csv_path=train_csv,
        label_columns=label_columns,
    )

    validation_dataframe = load_split_dataframe(
        csv_path=val_csv,
        label_columns=label_columns,
    )

    train_dataset = BioViLTNPZDataset(
        dataframe=train_dataframe,
        image_dir=train_npz_dir,
        label_columns=label_columns,
        transform=train_transform,
        normalization=normalization,
        image_key=image_key,
    )

    validation_dataset = BioViLTNPZDataset(
        dataframe=validation_dataframe,
        image_dir=val_npz_dir,
        label_columns=label_columns,
        transform=val_transform,
        normalization=normalization,
        image_key=image_key,
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
# BIOVIL-T MODEL
# ============================================================


class ExpandChannels:
    """
    Repeats a single-channel chest X-ray along the channel dimension.

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
                "ExpandChannels expected a tensor with shape [1,H,W], "
                f"mas recebeu {tuple(image.shape)}."
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
    Preprocessing for classification with BioViL-T.

    The official implementation uses Resize(512) followed by CenterCrop(448).
    Here, the final resolution is 224 x 224 to ensure comparability with
    other models. The convolutional encoder supports this resolution.

    The official pipeline does not apply mean/std normalization:
    it converts the image to a tensor in [0,1] and replicates the channel to RGB.
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


class BioViLTClassifier(nn.Module):
    """
    BioViL-T visual encoder with a linear multi-label classifier.

    For a single chest X-ray, the BioViL-T encoder combines:
        - static ResNet-50 features;
        - the learned embedding representing a missing previous image.

    The classifier receives the 512-dimensional global pre-projection
    embedding, following the pathway used by the official downstream
    classifier.
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
                f"mas foi recebido {tuple(features.shape)}."
            )

        if features.shape[1] != self.feature_size:
            raise ValueError(
                "The BioViL-T embedding dimension is incompatible "
                "with the classifier: "
                f"{features.shape[1]} vs {self.feature_size}."
            )

        return self.classifier(
            features
        )


def load_biovil_t_classifier(
    num_classes: int,
    dropout: float,
):
    """
    Loads the BioViL-T visual encoder and weights published on
    the Hugging Face Hub.

    Important:
        AutoModel.from_pretrained("microsoft/BiomedVLP-BioViL-T")
        loads the CXR-BERT text encoder, not the image encoder.

    The official get_biovil_t_image_encoder() builds the visual
    ResNet-50 + temporal module architecture and automatically downloads
    the biovil_t_image_model_proj_size_128.pt file.
    """

    try:
        pretrained_image_model = (
            get_biovil_t_image_encoder()
        )
    except Exception as error:
        raise RuntimeError(
            "Could not load the BioViL-T visual encoder. "
            "Ensure that the 'health_multimodal' package is installed "
            "and Hugging Face is accessible on the first run."
        ) from error

    if not hasattr(
        pretrained_image_model,
        "encoder",
    ):
        raise AttributeError(
            "The BioViL-T model does not have an 'encoder' attribute."
        )

    feature_size = getattr(
        pretrained_image_model,
        "feature_size",
        None,
    )

    if feature_size is None:
        raise AttributeError(
            "Could not determine "
            "pretrained_image_model.feature_size."
        )

    model = BioViLTClassifier(
        encoder=pretrained_image_model.encoder,
        feature_size=int(feature_size),
        num_classes=num_classes,
        dropout=dropout,
    )

    preprocess = build_biovil_t_preprocess(
        image_size=BIOVIL_T_IMAGE_SIZE,
    )

    return (
        model,
        preprocess,
    )


# ============================================================
# TRAINING PROTOCOL
# ============================================================


def get_resnet_stages(
    model: nn.Module,
) -> list[tuple[str, nn.Module]]:
    """
    Returns the four residual stages of the BioViL-T ResNet-50.
    """

    if not hasattr(
        model,
        "encoder",
    ):
        raise AttributeError(
            "The model does not have an 'encoder' attribute."
        )

    biovil_encoder = model.encoder

    if not hasattr(
        biovil_encoder,
        "encoder",
    ):
        raise AttributeError(
            "The BioViL-T encoder does not contain the internal ResNet backbone."
        )

    resnet = biovil_encoder.encoder
    stage_names = [
        "layer1",
        "layer2",
        "layer3",
        "layer4",
    ]

    stages = []

    for stage_name in stage_names:
        if not hasattr(
            resnet,
            stage_name,
        ):
            raise AttributeError(
                "The BioViL-T ResNet-50 does not contain "
                f"the stage '{stage_name}'."
            )

        stages.append(
            (
                stage_name,
                getattr(
                    resnet,
                    stage_name,
                ),
            )
        )

    return stages


def configure_protocol(
    model: nn.Module,
) -> list[str]:
    """
    Partial fine-tuning for single-image chest X-ray classification:

        - freezes the entire model;
        - unfreezes the final NUM_UNFROZEN_RESNET_STAGES residual stages;
        - unfreezes the 1x1 backbone_to_vit projection;
        - unfreezes the missing-previous-image embedding;
        - unfreezes the multi-label classification head.

    The temporal Vision Transformer is not activated when previous_image=None.
    """

    if not hasattr(
        model,
        "classifier",
    ):
        raise AttributeError(
            "The model does not have a 'classifier' attribute."
        )

    stages = get_resnet_stages(
        model
    )

    number_of_stages = len(
        stages
    )

    if not (
        1
        <= NUM_UNFROZEN_RESNET_STAGES
        <= number_of_stages
    ):
        raise ValueError(
            "NUM_UNFROZEN_RESNET_STAGES must be between 1 and "
            f"{number_of_stages}, mas foi recebido "
            f"{NUM_UNFROZEN_RESNET_STAGES}."
        )

    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable_layers = []
    selected_stages = stages[
        -NUM_UNFROZEN_RESNET_STAGES:
    ]

    for stage_name, stage in selected_stages:
        for parameter in stage.parameters():
            parameter.requires_grad = True

        trainable_layers.append(
            f"encoder.encoder.{stage_name}"
        )

    if not hasattr(
        model.encoder,
        "backbone_to_vit",
    ):
        raise AttributeError(
            "The BioViL-T encoder does not contain 'backbone_to_vit'."
        )

    for parameter in (
        model.encoder
        .backbone_to_vit
        .parameters()
    ):
        parameter.requires_grad = True

    trainable_layers.append(
        "encoder.backbone_to_vit"
    )

    if not hasattr(
        model.encoder,
        "missing_previous_emb",
    ):
        raise AttributeError(
            "The BioViL-T encoder does not contain "
            "'missing_previous_emb'."
        )

    model.encoder.missing_previous_emb.requires_grad_(
        True
    )

    trainable_layers.append(
        "encoder.missing_previous_emb"
    )

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    trainable_layers.append(
        "classifier"
    )

    return trainable_layers


def configure_training_mode(
    model: nn.Module,
) -> None:
    """
    Keeps frozen encoder modules in evaluation mode and sets only the
    unfrozen modules and classification head to training mode.
    """

    model.train()
    model.encoder.eval()

    stages = get_resnet_stages(
        model
    )

    for _, stage in stages[
        -NUM_UNFROZEN_RESNET_STAGES:
    ]:
        stage.train()

    model.encoder.backbone_to_vit.train()
    model.classifier.train()


# ============================================================
# CLASS WEIGHTS
# ============================================================


def calculate_pos_weight(
    train_loader,
    device: torch.device,
) -> torch.Tensor:
    """Computes negative-to-positive class weights using only the training set."""

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
            "No positive training samples were found for: "
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
# SINGLE-EPOCH TRAINING
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
        "protocol": "partial_fine_tuning",
        "fine_tuning": "partial_fine_tuning",
        "model_name": "BioViL-T",
        "model_id": BIOVIL_T_MODEL_ID,
        "image_weights_filename": (
            BIOVIL_T_IMAGE_WEIGHTS_FILENAME
        ),
        "model_source": "huggingface_health_multimodal",
        "image_size": BIOVIL_T_IMAGE_SIZE,
        "official_resize_size": (
            BIOVIL_T_OFFICIAL_RESIZE_SIZE
        ),
        "official_crop_size": (
            BIOVIL_T_OFFICIAL_CROP_SIZE
        ),
        "feature_size": model.feature_size,
        "feature_source": "global_embedding_pre_projection",
        "single_image_mode": True,
        "previous_image": None,
        "temporal_transformer_active": False,
        "pretrained": True,
        "num_classes": NUM_CLASSES,
        "label_columns": LABEL_COLUMNS,
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
        "selection_metric": "validation_macro_auc",
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
        "trainable_layers": trainable_layers,
        "data_normalization": DATA_NORMALIZATION,
        "npz_image_key": NPZ_IMAGE_KEY,
        "dataset_implementation": "standalone_npz",
        "training_dtype": "float32",
        "seed": SEED,
        "num_unfrozen_resnet_stages": (
            NUM_UNFROZEN_RESNET_STAGES
        ),
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
        OUTPUT_DIR / "best_biovil_t_224_partial_fine_tuning.pt"
    )
    last_checkpoint_path = (
        OUTPUT_DIR / "last_biovil_t_224_partial_fine_tuning.pt"
    )
    history_path = OUTPUT_DIR / "training_history.csv"
    validation_results_path = (
        OUTPUT_DIR / "validation_results.csv"
    )

    print("=" * 60)
    print("BioViL-T — Partial fine-tuning")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print("AMP: not used")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # BioViL-T model and preprocessing
    # --------------------------------------------------------

    model, preprocess = load_biovil_t_classifier(
        num_classes=NUM_CLASSES,
        dropout=0.0,
    )

    if TRAIN_IN_FLOAT32:
        model = model.float()

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

    print(f"\nModel: {BIOVIL_T_MODEL_ID}")
    print(
        "Official image encoder resolution: "
        f"{BIOVIL_T_OFFICIAL_CROP_SIZE} x "
        f"{BIOVIL_T_OFFICIAL_CROP_SIZE}"
    )
    print(
        "Resolution used in this experiment: "
        f"{BIOVIL_T_IMAGE_SIZE} x {BIOVIL_T_IMAGE_SIZE}"
    )
    print(
        "Input mode: single chest X-ray; "
        "previous_image=None"
    )
    print(
        "Global embedding dimension: "
        f"{model.feature_size}"
    )
    print("Training protocol: partial fine-tuning")
    print("Trainable layers:")
    for layer_name in trainable_layers:
        print(f"  {layer_name}")
    print(f"Classes: {NUM_CLASSES}")
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
    print(
        "Scheduler: MultiStepLR for encoder and classifier | "
        f"milestones={SCHEDULER_MILESTONES} | "
        f"gamma={SCHEDULER_GAMMA}"
    )
    print(f"Total parameters: {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(f"Frozen parameters: {frozen_parameters:,}")

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
        image_key=NPZ_IMAGE_KEY,
        label_columns=LABEL_COLUMNS,
    )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")

    # --------------------------------------------------------
    # Loss, optimizers, and scheduler
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

        # Apply the fixed scheduler to both parameter groups at the end of each epoch.
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
                "Encoder LR reduced by the scheduler: "
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
                "Classifier LR reduced by the scheduler: "
                f"{classifier_lr_used:.8f} -> "
                f"{next_classifier_lr:.8f}"
            )
        else:
            print(
                "Next classifier LR: "
                f"{next_classifier_lr:.8f}"
            )

        current_validation_macro_auc = float(
            validation_metrics["macro_auc"]
        )
        improved = (
            np.isfinite(current_validation_macro_auc)
            and current_validation_macro_auc
            > best_validation_macro_auc
        )

        if improved:
            best_validation_macro_auc = (
                current_validation_macro_auc
            )

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
            best_validation_macro_auc=(
                best_validation_macro_auc
            ),
            trainable_layers=trainable_layers,
        )

        torch.save(checkpoint, last_checkpoint_path)
        print(
            "Last checkpoint saved to:\n"
            f"{last_checkpoint_path}"
        )

        if improved:
            torch.save(checkpoint, best_checkpoint_path)
            print(
                "New best checkpoint saved to:\n"
                f"{best_checkpoint_path}"
            )
        else:
            print(
                "Validation macro-AUC did not exceed the best value: "
                f"{best_validation_macro_auc:.6f}"
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

    print("\nBioViL-T partial fine-tuning training completed.")
    print(f"Training history saved to:\n{history_path}")
    print(
        "Validation results saved to:\n"
        f"{validation_results_path}"
    )
    print(f"Best checkpoint:\n{best_checkpoint_path}")
    print(f"Last checkpoint:\n{last_checkpoint_path}")


if __name__ == "__main__":
    main()
