from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# ============================================================
# LABELS
# ============================================================
DEFAULT_LABEL_COLUMNS = [
    "Lung tumor",
    "Pneumonia",
    "Tuberculosis",
    "Other diseases",
    "No finding",
]

# ============================================================
# NPZ LOADING
# ============================================================
NPZ_COMMON_IMAGE_KEYS = (
    "image",
    "arr_0",
    "pixel_array",
    "pixels",
    "data",
)


def load_npz_image_array(
    image_path: str | Path,
    image_key: str | None = None,
) -> np.ndarray:
    """Loads and copies the image array stored in an NPZ archive."""
    image_path = Path(image_path)

    try:
        with np.load(image_path, allow_pickle=False) as archive:
            available_keys = list(archive.files)

            if not available_keys:
                raise ValueError(
                    f"The NPZ archive contains no arrays:\n{image_path}"
                )

            if image_key is not None:
                selected_key = str(image_key)
                if selected_key not in archive:
                    raise KeyError(
                        f"The key {selected_key!r} does not exist in NPZ "
                        f"{image_path}. Available keys: {available_keys}"
                    )
            else:
                selected_key = next(
                    (
                        key
                        for key in NPZ_COMMON_IMAGE_KEYS
                        if key in archive
                    ),
                    None,
                )

                if selected_key is None:
                    if len(available_keys) == 1:
                        selected_key = available_keys[0]
                    else:
                        raise ValueError(
                            "Could not automatically determine "
                            "which array contains the image. Set "
                            "NPZ_IMAGE_KEY/npz_image_key. "
                            f"Path: {image_path} | "
                            f"available keys: {available_keys}"
                        )

            return np.array(archive[selected_key], copy=True)
    except (OSError, ValueError, KeyError):
        raise
    except Exception as error:
        raise RuntimeError(
            f"Could not read the NPZ archive:\n{image_path}"
        ) from error


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_image(
    image: np.ndarray,
    method: str = "minmax",
) -> np.ndarray:
    """
    Normalizes the image to the [0, 1] range.

    Methods:
        minmax:
            Normalization using the image minimum and maximum.

        percentile:
            Clipping between the 1st and 99th percentiles, followed by
            normalization to [0, 1].

        none:
            Assumes the image is already normalized to [0, 1].
    """

    image = np.asarray(image, dtype=np.float32)

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if method == "minmax":
        minimum = float(image.min())
        maximum = float(image.max())

        if maximum > minimum:
            image = (image - minimum) / (maximum - minimum)
        else:
            image = np.zeros_like(image, dtype=np.float32)

    elif method == "percentile":
        lower = float(np.percentile(image, 1))
        upper = float(np.percentile(image, 99))

        if upper > lower:
            image = np.clip(image, lower, upper)
            image = (image - lower) / (upper - lower)
        else:
            image = np.zeros_like(image, dtype=np.float32)

    elif method == "none":
        minimum = float(image.min())
        maximum = float(image.max())

        if minimum < 0.0 or maximum > 1.0:
            raise ValueError(
                "normalization='none' was selected, but the image "
                "is not in the [0, 1] range. "
                f"Observed range: [{minimum:.4f}, {maximum:.4f}]"
            )

    else:
        raise ValueError(
            "Invalid normalization method. "
            "Choose 'minmax', 'percentile', or 'none'."
        )

    return image.astype(np.float32, copy=False)


# ============================================================
# DATASET
# ============================================================

class ChestXrayNPZDataset(Dataset):
    """
    Dataset for multi-label classification of chest X-rays
    stored in .npz files.

    Each CSV row must correspond to a single image.

    Processing:
        1. load the .npz file;
        2. convert the image to [H, W];
        3. normalize the image to [0, 1];
        4. convert to uint8;
        5. convert grayscale to RGB;
        6. apply the CLIP preprocessing pipeline.
    """

    def __init__(
        self,
        csv_path: str,
        npz_dir: str,
        transform: Callable[[Image.Image], torch.Tensor],
        label_columns: list[str] | None = None,
        image_id_column: str = "image_id",
        normalization: str = "minmax",
        image_key: str | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.npz_dir = Path(npz_dir)
        self.transform = transform
        self.image_id_column = image_id_column
        self.normalization = normalization
        self.image_key = image_key

        self.label_columns = (
            DEFAULT_LABEL_COLUMNS.copy()
            if label_columns is None
            else list(label_columns)
        )

        if not self.csv_path.is_file():
            raise FileNotFoundError(
                f"The CSV file does not exist:\n{self.csv_path}"
            )

        if not self.npz_dir.is_dir():
            raise FileNotFoundError(
                f"The image directory does not exist:\n{self.npz_dir}"
            )

        dataframe = pd.read_csv(self.csv_path)

        required_columns = [
            self.image_id_column,
            *self.label_columns,
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise KeyError(
                "The following required CSV columns are missing:\n"
                + "\n".join(missing_columns)
            )

        # Keep only the required columns.
        dataframe = dataframe[required_columns].copy()

        # Reject missing image identifiers.
        if dataframe[self.image_id_column].isna().any():
            rows = (
                dataframe.index[
                    dataframe[self.image_id_column].isna()
                ]
                + 2
            ).tolist()

            raise ValueError(
                "The CSV contains empty image_id values. "
                f"Rows: {rows[:10]}"
            )

        dataframe[self.image_id_column] = (
            dataframe[self.image_id_column]
            .astype(str)
            .str.strip()
        )

        empty_id_mask = dataframe[self.image_id_column].eq("")

        if empty_id_mask.any():
            rows = (dataframe.index[empty_id_mask] + 2).tolist()

            raise ValueError(
                "The CSV contains empty image_id values. "
                f"Rows: {rows[:10]}"
            )

        # Ensure that each image has only one annotation.
        duplicated_mask = dataframe[
            self.image_id_column
        ].duplicated(keep=False)

        if duplicated_mask.any():
            duplicated_ids = (
                dataframe.loc[
                    duplicated_mask,
                    self.image_id_column,
                ]
                .drop_duplicates()
                .tolist()
            )

            raise ValueError(
                "Duplicate image_id values were found in the CSV. "
                "Each image must have only one annotation.\n"
                f"Examples: {duplicated_ids[:10]}\n"
                f"Total duplicate IDs: {len(duplicated_ids)}"
            )

        # Convert and validate the labels.
        for column in self.label_columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="raise",
            )

        if dataframe[self.label_columns].isna().any().any():
            raise ValueError(
                "Some label columns contain missing values."
            )

        label_values = dataframe[
            self.label_columns
        ].to_numpy()

        valid_mask = np.isin(label_values, [0, 1])

        if not valid_mask.all():
            invalid_values = np.unique(
                label_values[~valid_mask]
            )

            raise ValueError(
                "Labels must contain only 0 and 1. "
                f"Invalid values: {invalid_values}"
            )

        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.dataframe)

    def _get_npz_path(
        self,
        image_id: str,
    ) -> Path:
        """
        Examples:
            abc123       -> abc123.npz
            abc123.dcm   -> abc123.npz
            abc123.npz   -> abc123.npz
        """

        filename = f"{Path(image_id).stem}.npz"
        return self.npz_dir / filename

    @staticmethod
    def _convert_to_2d(
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Converts common image formats to [H, W].
        """

        image = np.asarray(image)
        image = np.squeeze(image)

        if image.ndim == 2:
            return image

        if image.ndim == 3:
            # [1, H, W]
            if image.shape[0] == 1:
                return image[0]

            # [H, W, 1]
            if image.shape[-1] == 1:
                return image[..., 0]

            # [3, H, W]
            if image.shape[0] == 3:
                return image.mean(axis=0)

            # [H, W, 3]
            if image.shape[-1] == 3:
                return image.mean(axis=-1)

        raise ValueError(
            "The image has an unsupported shape. "
            f"Received shape: {image.shape}"
        )

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]

        image_id = str(
            row[self.image_id_column]
        ).strip()

        image_path = self._get_npz_path(image_id)

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image file not found:\n{image_path}"
            )

        image = load_npz_image_array(
            image_path=image_path,
            image_key=self.image_key,
        )

        image = self._convert_to_2d(image)

        image = normalize_image(
            image=image,
            method=self.normalization,
        )

        # Convert [0, 1] values to uint8 [0, 255].
        image = np.round(
            image * 255.0
        ).astype(np.uint8)

        # CLIP expects RGB images.
        pil_image = Image.fromarray(image).convert("RGB")

        image_tensor = self.transform(pil_image)

        labels = row[
            self.label_columns
        ].to_numpy(dtype=np.float32)

        labels_tensor = torch.from_numpy(labels)

        return image_tensor, labels_tensor


# ============================================================
# TRAINING AND VALIDATION DATALOADERS
# ============================================================

def create_train_val_dataloaders(
    train_csv: str,
    val_csv: str,
    train_npz_dir: str,
    val_npz_dir: str,
    train_transform: Callable[[Image.Image], torch.Tensor],
    val_transform: Callable[[Image.Image], torch.Tensor] | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    normalization: str = "minmax",
    npz_image_key: str | None = None,
    label_columns: list[str] | None = None,
) -> dict[str, DataLoader]:
    """
    Creates the training and validation DataLoaders only.

    The test set is loaded by a separate script.
    """

    if val_transform is None:
        val_transform = train_transform

    train_dataset = ChestXrayNPZDataset(
        csv_path=train_csv,
        npz_dir=train_npz_dir,
        transform=train_transform,
        label_columns=label_columns,
        normalization=normalization,
        image_key=npz_image_key,
    )

    val_dataset = ChestXrayNPZDataset(
        csv_path=val_csv,
        npz_dir=val_npz_dir,
        transform=val_transform,
        label_columns=label_columns,
        normalization=normalization,
        image_key=npz_image_key,
    )

    loader_arguments = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        drop_last=False,
        **loader_arguments,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        shuffle=False,
        drop_last=False,
        **loader_arguments,
    )

    print("Datasets loaded:")
    print(f"  Training:  {len(train_dataset)} images")
    print(f"  Validation: {len(val_dataset)} images")

    return {
        "train": train_loader,
        "val": val_loader,
    }
