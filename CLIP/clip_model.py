from pathlib import Path
from typing import Optional, Tuple

import clip
import torch
import torch.nn as nn
from torchvision.transforms import Compose


class CLIPImageClassifier(nn.Module):
    """
    Image classifier based on the CLIP visual encoder.

    Architecture:
        image
          -> CLIP visual encoder
          -> embedding
          -> linear layer
          -> class logits
    """

    def __init__(
        self,
        visual_encoder: nn.Module,
        feature_dim: int,
        num_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.visual = visual_encoder

        if dropout > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(feature_dim, num_classes),
            )
        else:
            self.classifier = nn.Linear(
                feature_dim,
                num_classes,
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        images:
            Tensor of shape [batch_size, 3, height, width].

        Returns
        -------
        logits:
            Tensor of shape [batch_size, num_classes].
        """

        features = self.visual(images)

        # The classification layer is kept in float32.
        features = features.float()

        logits = self.classifier(features)

        return logits


def load_clip_classifier(
    model_name: str = "ViT-B/32",
    num_classes: int = 5,
    dropout: float = 0.0,
    device: Optional[torch.device] = None,
    download_root: Optional[str] = None,
) -> Tuple[CLIPImageClassifier, Compose]:
    """
    Loads the pretrained CLIP visual encoder and adds
    a classification layer.

    The model is initially loaded on the CPU and converted to
    float32. This simplifies training without AMP.

    Parameters
    ----------
    model_name:
        CLIP architecture. For example:
            ViT-B/32
            ViT-B/16
            RN50

    num_classes:
        Number of task classes.

    dropout:
        Dropout applied before the classification layer.

    device:
        Target device for the model.

    download_root:
        Optional directory for storing CLIP weights.

    Returns
    -------
    model:
        Classification model.

    preprocess:
        Official preprocessing transforms for the CLIP model.
    """

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if download_root is not None:
        download_root = str(Path(download_root).expanduser())

    # Loading on the CPU prevents CLIP from initially
    # being placed in FP16 on the GPU.
    clip_model, preprocess = clip.load(
        name=model_name,
        device="cpu",
        jit=False,
        download_root=download_root,
    )

    clip_model = clip_model.float()

    visual_encoder = clip_model.visual

    if not hasattr(visual_encoder, "output_dim"):
        raise AttributeError(
            "The visual encoder does not have the 'output_dim' attribute."
        )

    feature_dim = visual_encoder.output_dim

    model = CLIPImageClassifier(
        visual_encoder=visual_encoder,
        feature_dim=feature_dim,
        num_classes=num_classes,
        dropout=dropout,
    )

    model = model.to(device)

    return model, preprocess


if __name__ == "__main__":
    selected_device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, preprocess = load_clip_classifier(
        model_name="ViT-B/32",
        num_classes=5,
        dropout=0.0,
        device=selected_device,
    )

    print(f"Device: {selected_device}")
    print(f"Visual model: {model.visual.__class__.__name__}")
    print(f"Classifier: {model.classifier}")
    print(f"Expected resolution: {model.visual.input_resolution}")
