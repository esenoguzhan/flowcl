"""Frozen vision and language encoders.

Spec §0: the visual encoder is **frozen** during all continual learning, which
confines forgetting to the trunk and flow decoder so that ``rho_l``/``c_l`` measure
the mechanism under study rather than perception drift.

Spec §4.1: DINOv2-S or SigLIP-B, frozen, with a *trainable* linear patch projection.
Language uses a frozen text encoder, cached per task string, because there are only a
handful of unique instructions and running the text tower every step is pure waste.

Deviation from §4.1, agreed in planning: language is kept as **per-token** embeddings
rather than one pooled vector. In ``seq_correlated`` the four tasks differ only by the
object noun ("pick up the milk / the ketchup / ..."), and pooled sentence embeddings
for those are nearly collinear. If the policy cannot tell the tasks apart, the study
measures under-conditioning instead of interference.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

# Vision backbones we support, with their HuggingFace ids and output widths.
VISION_BACKBONES = {
    "dinov2_s": ("facebook/dinov2-small", 384),
    "dinov2_b": ("facebook/dinov2-base", 768),
    "siglip_b": ("google/siglip-base-patch16-224", 768),
    # §4.1 escape hatch: a small CNN trained from scratch, for the single-task
    # capability study if frozen features underperform at 50-demo scale.
    "resnet18_scratch": ("resnet18", 512),
}

TEXT_BACKBONES = {
    "clip_b": ("openai/clip-vit-base-patch32", 512),
    "t5_small": ("t5-small", 512),
}


def _assert_frozen(module: nn.Module, name: str) -> None:
    trainable = [n for n, p in module.named_parameters() if p.requires_grad]
    if trainable:
        raise RuntimeError(
            f"{name} must be frozen per §0 but these parameters require grad: "
            f"{trainable[:5]}{'...' if len(trainable) > 5 else ''}"
        )


@dataclass(frozen=True)
class VisionOutput:
    """Patch tokens from one camera."""

    tokens: torch.Tensor  # (B, n_patches, d_model)


class FrozenVisionEncoder(nn.Module):
    """A frozen ViT (or from-scratch ResNet) plus a trainable linear patch projection.

    The backbone is frozen and placed in ``eval`` mode permanently: its BatchNorm /
    dropout must not drift with the training distribution, or "frozen encoder" would
    be untrue in a way that is invisible in the parameter count.

    Args:
        backbone: Key into :data:`VISION_BACKBONES`.
        d_model: Trunk width to project patch features into.
        pretrained: Load pretrained weights. Off in tests for speed.
    """

    def __init__(
        self,
        backbone: str = "dinov2_s",
        d_model: int = 512,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if backbone not in VISION_BACKBONES:
            raise ValueError(
                f"Unknown vision backbone {backbone!r}; choose from "
                f"{sorted(VISION_BACKBONES)}"
            )
        self.backbone_name = backbone
        self.d_model = d_model
        hf_id, feature_dim = VISION_BACKBONES[backbone]
        self.feature_dim = feature_dim
        self.is_scratch_cnn = backbone == "resnet18_scratch"

        if self.is_scratch_cnn:
            # §4.1 escape hatch. Trained from scratch, so NOT frozen; recorded
            # explicitly because it changes what §0 guarantees.
            from torchvision.models import resnet18

            net = resnet18(weights=None)
            self.backbone = nn.Sequential(*list(net.children())[:-2])
            self.frozen = False
        else:
            from transformers import AutoConfig, AutoModel

            if pretrained:
                self.backbone = AutoModel.from_pretrained(hf_id)
            else:
                self.backbone = AutoModel.from_config(AutoConfig.from_pretrained(hf_id))
            if backbone.startswith("siglip"):
                self.backbone = self.backbone.vision_model
            self.backbone.requires_grad_(False)
            self.backbone.eval()
            self.frozen = True

        # The one trainable part of the visual path (§4.1).
        self.patch_projection = nn.Linear(feature_dim, d_model)

        if self.frozen:
            _assert_frozen(self.backbone, f"vision backbone {backbone}")

    def train(self, mode: bool = True):  # noqa: D102
        super().train(mode)
        # Keep the frozen backbone in eval mode regardless of the parent's mode.
        if self.frozen:
            self.backbone.eval()
        return self

    @torch.no_grad()
    def _features(self, images: torch.Tensor) -> torch.Tensor:
        """Backbone features for ``(B, 3, H, W)`` float images, without gradients."""
        if self.is_scratch_cnn:
            raise RuntimeError("_features is only for the frozen path")
        out = self.backbone(pixel_values=images).last_hidden_state
        return out

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Project one camera's frames to trunk tokens.

        Args:
            images: ``(B, 3, H, W)`` float32, already normalised.

        Returns:
            ``(B, n_tokens, d_model)``.
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"expected (B, 3, H, W) images, got {tuple(images.shape)}")

        if self.is_scratch_cnn:
            feature_map = self.backbone(images)
            tokens = feature_map.flatten(2).transpose(1, 2)
        else:
            tokens = self._features(images)
        return self.patch_projection(tokens)


class CachedTextEncoder(nn.Module):
    """Frozen text encoder with a per-string cache of per-token embeddings.

    §4.1: "cached per task string (there are few unique instructions — cache them, do
    not run the text tower every step)". The cache is keyed by the raw instruction
    string and survives for the encoder's lifetime.

    Cached tensors are stored on CPU and moved on use, so the cache does not pin GPU
    memory proportional to the number of tasks.
    """

    def __init__(
        self,
        backbone: str = "clip_b",
        d_model: int = 512,
        pretrained: bool = True,
        max_length: int = 32,
    ) -> None:
        super().__init__()
        if backbone not in TEXT_BACKBONES:
            raise ValueError(
                f"Unknown text backbone {backbone!r}; choose from "
                f"{sorted(TEXT_BACKBONES)}"
            )
        self.backbone_name = backbone
        self.d_model = d_model
        self.max_length = max_length
        hf_id, feature_dim = TEXT_BACKBONES[backbone]
        self.feature_dim = feature_dim

        from transformers import AutoConfig, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        if backbone.startswith("clip"):
            from transformers import CLIPTextModel

            self.backbone = (
                CLIPTextModel.from_pretrained(hf_id)
                if pretrained
                else CLIPTextModel(AutoConfig.from_pretrained(hf_id).text_config)
            )
        else:
            from transformers import T5EncoderModel

            self.backbone = (
                T5EncoderModel.from_pretrained(hf_id)
                if pretrained
                else T5EncoderModel(AutoConfig.from_pretrained(hf_id))
            )
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        _assert_frozen(self.backbone, f"text backbone {backbone}")

        # Trainable projection into the trunk width, mirroring the visual path.
        self.token_projection = nn.Linear(feature_dim, d_model)

        self._cache: dict[str, torch.Tensor] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def train(self, mode: bool = True):  # noqa: D102
        super().train(mode)
        self.backbone.eval()
        return self

    @property
    def cache_stats(self) -> dict[str, int]:
        return {
            "size": len(self._cache),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
        }

    def clear_cache(self) -> None:
        self._cache.clear()

    @torch.no_grad()
    def _encode_uncached(self, texts: list[str]) -> torch.Tensor:
        device = next(self.backbone.parameters()).device
        batch = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)
        out = self.backbone(**batch).last_hidden_state
        return out

    def forward(self, texts: list[str]) -> torch.Tensor:
        """Encode instructions to ``(B, max_length, d_model)`` token embeddings.

        Only strings absent from the cache are pushed through the text tower.
        """
        if isinstance(texts, str):
            raise TypeError(
                "pass a list of instructions, not a single string, so the batch "
                "dimension is unambiguous"
            )

        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            encoded = self._encode_uncached(missing)
            for text, row in zip(missing, encoded):
                self._cache[text] = row.detach().cpu()
        self._cache_misses += len(missing)
        self._cache_hits += len(texts) - len(missing)

        device = self.token_projection.weight.device
        stacked = torch.stack([self._cache[t] for t in texts]).to(device)
        return self.token_projection(stacked)
