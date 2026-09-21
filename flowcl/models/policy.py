"""The flow-matching policy, plus the §4.5 layer registry and §7.4 allowlist.

Spec §4.5:

    Maintain an explicit registry of projectable linear layers: trunk attention
    Q/K/V/O, trunk MLPs, decoder cross-attention, decoder MLPs, action-projection
    layers. Analysis and projection methods iterate this registry, not ad-hoc
    ``named_modules()`` matching. Order must be deterministic.

Spec §7.4 (the exclusion list projection experiments must honour):

    freeze LayerNorm weights/biases, all linear biases, AdaLN/FiLM modulation
    parameters, and ``s`` embeddings.

Why a registry rather than a name regex: a regex silently picks up new layers as the
model changes, and silently misses renamed ones. A hard registry with a completeness
assertion (:meth:`FlowPolicy.assert_registry_complete`) makes both failure modes loud.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from flowcl.data.spec import EmbodimentSpec
from flowcl.models.encoders import CachedTextEncoder, FrozenVisionEncoder
from flowcl.models.flow_head import FlowHead, SSampler, UniformSSampler, draw_with_generator
from flowcl.models.losses import flow_matching_loss, interpolate_actions
from flowcl.models.trunk import ObservationTrunk

# ImageNet statistics, matching what the frozen backbones were trained with.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class RegistryEntry:
    """One projectable layer (§4.5).

    Attributes:
        name: Dotted module path, unique and stable.
        module: The ``nn.Linear`` itself.
        group: Coarse role, e.g. ``trunk_attn``, ``decoder_mlp``. Used to report
            ``rho_l``/``c_l`` grouped by layer type.
        d_in: Input width, i.e. the dimension a subspace basis ``M_l`` lives in.
    """

    name: str
    module: nn.Linear
    group: str

    @property
    def d_in(self) -> int:
        return self.module.in_features

    @property
    def d_out(self) -> int:
        return self.module.out_features


class FlowPolicy(nn.Module):
    """Language-conditioned flow-matching policy over action chunks.

    Args:
        spec: Embodiment spec; determines ``d_state``, ``d_action`` and ``H``.
        vision_backbone / text_backbone: Encoder choices (§4.1).
        d_model, n_trunk_layers, n_heads: Trunk config (§4.2).
        n_decoder_layers: Flow-decoder depth (§4.3).
        s_sampler: ``p(s)`` used during training.
        pretrained: Load pretrained encoder weights.
    """

    def __init__(
        self,
        spec: EmbodimentSpec,
        vision_backbone: str = "dinov2_s",
        text_backbone: str = "clip_b",
        d_model: int = 512,
        n_trunk_layers: int = 8,
        n_heads: int = 8,
        n_decoder_layers: int = 4,
        n_context_tokens: int = 32,
        mlp_ratio: float = 4.0,
        max_tokens: int = 1024,
        s_sampler: SSampler | None = None,
        pretrained: bool = True,
        euler_steps: int = 10,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.d_model = d_model
        self.horizon = spec.action.chunk_horizon
        self.d_action = spec.d_action
        self.euler_steps = euler_steps
        self.s_sampler = s_sampler or UniformSSampler()

        self.vision_encoder = FrozenVisionEncoder(
            backbone=vision_backbone, d_model=d_model, pretrained=pretrained
        )
        self.text_encoder = CachedTextEncoder(
            backbone=text_backbone, d_model=d_model, pretrained=pretrained
        )
        self.trunk = ObservationTrunk(
            d_state=spec.d_state,
            d_model=d_model,
            n_layers=n_trunk_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            n_context_tokens=n_context_tokens,
            max_tokens=max_tokens,
        )
        self.flow_head = FlowHead(
            d_action=spec.d_action,
            horizon=self.horizon,
            d_model=d_model,
            n_layers=n_decoder_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
        )

        self.register_buffer(
            "image_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

        self._registry: tuple[RegistryEntry, ...] = self._build_registry()

    # ---- §4.5 layer registry --------------------------------------------------

    def _build_registry(self) -> tuple[RegistryEntry, ...]:
        """Enumerate projectable linear layers in a deterministic order.

        Order is: trunk state projection, then trunk blocks in depth order (attention
        Q, K, V, O, then MLP fc1, fc2), then the decoder's action input, then decoder
        blocks in depth order (self-attention, cross-attention, MLP), then the action
        output projection.

        Deliberately excluded:

        * frozen vision/text backbones — §0 freezes them
        * the encoders' trainable projections — they are the *interface* to frozen
          features, not part of the trunk/decoder under study
        * every AdaLN modulation projection — §7.4 excludes them
        * the ``s`` embedding MLP — §7.4 excludes it
        """
        entries: list[RegistryEntry] = []

        def add(name: str, module: nn.Module, group: str) -> None:
            if not isinstance(module, nn.Linear):
                raise TypeError(
                    f"registry entry {name} is {type(module).__name__}, not nn.Linear"
                )
            entries.append(RegistryEntry(name=name, module=module, group=group))

        add("trunk.state_projection", self.trunk.state_projection, "trunk_input")
        for i, block in enumerate(self.trunk.blocks):
            for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
                add(
                    f"trunk.blocks.{i}.attn.{proj}",
                    getattr(block.attn, proj),
                    "trunk_attn",
                )
            add(f"trunk.blocks.{i}.mlp.fc1", block.mlp.fc1, "trunk_mlp")
            add(f"trunk.blocks.{i}.mlp.fc2", block.mlp.fc2, "trunk_mlp")

        add("flow_head.action_in", self.flow_head.action_in, "decoder_input")
        for i, block in enumerate(self.flow_head.blocks):
            for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
                add(
                    f"flow_head.blocks.{i}.self_attn.{proj}",
                    getattr(block.self_attn, proj),
                    "decoder_self_attn",
                )
            for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
                add(
                    f"flow_head.blocks.{i}.cross_attn.{proj}",
                    getattr(block.cross_attn, proj),
                    "decoder_cross_attn",
                )
            add(f"flow_head.blocks.{i}.mlp.fc1", block.mlp.fc1, "decoder_mlp")
            add(f"flow_head.blocks.{i}.mlp.fc2", block.mlp.fc2, "decoder_mlp")
        add("flow_head.action_out", self.flow_head.action_out, "decoder_output")

        names = [entry.name for entry in entries]
        if len(set(names)) != len(names):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise RuntimeError(f"duplicate registry names: {duplicates}")
        return tuple(entries)

    def projectable_layers(self) -> tuple[RegistryEntry, ...]:
        """The §4.5 registry, in deterministic order."""
        return self._registry

    def registry_names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self._registry)

    def assert_registry_complete(self) -> None:
        """Assert the registry is exactly the set of eligible ``nn.Linear`` layers.

        "Eligible" means: inside the trunk or the flow decoder, and not an AdaLN
        modulation projection or part of the ``s`` embedding MLP (§7.4).

        Catches both directions of drift — a new linear layer that nobody registered,
        and a registry entry that no longer exists.
        """
        registered = set(self.registry_names())

        eligible: set[str] = set()
        for prefix, root in (("trunk", self.trunk), ("flow_head", self.flow_head)):
            for name, module in root.named_modules():
                if not isinstance(module, nn.Linear):
                    continue
                full = f"{prefix}.{name}"
                if ".modulation." in full or full.endswith("modulation.proj"):
                    continue  # AdaLN modulation (§7.4)
                if "final_modulation" in full:
                    continue  # AdaLN modulation (§7.4)
                if "s_embedding" in full:
                    continue  # flow-time embedding (§7.4)
                eligible.add(full)

        missing = sorted(eligible - registered)
        extra = sorted(registered - eligible)
        if missing or extra:
            raise RuntimeError(
                "§4.5 layer registry is out of sync with the model.\n"
                f"  eligible but unregistered: {missing}\n"
                f"  registered but not eligible: {extra}\n"
                "Update _build_registry(); analysis and projection methods iterate "
                "this registry, so a missing layer silently drops out of every "
                "measurement."
            )

    # ---- §7.4 parameter allowlist ---------------------------------------------

    def adaln_and_frozen_params(self) -> tuple[str, ...]:
        """Parameter names §7.4 requires be frozen in projection experiments.

        Returns LayerNorm weights and biases, *all* linear biases, AdaLN/FiLM
        modulation parameters, and the ``s`` embedding — as fully-qualified parameter
        names, in sorted order.

        Rationale from §7.4: these parameters can absorb a task shift without moving
        the weight matrices whose gradients are being projected, so leaving them
        trainable lets the model route around the projection and makes the measured
        effect of projection meaningless.
        """
        names: set[str] = set()

        for module_name, module in self.named_modules():
            if isinstance(module, nn.LayerNorm):
                for param_name, param in module.named_parameters(recurse=False):
                    if param is not None:
                        names.add(f"{module_name}.{param_name}")
            if isinstance(module, nn.Linear) and module.bias is not None:
                names.add(f"{module_name}.bias")

        for param_name, _ in self.named_parameters():
            if ".modulation." in param_name or "final_modulation" in param_name:
                names.add(param_name)
            if "s_embedding" in param_name:
                names.add(param_name)
            if param_name.endswith("chunk_position"):
                # Flow-time-adjacent: positional code over the chunk axis, which the
                # decoder can use to encode where in the chunk it is. Not a weight
                # matrix, so it has no input subspace to project against.
                names.add(param_name)

        # Sanity: every name must resolve to a real parameter.
        actual = dict(self.named_parameters())
        unknown = sorted(n for n in names if n not in actual)
        if unknown:
            raise RuntimeError(
                f"adaln_and_frozen_params() produced names with no matching "
                f"parameter: {unknown}"
            )
        return tuple(sorted(names))

    def projectable_parameters(self) -> tuple[str, ...]:
        """Weight matrices of registry layers — the parameters projection acts on."""
        return tuple(f"{entry.name}.weight" for entry in self._registry)

    # ---- parameter accounting -------------------------------------------------

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def frozen_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def parameter_report(self) -> dict:
        """Per-group parameter counts, for the §8.2 systems table."""
        by_group: dict[str, int] = {}
        for entry in self._registry:
            n = sum(p.numel() for p in entry.module.parameters())
            by_group[entry.group] = by_group.get(entry.group, 0) + n
        return {
            "trainable": self.trainable_parameter_count(),
            "frozen": self.frozen_parameter_count(),
            "registry_layers": len(self._registry),
            "by_group": by_group,
        }

    # ---- forward --------------------------------------------------------------

    def _prepare_images(self, images: dict[str, torch.Tensor]) -> torch.Tensor:
        """``{camera: (B, H, W, 3) uint8}`` -> normalised ``(B, 3, H, W)`` per camera.

        Cameras are concatenated along the token axis in ``spec.cameras`` order, which
        is the order the registry and every analysis assume.
        """
        missing = [c for c in self.spec.cameras if c not in images]
        if missing:
            raise KeyError(
                f"batch is missing cameras {missing}; got {sorted(images)}, spec "
                f"requires {list(self.spec.cameras)}"
            )

        token_groups = []
        for camera in self.spec.cameras:
            frames = images[camera]
            if frames.ndim != 4 or frames.shape[-1] != 3:
                raise ValueError(
                    f"images[{camera!r}] must be (B, H, W, 3), got "
                    f"{tuple(frames.shape)}"
                )
            pixels = frames.permute(0, 3, 1, 2).to(
                device=self.image_mean.device, dtype=self.image_mean.dtype
            )
            pixels = pixels / 255.0
            pixels = (pixels - self.image_mean) / self.image_std
            token_groups.append(self.vision_encoder(pixels))
        return torch.cat(token_groups, dim=1)

    def encode_observation(self, batch: dict) -> torch.Tensor:
        """Run encoders and trunk, returning context tokens ``(B, n_context, d_model)``."""
        vision_tokens = self._prepare_images(batch["images"])
        language_tokens = self.text_encoder(batch["language"])
        state = batch["state"].to(
            device=vision_tokens.device, dtype=vision_tokens.dtype
        )
        return self.trunk(state, language_tokens, vision_tokens)

    def forward(
        self,
        batch: dict,
        s: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        normalizer: float | torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict:
        """One training forward pass.

        Args:
            batch: Collated batch from :func:`flowcl.data.dataset.collate_chunks`.
            s: Flow times ``(B,)``. Drawn from :attr:`s_sampler` when omitted.
                Supplied explicitly by §7.5's per-bin microbatches.
            noise: ``A_0``, ``(B, H, D)``. Sampled when omitted.
            normalizer: Loss normaliser; see
                :func:`flowcl.models.losses.masked_mse`.
            generator: RNG for ``s`` and ``A_0``.

        Returns:
            ``{"loss", "velocity", "s", "noise", "context", "target_velocity"}``.
            ``s`` is returned so §7.1 hooks can tag captured activations with the
            flow time that produced them.
        """
        context = self.encode_observation(batch)
        target_actions = batch["actions"].to(
            device=context.device, dtype=context.dtype
        )
        mask = batch["action_mask"].to(device=context.device, dtype=context.dtype)
        batch_size = target_actions.shape[0]

        if s is None:
            s = self.s_sampler.sample(batch_size, context.device, generator=generator)
        else:
            s = s.to(device=context.device, dtype=context.dtype)
        if noise is None:
            noise = draw_with_generator(
                tuple(target_actions.shape),
                device=context.device,
                generator=generator,
                dtype=context.dtype,
                normal=True,
            )

        noisy_actions = interpolate_actions(noise, target_actions, s)
        velocity = self.flow_head(noisy_actions, context, s)
        loss = flow_matching_loss(
            velocity, noise, target_actions, mask, normalizer=normalizer
        )

        return {
            "loss": loss,
            "velocity": velocity,
            "target_velocity": target_actions - noise,
            "s": s,
            "noise": noise,
            "context": context,
        }

    @torch.no_grad()
    def sample(
        self,
        batch: dict,
        n_steps: int | None = None,
        generator: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict an action chunk, ``(B, H, d_action)`` (§4.4)."""
        context = self.encode_observation(batch)
        return self.flow_head.sample(
            context,
            n_steps=n_steps if n_steps is not None else self.euler_steps,
            generator=generator,
            noise=noise,
        )
