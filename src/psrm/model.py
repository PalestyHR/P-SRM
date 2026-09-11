"""Mask-Scoring-inspired candidate-quality head for P-SRM P-SRM.

The module estimates the quality of the exact candidate represented by the
candidate-support plane.  It neither receives the native margin nor produces a
new candidate.  Native-margin fusion is fitted later from grouped-OOF scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PointQualitySidecarConfig:
    input_channels: int = 3
    conv_channels: int = 64
    hidden_width: int = 256
    metadata_dim: int = 4
    dropout: float = 0.0
    activation: str = "relu"
    leaky_relu_slope: float = 0.01
    aux_mode: str = "none"

    def validate(self) -> None:
        if self.input_channels != 3:
            raise ValueError("P-SRM dense projection must contain exactly three role planes")
        if self.conv_channels <= 0 or self.hidden_width <= 0:
            raise ValueError("network widths must be positive")
        if self.metadata_dim != 4:
            raise ValueError("metadata is fixed to x, y, in-frame, and boundary-state")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.activation not in {"relu", "leaky_relu"}:
            raise ValueError("activation must be relu or leaky_relu")
        if not 0.0 < self.leaky_relu_slope < 1.0:
            raise ValueError("leaky_relu_slope must be in (0, 1)")
        if self.aux_mode not in {"none", "presence"}:
            raise ValueError("aux_mode must be none or presence")


class PointQualitySidecar(nn.Module):
    """Four-convolution quality head adapted from the MaskIoU-head pattern.

    Input
    -----
    dense_evidence:
        ``[B, 3, 32, 32]`` with task-evidence, candidate-support, and
        valid-support planes.
    candidate_metadata:
        ``[B, 4]`` containing normalized frame x/y, in-frame state, and
        support-boundary state.  These fields are mechanical provenance, not
        handcrafted reliability features.

    Output
    ------
    A pre-sigmoid candidate-quality logit of shape ``[B]``.
    """

    def __init__(self, config: PointQualitySidecarConfig | None = None) -> None:
        super().__init__()
        self.config = config or PointQualitySidecarConfig()
        self.config.validate()
        c = self.config.conv_channels
        activation = self._activation
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(3, c, kernel_size=3, stride=1, padding=1),
            activation(),
            nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1),
            activation(),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1),
            activation(),
            nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1),
            activation(),
        )
        flattened = c * 8 * 8
        h = self.config.hidden_width
        self.representation_head = nn.Sequential(
            nn.Linear(flattened + self.config.metadata_dim, h),
            activation(),
            nn.Dropout(self.config.dropout),
            nn.Linear(h, h),
            activation(),
            nn.Dropout(self.config.dropout),
        )
        self.admission_head = nn.Linear(h, 1)
        self.presence_head = (
            nn.Linear(h, 1)
            if self.config.aux_mode in {"presence", "combined"}
            else None
        )
        self.localization_quality_head = (
            nn.Linear(h, 1)
            if self.config.aux_mode in {"quality", "combined"}
            else None
        )
        self.reset_parameters()

    def _activation(self) -> nn.Module:
        if self.config.activation == "relu":
            return nn.ReLU(inplace=True)
        return nn.LeakyReLU(self.config.leaky_relu_slope, inplace=True)

    def reset_parameters(self) -> None:
        nonlinearity = "relu" if self.config.activation == "relu" else "leaky_relu"
        slope = 0.0 if self.config.activation == "relu" else self.config.leaky_relu_slope
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    module.weight,
                    a=slope,
                    nonlinearity=nonlinearity,
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(
        self,
        dense_evidence: torch.Tensor,
        candidate_metadata: torch.Tensor,
    ) -> torch.Tensor:
        if dense_evidence.ndim != 4 or tuple(dense_evidence.shape[1:]) != (3, 32, 32):
            raise ValueError(
                f"dense_evidence must have shape [B,3,32,32], got {tuple(dense_evidence.shape)}"
            )
        if candidate_metadata.ndim != 2 or candidate_metadata.shape != (
            dense_evidence.shape[0],
            4,
        ):
            raise ValueError(
                "candidate_metadata must have shape [B,4] aligned to dense_evidence"
            )
        features = self.spatial_encoder(dense_evidence).flatten(1)
        return self.representation_head(
            torch.cat([features, candidate_metadata], dim=1)
        )

    def forward(
        self,
        dense_evidence: torch.Tensor,
        candidate_metadata: torch.Tensor,
    ) -> torch.Tensor:
        """Return only the runtime admission logit.

        Auxiliary heads are deliberately excluded from this interface so that
        R1 changes supervision during training without changing the deployed
        P-SRM decision path.
        """

        latent = self.encode(dense_evidence, candidate_metadata)
        return self.admission_head(latent).squeeze(1)

    def forward_with_aux(
        self,
        dense_evidence: torch.Tensor,
        candidate_metadata: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return admission and any enabled train-only auxiliary logits."""

        latent = self.encode(dense_evidence, candidate_metadata)
        output = {"admission": self.admission_head(latent).squeeze(1)}
        if self.presence_head is not None:
            output["presence"] = self.presence_head(latent).squeeze(1)
        if self.localization_quality_head is not None:
            output["localization_quality"] = self.localization_quality_head(latent).squeeze(1)
        return output

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
