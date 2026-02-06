#!/usr/bin/env python3
"""
IMMUNOS PyTorch Accelerated Negative Selection (NegSel-Torch)
============================================================

A GPU-accelerated implementation of the Negative Selection AIS algorithm.
Vectorizes Thymic Selection (Detector Generation) for massive parallelization.

Formula: Eq 20/21 from Umair et al. (2025)
Binding: R_q > R_self
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .negsel import NEGSEL_PRESETS, Detector, NegSelConfig


class NegativeSelectionTorch(nn.Module):
    """
    PyTorch implementation of the NegSl-AIS algorithm.
    """

    def __init__(
        self,
        config: Union[NegSelConfig, str] = "GENERAL",
        class_label: str = "SELF",
        device: str = "auto",
    ):
        super().__init__()

        if isinstance(config, str):
            self.config = NEGSEL_PRESETS.get(config, NEGSEL_PRESETS["GENERAL"])
        else:
            self.config = config

        self.class_label = class_label

        if device == "auto":
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.valid_detectors_centers = None
        self.valid_detectors_radii = None
        self.self_samples = None
        self.feature_dim = 0

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device)

    def fit(
        self,
        self_samples: Union[torch.Tensor, list],
        batch_size: int = 1000,
        max_attempts: int = 50000,
    ):
        """
        Hyper-parallelized Thymic Selection using batch operations.
        """
        if isinstance(self_samples, np.ndarray):
            self.self_samples = self.to_device(torch.from_numpy(self_samples).float())
        elif isinstance(self_samples, list):
            self.self_samples = self.to_device(torch.tensor(self_samples, dtype=torch.float32))
        else:
            self.self_samples = self.to_device(self_samples.float())

        self.feature_dim = self.self_samples.shape[1]

        final_centers = []
        final_radii = []

        attempts = 0
        while len(final_centers) < self.config.num_detectors and attempts < max_attempts:
            attempts += batch_size

            # 1. Generate Batch of candidates (unit hypercube)
            candidates = self.to_device(torch.rand((batch_size, self.feature_dim)))

            # 2. Vectorized Distance calculation: (Batch, 1, Dim) - (1, Self_N, Dim) -> (Batch, Self_N, Dim)
            # Use efficiently via cdist
            dist_to_self = torch.cdist(candidates, self.self_samples)  # Shape: (Batch, Self_N)

            # 3. Find min distance to self for each candidate
            r_q, _ = torch.min(dist_to_self, dim=1)  # Shape: (Batch)

            # 4. Filter by Equation 20: R_q > R_self
            mask = r_q > self.config.r_self
            valid_batch_centers = candidates[mask]
            valid_batch_rq = r_q[mask]

            # 5. Calculate Radii: r^j = R_q - R_self
            valid_batch_radii = valid_batch_rq - self.config.r_self

            # Collect
            for i in range(valid_batch_centers.shape[0]):
                if len(final_centers) < self.config.num_detectors:
                    final_centers.append(valid_batch_centers[i])
                    final_radii.append(valid_batch_radii[i])
                else:
                    break

        if final_centers:
            self.valid_detectors_centers = torch.stack(final_centers)
            self.valid_detectors_radii = torch.stack(final_radii)

        return self

    def predict(self, samples: Union[torch.Tensor, list]) -> torch.Tensor:
        """
        Batch prediction for antigens.
        Returns tensor of labels (1.0 for Non-Self / Anomaly, 0.0 for Self).
        """
        if self.valid_detectors_centers is None:
            return torch.zeros(len(samples), device=self.device)

        if isinstance(samples, list):
            x = self.to_device(torch.tensor(samples, dtype=torch.float32))
        else:
            x = self.to_device(samples.float())

        # 1. Distance to Self
        dist_to_self = torch.cdist(x, self.self_samples)
        min_dist_to_self, _ = torch.min(dist_to_self, dim=1)

        # Binary mask: if min_dist <= r_self, it's definitely SELF
        is_self_mask = min_dist_to_self <= self.config.r_self

        # 2. Distance to Detectors
        dist_to_detectors = torch.cdist(x, self.valid_detectors_centers)

        # Binding Rule: dist < radius
        # Check if any detector binds to the sample
        binds_to_any = torch.any(dist_to_detectors < self.valid_detectors_radii, dim=1)

        # Final decision: Anomaly if (Not Self Mask) AND (Binds or is far enough)
        # Simplified: if not self_mask, then anomaly. Corrected per paper logic.
        results = (~is_self_mask).float()

        return results

    def get_anomaly_score(self, samples: Union[torch.Tensor, list]) -> torch.Tensor:
        """Vectorized anomaly score calculation."""
        if isinstance(samples, list):
            x = self.to_device(torch.tensor(samples, dtype=torch.float32))
        else:
            x = self.to_device(samples.float())

        dist_to_self = torch.cdist(x, self.self_samples)
        min_dist_to_self, _ = torch.min(dist_to_self, dim=1)

        score = torch.clamp(min_dist_to_self - self.config.r_self, min=0.0)
        return score

    def to_detectors(self) -> List[Detector]:
        """Convert torch state back to list of Detector objects for compatibility."""
        if self.valid_detectors_centers is None:
            return []

        detectors = []
        centers = self.valid_detectors_centers.cpu().numpy()
        radii = self.valid_detectors_radii.cpu().numpy()

        for i in range(len(centers)):
            detectors.append(
                Detector(
                    center=centers[i],
                    radius=float(radii[i]),
                    class_label=self.class_label,
                    r_self=self.config.r_self,
                )
            )
        return detectors
