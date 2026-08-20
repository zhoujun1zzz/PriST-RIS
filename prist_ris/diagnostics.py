from __future__ import annotations

import math

import torch

from .models import PriSTRIS


SPATIAL_GRADIENT_GROUPS: dict[str, tuple[str, ...]] = {
    "backbone.input": ("backbone.input.",),
    "backbone.stages": (
        "backbone.stages.",
        "backbone.final_refine.",
        "backbone.ris_coordinate_encoder.",
        "backbone.antenna_index_encoder.",
    ),
    "prior_encoder": ("prior_encoder.",),
    "observed_dense_attention": ("observed_dense_attention.",),
    "anchor_feature": ("anchor_feature.",),
    "anchor_heads": ("anchor_heads.",),
}


def parameter_group_gradient_norms(model: PriSTRIS) -> dict[str, dict[str, float | int]]:
    """Summarize finite L2 gradient norms for the spatial parameter groups."""

    result: dict[str, dict[str, float | int]] = {}
    named = tuple(model.named_parameters())
    for group, prefixes in SPATIAL_GRADIENT_GROUPS.items():
        parameters = [
            parameter
            for name, parameter in named
            if name.startswith(prefixes)
        ]
        gradients = [
            parameter.grad.detach().float()
            for parameter in parameters
            if parameter.grad is not None
        ]
        squared_norm = sum(float(gradient.square().sum()) for gradient in gradients)
        norm = math.sqrt(squared_norm)
        if not math.isfinite(norm):
            raise FloatingPointError(f"Non-finite gradient norm for {group}.")
        result[group] = {
            "l2_norm": norm,
            "parameter_tensors": len(parameters),
            "gradient_tensors": len(gradients),
            "parameters": sum(parameter.numel() for parameter in parameters),
        }
    return result
