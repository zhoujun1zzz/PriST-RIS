from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


MODEL_DISPLAY_NAME = "PriST-RIS"
MODEL_KEYS = (
    "prist_ris_a",
    "prist_ris_b",
    "prist_ris_c",
    "prist_ris_full",
)
MODEL_ALIASES = {
    "v3_a_structured_progressive": "prist_ris_a",
    "v3_b_prior_guided": "prist_ris_b",
    "v3_c_prior_crossattn": "prist_ris_c",
    "v3_full": "prist_ris_full",
}
OBSERVED_RIS_INDICES = tuple(range(0, 256, 8))
GRID_HEIGHT = GRID_WIDTH = 16
ANTENNAS = 64
COMPLEX_LAYOUT = "grouped"
METRIC_CONTRACT = "sample_mean_linear_then_db"


def canonical_model_key(key: str) -> str:
    normalized = MODEL_ALIASES.get(key, key)
    if normalized not in MODEL_KEYS:
        raise ValueError(f"Unknown PriST-RIS model key {key!r}; choose from {MODEL_KEYS}.")
    return normalized


@dataclass(frozen=True)
class DataSemantics:
    domain: str
    input_shape: tuple[int | str, ...]
    target_shape: tuple[int | str, ...]
    obs_ris_index: tuple[int, ...]
    obs_time_index: tuple[int, ...]
    query_time: tuple[int, ...]
    complex_layout: str = COMPLEX_LAYOUT
    grid_shape: tuple[int, int] = (GRID_HEIGHT, GRID_WIDTH)
    grid_index_rule: str = "index=16*row+column"
    mobility_scope: str = "within_sample_2_to_6"
    metric_contract: str = METRIC_CONTRACT

    @classmethod
    def for_domain(cls, domain: str) -> "DataSemantics":
        if domain == "quasi":
            return cls(
                domain=domain,
                input_shape=("B", 1, 32, 64, 2),
                target_shape=("B", 1, 256, 64, 2),
                obs_ris_index=OBSERVED_RIS_INDICES,
                obs_time_index=(0,),
                query_time=(0,),
            )
        if domain == "mobility":
            return cls(
                domain=domain,
                input_shape=("B", 2, 32, 64, 2),
                target_shape=("B", 6, 256, 64, 2),
                obs_ris_index=OBSERVED_RIS_INDICES,
                obs_time_index=(0, 1),
                query_time=(0, 1, 2, 3, 4, 5),
            )
        raise ValueError("domain must be 'quasi' or 'mobility'.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def stable_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ris_index_to_grid(index: int) -> tuple[int, int]:
    if not 0 <= index < 256:
        raise ValueError("RIS index must be in [0, 255].")
    return divmod(index, GRID_WIDTH)
