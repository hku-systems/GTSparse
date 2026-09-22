from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True, slots=True)
class GeometricTemplateReverseEdge:
    runtime: object
    coord_hashmap: Optional[torch.Tensor] = None


@dataclass(frozen=True, slots=True)
class GeometricTemplateMetadata:
    subm_runtime: object | None = None
    reverse_chain: tuple[GeometricTemplateReverseEdge, ...] = ()


def empty_metadata() -> GeometricTemplateMetadata:
    return GeometricTemplateMetadata()


def ensure_metadata(value: object | None) -> GeometricTemplateMetadata:
    if isinstance(value, GeometricTemplateMetadata):
        return value
    return empty_metadata()
