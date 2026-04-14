from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Optional

from zrt.common.tensor_base import DType


class Vendor(Enum):
    NVIDIA = auto()
    ASCEND = auto()

    @classmethod
    def from_str(cls, s: str) -> "Vendor":
        key = s.strip().upper()
        for m in cls:
            if m.name == key:
                return m
        raise ValueError(f"Unknown vendor: {s!r}")


def _parse_tflops_table(raw: Optional[Dict[str, float]]) -> Dict[DType, float]:
    if not raw:
        return {}
    return {DType.from_str(k): float(v) for k, v in raw.items()}


@dataclass
class InterconnectSpec:
    """Inter-chip link (NVLink / HCCS / PCIe / ...). Bandwidth in GB/s, uni-directional."""
    name: str
    bandwidth_gbps: float


@dataclass
class ChipSpec:
    """Vendor-neutral accelerator description.

    Compute is split into cube (tensor core / cube unit) and vector
    (cuda core / vector unit) peak throughput, per dtype, in TFLOPS.
    """
    name: str
    vendor: Vendor

    cube_tflops: Dict[DType, float]
    vector_tflops: Dict[DType, float] = field(default_factory=dict)

    hbm_bandwidth_gbps: float = 0.0
    hbm_capacity_gb: float = 0.0

    l2_capacity_mb: Optional[float] = None
    sram_capacity_mb: Optional[float] = None

    # Keyed by role ("scaleup", "host", ...)
    interconnects: Dict[str, InterconnectSpec] = field(default_factory=dict)

    # Vendor-specific extras that don't fit the abstraction
    extras: Dict[str, Any] = field(default_factory=dict)

    def peak_tflops(self, dtype: DType, unit: str = "cube") -> float:
        table = self.cube_tflops if unit == "cube" else self.vector_tflops
        if dtype not in table:
            raise KeyError(f"{self.name} has no {unit} TFLOPS entry for {dtype}")
        return table[dtype]

    def interconnect(self, role: str) -> InterconnectSpec:
        if role not in self.interconnects:
            raise KeyError(
                f"{self.name} has no {role!r} interconnect; "
                f"known: {sorted(self.interconnects)}"
            )
        return self.interconnects[role]

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "ChipSpec":
        interconnects = {
            role: InterconnectSpec(name=v["name"], bandwidth_gbps=float(v["bandwidth_gbps"]))
            for role, v in (data.get("interconnects") or {}).items()
        }
        return cls(
            name=name,
            vendor=Vendor.from_str(data["vendor"]),
            cube_tflops=_parse_tflops_table(data.get("cube_tflops")),
            vector_tflops=_parse_tflops_table(data.get("vector_tflops")),
            hbm_bandwidth_gbps=float(data.get("hbm_bandwidth_gbps", 0.0)),
            hbm_capacity_gb=float(data.get("hbm_capacity_gb", 0.0)),
            l2_capacity_mb=data.get("l2_capacity_mb"),
            sram_capacity_mb=data.get("sram_capacity_mb"),
            interconnects=interconnects,
            extras=dict(data.get("extras") or {}),
        )


# repo_root/configs/chips.yaml — user-editable catalog outside of python/
_DEFAULT_SPEC_FILE = Path(__file__).resolve().parents[3] / "configs" / "chips.yaml"


def _load_catalog(path: Path) -> Dict[str, Any]:
    import yaml
    return yaml.safe_load(path.read_text()) or {}


def load_chip_spec(name: str, spec_file: Optional[Path] = None) -> ChipSpec:
    """Load a ChipSpec by name from the combined catalog file."""
    path = Path(spec_file) if spec_file else _DEFAULT_SPEC_FILE
    catalog = _load_catalog(path)
    if name not in catalog:
        raise KeyError(f"Unknown chip {name!r}; known: {sorted(catalog)}")
    return ChipSpec.from_dict(name, catalog[name])


def list_chip_specs(spec_file: Optional[Path] = None) -> list[str]:
    path = Path(spec_file) if spec_file else _DEFAULT_SPEC_FILE
    return sorted(_load_catalog(path).keys())
