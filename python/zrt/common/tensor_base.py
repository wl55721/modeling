from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class DType(Enum):
    """Tensor element type with the per-element byte width."""

    FLOAT64 = ("float64", 8)
    FLOAT32 = ("float32", 4)
    FLOAT16 = ("float16", 2)
    BFLOAT16 = ("bfloat16", 2)
    FLOAT8_E4M3 = ("float8_e4m3", 1)
    FLOAT8_E5M2 = ("float8_e5m2", 1)
    FLOAT4_E2M1 = ("float4_e2m1", 0.5)
    INT64 = ("int64", 8)
    INT32 = ("int32", 4)
    INT16 = ("int16", 2)
    INT8 = ("int8", 1)
    INT4 = ("int4", 0.5)
    UINT8 = ("uint8", 1)
    BOOL = ("bool", 1)

    def __init__(self, label: str, byte_size: float):
        self.label = label
        self.byte_size = byte_size

    @classmethod
    def from_str(cls, s: str) -> "DType":
        key = s.strip().lower().replace("torch.", "")
        for dt in cls:
            if dt.label == key:
                return dt
        raise ValueError(f"Unknown dtype: {s!r}")


@dataclass
class TensorBase:
    """Tensor metadata: shape + dtype. No storage."""

    shape: Tuple[int, ...]
    dtype: DType

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def nbytes(self) -> float:
        return self.numel * self.dtype.byte_size

    def __repr__(self) -> str:
        return f"TensorBase(shape={list(self.shape)}, dtype={self.dtype.label})"

    # ---------- CSV parsing helpers ----------

    @staticmethod
    def parse_shapes(s: str) -> List[Tuple[int, ...]]:
        """Parse a CSV shape field like "[1,2,3]; [4,5]" into a list of tuples.

        Empty/whitespace input yields an empty list. Scalar tensors are
        represented as `[]` and parsed as `()`.
        """
        if s is None:
            return []
        text = s.strip()
        if not text:
            return []

        shapes: List[Tuple[int, ...]] = []
        for piece in text.split(";"):
            piece = piece.strip().strip("[]()").strip()
            if not piece:
                shapes.append(())
                continue
            dims = tuple(int(d.strip()) for d in piece.split(",") if d.strip())
            shapes.append(dims)
        return shapes

    @staticmethod
    def parse_dtypes(s: str) -> List[DType]:
        """Parse a CSV dtype field like "float32; bfloat16" into DType values."""
        if s is None:
            return []
        text = s.strip()
        if not text:
            return []
        return [DType.from_str(p) for p in text.split(";") if p.strip()]

    @staticmethod
    def from_parsed(shape_str: str, dtype_str: str) -> List["TensorBase"]:
        """Zip parsed shapes and dtypes into TensorBase objects.

        Lengths must match; raises ValueError otherwise.
        """
        shapes = TensorBase.parse_shapes(shape_str)
        dtypes = TensorBase.parse_dtypes(dtype_str)
        if len(shapes) != len(dtypes):
            raise ValueError(
                f"shape/dtype count mismatch: {len(shapes)} vs {len(dtypes)} "
                f"(shape_str={shape_str!r}, dtype_str={dtype_str!r})"
            )
        return [TensorBase(shape=sh, dtype=dt) for sh, dt in zip(shapes, dtypes)]
