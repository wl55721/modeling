from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DType(Enum):
    """张量数据类型枚举。"""

    FP64 = (8, "fp64")
    FP32 = (4, "fp32")
    FP16 = (2, "fp16")
    BF16 = (2, "bf16")
    FP8 = (1, "fp8")
    FP4 = (0.5, "fp4")
    INT8 = (1, "int8")
    INT4 = (0.5, "int4")
    INT64 = (8, "int64")
    INT32 = (4, "int32")

    @property
    def bytes(self) -> int:
        return self.value[0]

    @property
    def bits(self) -> int:
        return int(self.bytes * 8)

    @classmethod
    def from_str(cls, s: str) -> "DType":
        mapping = {
            "fp32": cls.FP32, "fp16": cls.FP16, "bf16": cls.BF16,
            "fp8": cls.FP8, "fp4": cls.FP4,
            "int8": cls.INT8, "int4": cls.INT4,
            "int32": cls.INT32, "int64": cls.INT64,
        }
        return mapping.get(str(s).lower(), cls.FP16)


@dataclass
class TensorBase:
    """张量基类 —— 描述张量的形状与数据类型。"""

    shape: list[int]
    dtype: DType = DType.FP16
    name: str = ""

    @property
    def numel(self) -> int:
        if not self.shape:
            return 0
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def nbytes(self) -> int:
        return int(self.numel * self.dtype.bytes)

    @classmethod
    def from_spec(cls, name_or_dict: str | dict, shape_str: str = "",
                  dtype_str: str = "fp16", context: dict | None = None) -> "TensorBase":
        """从规格字典或 (name, shape, dtype) 三元组解析 TensorBase。

        参数字典格式: {"name": "x", "shape": "[B, S, 4096]", "dtype": "fp16"}
        shape_str 支持:
          - "[dim1, dim2, ...]" 数组形式，元素可为变量名或数字
          - "4096" 标量形式
        """
        if isinstance(name_or_dict, dict):
            d = name_or_dict
            return cls.from_spec(d.get("name", ""), d.get("shape", ""),
                                 d.get("dtype", "fp16"), context=context)
        try:
            shape = cls._parse_shape(shape_str, context or {})
        except KeyError as e:
            raise KeyError(f"name={name_or_dict} 解析 shape 时出错: {e}")
        dtype = DType.from_str(dtype_str)
        return cls(shape=shape, dtype=dtype, name=name_or_dict)

    @staticmethod
    def _parse_shape(shape_str: str, context: dict) -> list[int]:
        s = shape_str.strip()
        if s.startswith("[") and s.endswith("]"):
            parts = [p.strip() for p in s[1:-1].split(",") if p.strip()]
        else:
            parts = [s]

        result = []
        for p in parts:
            p = p.strip()
            if p == "-":
                continue
            try:
                result.append(int(p))
            except ValueError:
                if any(op in p for op in ("//", "*", "+", "-")):
                    try:
                        val = eval(p, {"__builtins__": {}}, context)
                        result.append(int(val))
                        continue
                    except Exception:
                        raise KeyError(f"shape 表达式 '{p}' 求值失败")
                resolved = context.get(p)
                if resolved is None:
                    text_cfg = context.get("text_config", {})
                    resolved = text_cfg.get(p) if isinstance(text_cfg, dict) else None
                if resolved is None:
                    raise KeyError(f"shape 变量 '{p}' 在上下文中未找到")
                result.append(int(resolved))
        return result

    def reshape(self, new_shape: list[int]) -> TensorBase:
        if not self.shape or not new_shape:
            raise ValueError("shape 和 new_shape 不能为空")
        old_numel = self.numel
        new_numel = 1
        for d in new_shape:
            new_numel *= d
        if new_numel != old_numel:
            raise ValueError(
                f"reshape 前后元素数不一致: {old_numel} vs {new_numel}"
            )
        self.shape = list(new_shape)
        return self

    def to(self, dtype: DType) -> TensorBase:
        self.dtype = dtype
        return self

    def clone(self) -> TensorBase:
        return TensorBase(shape=list(self.shape), dtype=self.dtype, name=self.name)

    def __repr__(self) -> str:
        return (
            f"TensorBase(shape={self.shape}, dtype={self.dtype.value}, "
            f"name={self.name!r}, numel={self.numel})"
        )
