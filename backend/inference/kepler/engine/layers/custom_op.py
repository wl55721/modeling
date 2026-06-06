from __future__ import annotations

from .op_base import OperatorBase, TensorBase


class CustomOp(OperatorBase):
    """通用自定义算子 fallback：当 model_json 中的算子名称无法匹配任何已知类时使用。

    根据 compute_unit 自动选择计算资源：
    - "cube": 使用 Cube FLOPS（矩阵乘等密集计算）
    - "vector": 使用 Vector FLOPS（逐元素操作）
    - "mix": 使用 Cube + Vector 混合（FlashAttention 等）
    - "communication": 使用通信带宽（AllReduce 等）
    """

    def __init__(self, weights: list[TensorBase] | None = None, **attrs):
        compute_unit = attrs.pop("compute_unit", "cube")
        is_vector = compute_unit in ("vector", "sfu")
        super().__init__(weights, is_vector=is_vector, **attrs)
        self.compute_unit = compute_unit
        _static_map = {"cube": 5, "vector": 2, "mix": 3, "sfu": 2, "communication": 10}
        self.static_cost_us = _static_map.get(compute_unit, 5)

    def calc_compute_flops(self):
        compute_flops = eval(self.compute_flops_str, {"__builtins__": {}}, self.cfg)

        if self.compute_unit == "sfu":
            # SFU 算子统一走 SFU 流水线
            self.sfu_compute_flops = compute_flops
        else:
            self.compute_flops = compute_flops

