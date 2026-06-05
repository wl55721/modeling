from __future__ import annotations

import json
import os


class AIChipConfig:
    """芯片 / 服务器 / POD 硬件规格配置。"""

    def __init__(
        self,
        *,
        name: str = "",
        chip: str = "",
        vendor: str = "",
        # 计算
        spec_cube_fp16: float = 0, # GFLOPS
        spec_vect_fp16: float = 0, # GFLOPS
        spec_sfu_fp16: float = 0, # GFLOPS
        # 显存
        spec_memory_size: int = 0, # GB
        spec_bw_memory: float = 0, # GB/s
        spec_l2cache_size: float = 0, # MB
        memory_noise: float = 0,
        # 通信
        spec_comm_intra: float = 0, # GB/s
        spec_comm_inter: float = 0, # GB/s
        spec_comm_bwsio: float = 0,
        # 效率折算比
        compute_ratio: float = 1.0,
        bw_gmem_ratio: float = 1.0,
        comm_intra_ratio: float = 1.0,
        comm_inter_ratio: float = 1.0,
        comm_bwsio_ratio: float = 1.0,
        l2_cache_ratio: float = 1.0,
        # 规模限制
        superpod_limit: int = 1,
        bwsio_limit: int = 1,
    ):
        self.name = name
        self.chip = chip
        self.vendor = vendor
        self.spec_cube_fp16 = spec_cube_fp16
        self.spec_vect_fp16 = spec_vect_fp16
        self.spec_sfu_fp16 = spec_sfu_fp16
        self.spec_memory_size = spec_memory_size
        self.spec_bw_memory = spec_bw_memory
        self.spec_l2cache_size = spec_l2cache_size
        self.memory_noise = memory_noise
        self.spec_comm_intra = spec_comm_intra
        self.spec_comm_inter = spec_comm_inter
        self.spec_comm_bwsio = spec_comm_bwsio
        self.compute_ratio = compute_ratio
        self.bw_gmem_ratio = bw_gmem_ratio
        self.comm_intra_ratio = comm_intra_ratio
        self.comm_inter_ratio = comm_inter_ratio
        self.comm_bwsio_ratio = comm_bwsio_ratio
        self.l2_cache_ratio = l2_cache_ratio
        self.superpod_limit = superpod_limit
        self.bwsio_limit = bwsio_limit

        self.fused_optim = False

    @property
    def flops_per_sec(self) -> float:
        raw_gf = self.spec_cube_fp16 + self.spec_vect_fp16 + self.spec_sfu_fp16
        return raw_gf * 1e9 * self.compute_ratio

    @property
    def mem_bw_bytes_per_sec(self) -> float:
        return self.spec_bw_memory * 1e9 * self.bw_gmem_ratio

    @property
    def intra_bw_bytes_per_sec(self) -> float:
        return self.spec_comm_intra * 1e9 * self.comm_intra_ratio

    @property
    def inter_bw_bytes_per_sec(self) -> float:
        return self.spec_comm_inter * 1e9 * self.comm_inter_ratio

    def print_detail(self) -> None:
        """打印芯片规格详情。"""
        print(f"[{self.name}] {self.chip}")
        print(f"  计算: Cube={self.spec_cube_fp16:.0f}G  Vect={self.spec_vect_fp16:.0f}G  SFU={self.spec_sfu_fp16:.0f}G  (GFLOPS)")
        print(f"  显存: {self.spec_memory_size}GB  BW={self.spec_bw_memory}GB/s  L2={self.spec_l2cache_size}MB  noise={self.memory_noise}GB")
        print(f"  通信: intra={self.spec_comm_intra}GB/s  inter={self.spec_comm_inter}GB/s  bw_sio={self.spec_comm_bwsio}GB/s")
        print(f"  折算: compute={self.compute_ratio}  bw_gmem={self.bw_gmem_ratio}  "
              f"intra={self.comm_intra_ratio}  inter={self.comm_inter_ratio}")
        print(f"  规模: superpod_limit={self.superpod_limit}  bwsio_limit={self.bwsio_limit}")


    _subclass_registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str, subclass: type) -> None:
        cls._subclass_registry[name] = subclass

    @classmethod
    def from_dict(cls, d: dict) -> AIChipConfig:
        if cls is not AIChipConfig:
            return cls(**d)
        name = d.get("name", "")
        # Only route to Ascend config if cube_core_cnt is explicitly set (>0)
        has_ascend_cores = d.get("cube_core_cnt", 0) > 0
        # Check subclass registry by name first
        subclass = cls._subclass_registry.get(name)
        if subclass:
            return subclass(**d)
        if has_ascend_cores and name == "A3_POD":
            return cls._subclass_registry.get("A3_POD", AIChipConfig)(**d)
        if has_ascend_cores:
            return cls._subclass_registry.get("CustomAscend", AIChipConfig)(**d)
        return cls(**d)

    @classmethod
    def from_json(cls, path: str) -> AIChipConfig:
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_name(cls, name: str, data_dir: str = "") -> AIChipConfig:
        if not data_dir:
            data_dir = os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "data", "hardwares"
            )
        # 先按文件名精确匹配
        path = os.path.join(data_dir, f"{name}.json")
        if os.path.exists(path):
            return cls.from_json(path)
        # 再按内部 name 字段搜索
        for fname in os.listdir(data_dir):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(data_dir, fname)
            with open(fpath) as f:
                d = json.load(f)
            if d.get("name") == name:
                return cls.from_dict(d)
        raise FileNotFoundError(f"芯片 '{name}' 未找到")
