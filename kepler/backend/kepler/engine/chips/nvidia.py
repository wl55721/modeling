from __future__ import annotations

from .config import AIChipConfig


class B300_ServerConfig(AIChipConfig):
    """B300 Server (8 卡) 硬件配置。

    __init__ 直接接收 JSON 文件内容作为参数，
    B300 的 spec_cube/vect/sfu_fp16 由 JSON 直接提供。
    """

    def __init__(self, **kwargs):
        name = kwargs.get("name", "B300_Server")
        chip = kwargs.get("chip", "NVIDIA B300")
        vendor = kwargs.get("vendor", "NVIDIA")
        spec_memory_size = kwargs.get("spec_memory_size", 288)
        spec_cube_fp16 = kwargs.get("spec_cube_fp16", 2250)
        spec_vect_fp16 = kwargs.get("spec_vect_fp16", 160)
        spec_sfu_fp16 = kwargs.get("spec_sfu_fp16", 22)
        spec_bw_memory = kwargs.get("spec_bw_memory", 8.0)
        spec_comm_intra = kwargs.get("spec_comm_intra", 900)
        spec_comm_inter = kwargs.get("spec_comm_inter", 800)
        spec_l2cache_size = kwargs.get("spec_l2cache_size", 50)
        memory_noise = kwargs.get("memory_noise", 5)
        compute_ratio = kwargs.get("compute_ratio", 0.6)
        bw_gmem_ratio = kwargs.get("bw_gmem_ratio", 0.8)
        comm_intra_ratio = kwargs.get("comm_intra_ratio", 0.8)
        comm_inter_ratio = kwargs.get("comm_inter_ratio", 0.8)
        l2_cache_ratio = kwargs.get("l2_cache_ratio", 1.0)
        superpod_limit = kwargs.get("superpod_limit", 8)
        bwsio_limit = kwargs.get("bwsio_limit", 1)

        super().__init__(
            name=name,
            chip=chip,
            vendor=vendor,
            spec_cube_fp16=spec_cube_fp16 * 1000,
            spec_vect_fp16=spec_vect_fp16 * 1000,
            spec_sfu_fp16=spec_sfu_fp16 * 1000,
            spec_memory_size=spec_memory_size,
            spec_bw_memory=spec_bw_memory * 1024,
            spec_l2cache_size=spec_l2cache_size,
            memory_noise=memory_noise,
            spec_comm_intra=spec_comm_intra,
            spec_comm_inter=spec_comm_inter,
            compute_ratio=compute_ratio,
            bw_gmem_ratio=bw_gmem_ratio,
            comm_intra_ratio=comm_intra_ratio,
            comm_inter_ratio=comm_inter_ratio,
            l2_cache_ratio=l2_cache_ratio,
            superpod_limit=superpod_limit,
            bwsio_limit=bwsio_limit,
        )

    @classmethod
    def from_dict(cls, d: dict) -> B300_ServerConfig:
        return cls(**d)


AIChipConfig.register("B300_Server", B300_ServerConfig)


class B300_PODConfig(AIChipConfig):
    """B300 POD (多机集群) 硬件配置。

    与 Server 的主要区别：机间带宽更小、集群规模更大。
    """

    def __init__(self, **kwargs):
        name = kwargs.get("name", "B300_POD")
        chip = kwargs.get("chip", "NVIDIA B300")
        vendor = kwargs.get("vendor", "NVIDIA")
        spec_memory_size = kwargs.get("spec_memory_size", 288)
        spec_cube_fp16 = kwargs.get("spec_cube_fp16", 2250)
        spec_vect_fp16 = kwargs.get("spec_vect_fp16", 160)
        spec_sfu_fp16 = kwargs.get("spec_sfu_fp16", 22)
        spec_bw_memory = kwargs.get("spec_bw_memory", 8.0)
        spec_comm_intra = kwargs.get("spec_comm_intra", 900)
        spec_comm_inter = kwargs.get("spec_comm_inter", 100)
        spec_l2cache_size = kwargs.get("spec_l2cache_size", 50)
        memory_noise = kwargs.get("memory_noise", 5)
        compute_ratio = kwargs.get("compute_ratio", 0.6)
        bw_gmem_ratio = kwargs.get("bw_gmem_ratio", 0.8)
        comm_intra_ratio = kwargs.get("comm_intra_ratio", 0.8)
        comm_inter_ratio = kwargs.get("comm_inter_ratio", 0.8)
        l2_cache_ratio = kwargs.get("l2_cache_ratio", 1.0)
        superpod_limit = kwargs.get("superpod_limit", 8)
        bwsio_limit = kwargs.get("bwsio_limit", 1)

        super().__init__(
            name=name,
            chip=chip,
            vendor=vendor,
            spec_cube_fp16=spec_cube_fp16 * 1000,
            spec_vect_fp16=spec_vect_fp16 * 1000,
            spec_sfu_fp16=spec_sfu_fp16 * 1000,
            spec_memory_size=spec_memory_size,
            spec_bw_memory=spec_bw_memory * 1024,
            spec_l2cache_size=spec_l2cache_size,
            memory_noise=memory_noise,
            spec_comm_intra=spec_comm_intra,
            spec_comm_inter=spec_comm_inter,
            compute_ratio=compute_ratio,
            bw_gmem_ratio=bw_gmem_ratio,
            comm_intra_ratio=comm_intra_ratio,
            comm_inter_ratio=comm_inter_ratio,
            l2_cache_ratio=l2_cache_ratio,
            superpod_limit=superpod_limit,
            bwsio_limit=bwsio_limit,
        )

    @classmethod
    def from_dict(cls, d: dict) -> B300_PODConfig:
        return cls(**d)


AIChipConfig.register("B300_POD", B300_PODConfig)
