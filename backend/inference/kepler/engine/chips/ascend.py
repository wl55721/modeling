from __future__ import annotations

from .config import AIChipConfig


def _derive_ascend_flops(cube_cnt: int, vector_cnt: int, sfu_cnt: int, freq: float):
    gf_cube = cube_cnt * freq * (16 ** 3) * 2
    gf_vector = vector_cnt * freq * (16 ** 2)
    gf_sfu = sfu_cnt * freq * (16 ** 2)
    return gf_cube, gf_vector, gf_sfu


class A3_PODConfig(AIChipConfig):
    """Ascend A3 POD 硬件配置。

    __init__ 直接接收 JSON 文件内容作为参数，
    根据 JSON 字段动态添加对应属性。
    Cube/Vector/SFU 算力由 core_cnt * freq 推导。
    """

    def __init__(self, **kwargs):
        name = kwargs.get("name", "A3_POD")
        chip = kwargs.get("chip", "Ascend 910B3")
        vendor = kwargs.get("vendor", "HUAWEI")
        spec_memory_size = kwargs.get("spec_memory_size", 64)
        cube_core_cnt = kwargs.get("cube_core_cnt", 24)
        vector_core_cnt = kwargs.get("vector_core_cnt", 48)
        sfu_core_cnt = kwargs.get("sfu_core_cnt", 12)
        cube_freq = kwargs.get("cube_freq", 1.8)
        spec_bw_memory = kwargs.get("spec_bw_memory", 1.6)
        spec_comm_intra = kwargs.get("spec_comm_intra", 196)
        spec_comm_inter = kwargs.get("spec_comm_inter", 50)
        spec_comm_bwsio = kwargs.get("spec_comm_bwsio", 224)
        spec_l2cache_size = kwargs.get("spec_l2cache_size", 192)
        memory_noise = kwargs.get("memory_noise", 5)
        compute_ratio = kwargs.get("compute_ratio", 0.7)
        bw_gmem_ratio = kwargs.get("bw_gmem_ratio", 0.6)
        comm_intra_ratio = kwargs.get("comm_intra_ratio", 0.5)
        comm_inter_ratio = kwargs.get("comm_inter_ratio", 0.5)
        comm_bwsio_ratio = kwargs.get("comm_bwsio_ratio", 0.8)
        l2_cache_ratio = kwargs.get("l2_cache_ratio", 1.0)
        superpod_limit = kwargs.get("superpod_limit", 768)
        bwsio_limit = kwargs.get("bwsio_limit", 2)

        gf_cube, gf_vector, gf_sfu = _derive_ascend_flops(
            cube_core_cnt, vector_core_cnt, sfu_core_cnt, cube_freq)

        super().__init__(
            name=name,
            chip=chip,
            vendor=vendor,
            spec_cube_fp16=gf_cube,
            spec_vect_fp16=gf_vector,
            spec_sfu_fp16=gf_sfu,
            spec_memory_size=spec_memory_size,
            spec_bw_memory=spec_bw_memory * 1024,
            spec_l2cache_size=spec_l2cache_size,
            memory_noise=memory_noise,
            spec_comm_intra=spec_comm_intra,
            spec_comm_inter=spec_comm_inter,
            spec_comm_bwsio=spec_comm_bwsio,
            compute_ratio=compute_ratio,
            bw_gmem_ratio=bw_gmem_ratio,
            comm_intra_ratio=comm_intra_ratio,
            comm_inter_ratio=comm_inter_ratio,
            comm_bwsio_ratio=comm_bwsio_ratio,
            l2_cache_ratio=l2_cache_ratio,
            superpod_limit=superpod_limit,
            bwsio_limit=bwsio_limit,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "A3_PODConfig":
        return cls(**d)


AIChipConfig.register("A3_POD", A3_PODConfig)


class CustomAscendConfig(AIChipConfig):
    """通用 Ascend 芯片配置。

    处理前端回传的自定义 Ascend JSON 内容。
    与 A3_PODConfig 逻辑相同，但无硬编码默认值 ——
    所有字段均由 JSON 提供，缺失则使用合理默认值。
    """

    def __init__(self, **kwargs):
        name = kwargs.get("name", "CustomAscend")
        chip = kwargs.get("chip", "Ascend")
        vendor = kwargs.get("vendor", "HUAWEI")
        spec_memory_size = kwargs.get("spec_memory_size", 0)
        cube_core_cnt = kwargs.get("cube_core_cnt", 0)
        vector_core_cnt = kwargs.get("vector_core_cnt", 0)
        sfu_core_cnt = kwargs.get("sfu_core_cnt", 0)
        cube_freq = kwargs.get("cube_freq", 0.0)
        spec_bw_memory = kwargs.get("spec_bw_memory", 0.0)
        spec_comm_intra = kwargs.get("spec_comm_intra", 0)
        spec_comm_inter = kwargs.get("spec_comm_inter", 0)
        spec_comm_bwsio = kwargs.get("spec_comm_bwsio", 0)
        spec_l2cache_size = kwargs.get("spec_l2cache_size", 0)
        memory_noise = kwargs.get("memory_noise", 0)
        compute_ratio = kwargs.get("compute_ratio", 1.0)
        bw_gmem_ratio = kwargs.get("bw_gmem_ratio", 1.0)
        comm_intra_ratio = kwargs.get("comm_intra_ratio", 1.0)
        comm_inter_ratio = kwargs.get("comm_inter_ratio", 1.0)
        comm_bwsio_ratio = kwargs.get("comm_bwsio_ratio", 1.0)
        l2_cache_ratio = kwargs.get("l2_cache_ratio", 1.0)
        superpod_limit = kwargs.get("superpod_limit", 1)
        bwsio_limit = kwargs.get("bwsio_limit", 0)

        gf_cube, gf_vector, gf_sfu = _derive_ascend_flops(
            cube_core_cnt, vector_core_cnt, sfu_core_cnt, cube_freq)

        super().__init__(
            name=name,
            chip=chip,
            vendor=vendor,
            spec_cube_fp16=gf_cube,
            spec_vect_fp16=gf_vector,
            spec_sfu_fp16=gf_sfu,
            spec_memory_size=spec_memory_size,
            spec_bw_memory=spec_bw_memory * 1024,
            spec_l2cache_size=spec_l2cache_size,
            memory_noise=memory_noise,
            spec_comm_intra=spec_comm_intra,
            spec_comm_inter=spec_comm_inter,
            spec_comm_bwsio=spec_comm_bwsio,
            compute_ratio=compute_ratio,
            bw_gmem_ratio=bw_gmem_ratio,
            comm_intra_ratio=comm_intra_ratio,
            comm_inter_ratio=comm_inter_ratio,
            comm_bwsio_ratio=comm_bwsio_ratio,
            l2_cache_ratio=l2_cache_ratio,
            superpod_limit=superpod_limit,
            bwsio_limit=bwsio_limit,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "CustomAscendConfig":
        return cls(**d)


AIChipConfig.register("CustomAscend", CustomAscendConfig)
