from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from zrt.common.chip_spec import ChipSpec

class PDDisaggregation(Enum):
    PREFILL = auto()
    DECODE = auto()

    @classmethod
    def from_str(cls, s: str) -> "PDDisaggregation":
        key = s.strip().lower()
        for member in cls:
            if member.name.lower() == key:
                return member
        raise ValueError(f"Unknown Disaggregation: {s!r}")

@dataclass
class RuntimeConfig():
    # chip
    chip_spec: ChipSpec

    # Parallel
    parallel_config:"ParallelConfig"
    
    # MTP
    speculative_num_steps:int
    speculative_accept_rate:float

    # Prefix Cache
    disable_radix_attention:bool

    # Chunked Prefill
    chunked_prefill_size:int


@dataclass
class ParallelConfig():
    world_size:int
    # PD Disaggregation
    disaggregation:PDDisaggregation
    # TP 
    tp_size:int
    attn_tp:int
    moe_tp:Optional[int]
    # dp
    dp_size:int
    moe_dp:Optional[int]
    # ep
    ep_size:int
    external_shared_expert_size:int
    # TODO PP
