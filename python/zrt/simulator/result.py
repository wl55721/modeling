"""SimResult: per-operator simulation output."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimResult:
    """Simulation result for a single operator node.

    Fields
    ------
    op_node_id          : matches OpNode.id in the originating graph
    latency_us          : total estimated execution time (μs)
    compute_us          : compute-bound time = flops / peak_flops (μs)
    memory_us           : memory-bound time  = bytes  / hbm_bw    (μs)
    flops               : floating-point (or integer) operations count
    read_bytes          : bytes read from HBM
    write_bytes         : bytes written to HBM
    arithmetic_intensity: flops / (read_bytes + write_bytes)  [ops/byte]
    bound               : "compute" | "memory" | "latency"
    hw_utilization      : fraction of peak compute actually used  [0-1]
    backend             : name of the simulator backend that produced this
    confidence          : estimate quality  [0-1]; roofline = 0.3, profDB = 0.9
    """

    op_node_id: str = ""
    latency_us: float = 0.0
    compute_us: float = 0.0
    memory_us: float = 0.0
    flops: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    arithmetic_intensity: float = 0.0
    bound: str = ""
    hw_utilization: float = 0.0
    backend: str = ""
    confidence: float = 0.0
    
    base_compute_us: float = 0.0
    base_memory_us: float = 0.0
    base_latency_us: float = 0.0
    
    saved_activation_bytes: int = 0
    activation_memory_us: float = 0.0
    
    checkpoint_activation_bytes: int = 0
    checkpoint_memory_us: float = 0.0
    
    recompute_flops: int = 0
    recompute_read_bytes: int = 0
    recompute_write_bytes: int = 0
    recompute_compute_us: float = 0.0
    recompute_memory_us: float = 0.0
    recompute_latency_us: float = 0.0
    
    mega_moe_dispatch_us: float = 0.0
    mega_moe_combine_us: float = 0.0
    mega_moe_exposed_comm_us: float = 0.0
    mega_moe_hidden_comm_us: float = 0.0
    mega_moe_internal_comm_us: float = 0.0
    mega_moe_effective_waves: int = 0
    mega_moe_dispatch_bytes: int = 0
    mega_moe_combine_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        return self.read_bytes + self.write_bytes

    @property
    def latency_ms(self) -> float:
        return self.latency_us / 1_000.0

    def __repr__(self) -> str:
        return (
            f"SimResult({self.op_node_id}, {self.latency_us:.2f}μs, "
            f"{self.bound}-bound, util={self.hw_utilization:.1%}, "
            f"backend={self.backend})"
        )
