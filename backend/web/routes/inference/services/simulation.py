from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timezone

from fastapi import HTTPException

from ..schemas import (
    SimulateRequest, SimulateResponse, SimulateSingleResult,
    SimulateMultiResponse, PerGPUResult, OperatorResult,
    OperatorStatistics, LayerResultPerRank, RankResult, WorkloadEntry,
    TensorInfo,
)
from backend.inference.kepler.engine.executor import execute_model, OperatorExecuteResult
from backend.inference.kepler.engine.model_config import ModelConfig
from backend.inference.kepler.engine.chips.config import AIChipConfig
from backend.utils.log import logger

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "inference", "data")


def _fmt_context(ctx: dict) -> str:
    return f"B={ctx['B']} S={ctx['S']} h={ctx['hidden_dim']} " \
           f"tp={ctx['tp_size']} dp={ctx['dp_size']} dtype={ctx['dtype_str']}"


class SimulationService:

    def simulate(self, req: SimulateRequest) -> SimulateMultiResponse:
        model_config = self._load_model(req.model_name, req.model_json)

        hw_configs: list[tuple[str, str | dict]] = [
            (h.name, h.config) for h in req.hardwares
        ]
        if not hw_configs and req.hardware_name:
            hw_configs.append(("default", req.hardware_name))

        wl = req.workloads[0] if req.workloads else WorkloadEntry()
        context = self._build_context(wl, req.hf_config_json)

        hw_names = [name for name, _ in hw_configs]
        logger.info("simulation start — operators=%d hw=%s context=%s",
                   len(model_config.operators), hw_names, _fmt_context(context))

        results: list[SimulateSingleResult] = []
        for hw_name, hw_config in hw_configs:
            chip = self._load_chip(hw_config)
            op_results = execute_model(model_config, copy.deepcopy(context), chip=chip)
            response = self._build_response(req, model_config, op_results, chip, hw_name)
            results.append(SimulateSingleResult(
                hardware_name=hw_name,
                result=response,
            ))

        logger.info("simulation done — %d hw results", len(results))
        return SimulateMultiResponse(results=results)

    # ── model ────────────────────────────────────────────

    def _load_model(
        self, model_name: str | None, model_json: dict | None,
    ) -> ModelConfig:
        if model_json:
            cfg = ModelConfig.from_dict(model_json)
            logger.info("loading inline model: %d operators", len(cfg.operators))
            return cfg
        if model_name:
            path = os.path.join(_DATA_DIR, "models", f"{model_name}.json")
            if os.path.exists(path):
                with open(path) as f:
                    raw = json.load(f)
                cfg = ModelConfig.from_dict(raw)
                logger.info("loaded model '%s' from %s", model_name, path)
                return cfg
            logger.warning("model '%s' not found at %s", model_name, path)
            raise HTTPException(404, f"模型 '{model_name}' 不存在")
        raise HTTPException(400, "需要 model_name 或 model_json")

    # ── chip ─────────────────────────────────────────────

    @staticmethod
    def _load_chip(config: str | dict) -> AIChipConfig:
        if not config:
            logger.info("no hardware specified, using default AIChipConfig")
            return AIChipConfig()
        if isinstance(config, dict):
            chip = AIChipConfig.from_dict(config)
            logger.info("loaded chip from inline dict: %s", config.get("name", "unnamed"))
            return chip
        try:
            chip = AIChipConfig.from_name(config)
            logger.info("loaded chip '%s' from hardware library", config)
            return chip
        except Exception:
            logger.warning("chip '%s' not found, using default AIChipConfig", config)
            return AIChipConfig()

    # ── context ──────────────────────────────────────────

    @staticmethod
    def _build_context(wl: WorkloadEntry, hf_config_json: dict | None) -> dict:
        cfg = hf_config_json or {}
        hidden_dim = cfg.get("hidden_size", cfg.get("hidden_dim", 4096))
        intermediate_size = cfg.get("intermediate_size", cfg.get("moe_intermediate_size", 4096))
        moe_intermediate_dim = cfg.get("moe_intermediate_size", cfg.get("intermediate_size", 4096))
        seq_len = wl.request.input_length + wl.request.output_length // 2
        expert_groups = cfg.get("n_routed_experts", 384) // wl.parallel.world_size
        bsz = (wl.request.batch_size + wl.parallel.dp_size - 1) // wl.parallel.dp_size
        return {
            "phase": wl.request.phase,
            "batch_size": wl.request.batch_size,
            "B": bsz,
            "attn_bs": bsz,
            "S": wl.request.input_length if wl.request.phase == "prefill" else 1 + wl.request.num_mtp_tokens,
            "seq_len": seq_len,
            "num_mtp_tokens": wl.request.num_mtp_tokens,
            "avg_accept_tokens": wl.request.avg_accept_tokens,
            "prefix_hit_ratio": wl.request.prefix_hit_ratio,
            "input_length": wl.request.input_length,
            "output_length": wl.request.output_length,
            "num_experts": cfg.get("n_routed_experts", 384),
            "top_k": cfg.get("num_experts_per_tok", 6),
            "expert_groups": expert_groups,
            "hidden_dim": hidden_dim,
            "intermediate_size": intermediate_size,
            "intermediate_dim": intermediate_size,
            "moe_intermediate_dim": moe_intermediate_dim,
            "world_size": wl.parallel.world_size,
            "tp_size": wl.parallel.tp_size,
            "TP_SIZE": wl.parallel.tp_size,
            "dp_size": wl.parallel.dp_size,
            "pp_size": wl.parallel.pp_size,
            "ep_size": wl.parallel.ep_size,
            "cp_size": wl.parallel.cp_size,
            "embed_tp_size": wl.parallel.embed_tp_size,
            "o_tp_size": wl.parallel.o_tp_size,
            "lmhead_tp_size": wl.parallel.lmhead_tp_size,
            "external_shared_expert_rank_size": wl.parallel.external_shared_expert_rank_size,
            "dtype_str": wl.quant.quant_global,
            "fp32": 4,
            "fp64": 8,
            "fp16": 2,
            "bf16": 2,
            "fp8": 1,
            "fp4": 0.5,
            "int8": 1,
            "int4": 1,
            "int32": 4,
            "int64": 8,
            **cfg,
        }

    # ── rank stats ────────────────────────────────────────

    # ── response builders ──────────────────────────────────

    @staticmethod
    def _build_operator_response(
        model: ModelConfig,
        op_results: dict[int, OperatorExecuteResult],
    ) -> list[OperatorResult]:
        op_module_map = {op.op_id: op.op_module for op in model.operators}
        return [
            OperatorResult(
                op_id=r.op_id,
                op_name=r.op_name,
                layer_idx=r.layer_idx,
                rank_idx=r.rank_idx,
                op_module=op_module_map.get(r.op_id, ""),
                compute_cost_us=round(r.compute_cost_us, 2),
                mem_cost_us=round(r.bw_cost_us, 2),
                comm_cost_us=round(r.comm_cost_us, 2),
                bound_type=r.bound_type or "none",
                total_cost_us=round(r.total_cost_us, 2),
                noise_us=round(r.static_cost_us, 2),
                start_time_ns=r.start_time_ns,
                end_time_ns=r.end_time_ns,
                inputs_info=[TensorInfo(name=t.name, shape=t.shape, dtype=t.dtype) for t in r.inputs_info],
                params_info=[TensorInfo(name=t.name, shape=t.shape, dtype=t.dtype) for t in r.params_info],
                outputs_info=[TensorInfo(name=t.name, shape=t.shape, dtype=t.dtype) for t in r.outputs_info],
            )
            for r in op_results.values()
        ]

    @staticmethod
    def _build_operator_statistics_response(
        model: ModelConfig,
        op_results: dict[int, OperatorExecuteResult],
    ) -> list[OperatorStatistics]:
        repeat_dict = model.layer_repeat
        stats_map: dict[str, dict] = {}
        for r in op_results.values():
            name = r.op_name
            if name not in stats_map:
                stats_map[name] = {"total_us": 0.0, "count": 0}
            repeat = repeat_dict[r.layer_idx]
            stats_map[name]["total_us"] += r.total_cost_us * repeat
            stats_map[name]["count"] += repeat
        return [
            OperatorStatistics(
                op_name=name,
                total_cost_us=round(s["total_us"], 2),
                num_calls=s["count"],
                avg_cost_us=round(s["total_us"] / s["count"], 2),
            )
            for name, s in sorted(stats_map.items())
        ]

    @staticmethod
    def _build_layer_cost_time(
        model: ModelConfig,
        op_results: dict[int, OperatorExecuteResult],
    ) -> list[RankResult]:
        if not model.ranks:
            return []

        layer_repeat = model.layer_repeat
        rank_data: dict[int, dict] = {}
        for op_id, r in op_results.items():
            rank_idx = r.rank_idx
            if rank_idx is None or rank_idx == -1: # end和start的rank_idx为-1
                continue

            if rank_idx not in rank_data:
                rank_data[rank_idx] = {"ops": [], "layer_map": {}, "start_time_ns": float("inf"), "end_time_ns": 0}
            rank_data[rank_idx]["ops"].append(op_id)
            rank_data[rank_idx]["start_time_ns"] = min(rank_data[rank_idx]["start_time_ns"], r.start_time_ns)
            rank_data[rank_idx]["end_time_ns"] = max(rank_data[rank_idx]["end_time_ns"], r.end_time_ns)

            # 处理每个rank内的layer结果
            li = r.layer_idx
            if li not in rank_data[rank_idx]["layer_map"]:
                rank_data[rank_idx]["layer_map"][li] = {"op_ids": [], "start_time_ns": float("inf"), "end_time_ns": 0}
            rank_data[rank_idx]["layer_map"][li]["op_ids"].append(op_id)
            rank_data[rank_idx]["layer_map"][li]["start_time_ns"] = min(rank_data[rank_idx]["layer_map"][li]["start_time_ns"], r.start_time_ns)
            rank_data[rank_idx]["layer_map"][li]["end_time_ns"] = max(rank_data[rank_idx]["layer_map"][li]["end_time_ns"], r.end_time_ns)

        result = []
        for rank_idx in sorted(rank_data.keys()):
            d = rank_data[rank_idx]
            rank_start = int(d["start_time_ns"]) if d["start_time_ns"] != float("inf") else 0
            rank_end = int(d["end_time_ns"])
            rank_layers = [
                LayerResultPerRank(
                    layer_idx=li,
                    rank_idx=rank_idx,
                    layer_cost_ns=(ls["end_time_ns"] - ls["start_time_ns"]) if ls["start_time_ns"] != float("inf") else 0,  # 重点计算这个值
                    op_ids=ls["op_ids"],
                    repeat=layer_repeat.get(li, 1),
                    start_time_ns=int(ls["start_time_ns"]) if ls["start_time_ns"] != float("inf") else 0,
                    end_time_ns=int(ls["end_time_ns"]),
                )
                for li, ls in sorted(d["layer_map"].items())
            ]
            result.append(RankResult(
                rank_idx=rank_idx,
                total_cost_ms=round((rank_end - rank_start) / 1_000_000.0, 2), # 这里计算的不准确，后面有调整
                num_ops=len(d["ops"]),
                ops=d["ops"],
                layers=rank_layers,
                start_time_ns=rank_start,
                end_time_ns=rank_end,
            ))
        return result

    @staticmethod
    def _adjust_layer_operator_start_and_end_time(
        model: ModelConfig,
        op_results: dict[int, OperatorExecuteResult],
        ranks: list[RankResult],
    ) -> None:
        """根据 layer_cost_ms 和 repeat 调整 Rank/Layer/Op 的时间戳。"""
        layer_repeat = model.layer_repeat

        for rank in ranks:
            if not rank.layers:
                continue

            sorted_layers = sorted(rank.layers, key=lambda l: l.layer_idx)

            cumulative_ns: int = 0
            for layer in sorted_layers:
                # 调整 LayerResultPerRank
                layer.start_time_ns = cumulative_ns
                layer.end_time_ns = layer.start_time_ns + layer.layer_cost_ns

                # 调整该层内所有算子的时间，并统计所属层的内存占用
                layer_min_start = min(
                    (op_results[pid].start_time_ns for pid in layer.op_ids if pid in op_results),
                    default=0,
                )
                offset = cumulative_ns - layer_min_start
                for op_id in layer.op_ids:
                    r = op_results.get(op_id)
                    if r is None:
                        continue
                    r.start_time_ns += offset
                    r.end_time_ns = r.start_time_ns + int(r.total_cost_us * 1000)
                    layer.param_bytes += r.param_bytes
                    layer.io_bytes += (r.input_bytes + r.output_bytes)

                cumulative_ns += layer.layer_cost_ns * layer_repeat.get(layer.layer_idx, 1)

    @staticmethod
    def _fill_mtp_results(
        model: ModelConfig,
        op_results: dict[int, OperatorExecuteResult],
        ranks: list[RankResult],
        req: SimulateRequest,
    ) -> None:
        """
        根据num_mtp_tokens，调整mtp的总耗时，从而影响总体模型的耗时：
        1. 所有mtp的算子，根据num_mtp_tokens的倍数进行复制
        2. 将layer_idx=0的层耗时，根据num_mtp_tokens的倍数进行复制扩展，例如：下面的流程应该复制num_mtp_tokens次
           Embedding->AllReduce->RMSNorm->RMSNorm->Linear->Linear->Layer_0->MHCHead->RMSNorm->ColumnParallelLinear->AllGather
           在这个流程中，MHCHead及之后的算子的开始时间应该后移Layer_0_start_time_ns纳秒
        """
        num_mtp_tokens = req.workloads[0].request.num_mtp_tokens
        if num_mtp_tokens == 0:
            return

        # 收集 MTP 层的layer_idx=980
        mtp_layer_idx = 980

        is_prefill = req.workloads[0].request.phase == "prefill"
        if is_prefill:
            in_len = req.workloads[0].request.input_length
            mtp_q_len = in_len + 1
        else: # decode
            avg_accept_tokens = req.workloads[0].request.avg_accept_tokens
            mtp_q_len = max(1, round(avg_accept_tokens))
            in_len = 1 + num_mtp_tokens


        # TODO: 暂时不考虑 prefix cache命中的情况
        def calc_layer_cost_ns(layer: LayerResultPerRank):
            attn_ops = ["FlashAttention", "PageAttention", "ScaledDotProductAttn", "SparseAttentionSharedKV", "SparseFlashAttention"]
            ratios = (req.hf_config_json or {}).get("compress_ratios")
            cmp_ratio = ratios[0] if ratios and 0 < len(ratios) else 0
            layer_cost_time_ns = layer.end_time_ns - layer.start_time_ns
            for op_id in layer.op_ids:
                op = op_results.get(op_id)
                if op and ("Compressor" in op.op_name or "Indexer" in op.op_name):
                    layer_cost_time_ns -= (op.end_time_ns - op.start_time_ns)
                elif op and op.op_name in attn_ops and cmp_ratio > 0:
                    layer_cost_time_ns += ((op.end_time_ns - op.start_time_ns - op.static_cost_us * 1000) * cmp_ratio)
                else:
                    continue
            return layer_cost_time_ns, layer.param_bytes, layer.io_bytes
            
        for rank in ranks:
            if not rank.layers:
                continue

            # 按 layer_idx 排序，MTP 层在常规层之后
            sorted_layers = sorted(rank.layers, key=lambda l: l.layer_idx)

            layer0_cost_time_ns = 0
            layer0_param_bytes, layer0_io_bytes = 0, 0
            for layer in sorted_layers:
                if layer.layer_idx == 0:
                    layer0_cost_time_ns, layer0_param_bytes, layer0_io_bytes = calc_layer_cost_ns(layer)
                    continue
                if layer.layer_idx != mtp_layer_idx:
                    continue

                # —————————————————— 开始计算所有MTP层的总耗时 ——————————————————
                # 首先，计算Embedding->AllReduce->RMSNorm->RMSNorm->Linear->Linear->MHCHead->RMSNorm->ColumnParallelLinear->AllGather的总耗时
                mtp_cost_ns = layer.end_time_ns - layer.start_time_ns                           # 980，第一层mtp
                mtp_cost_ns += (mtp_cost_ns * (num_mtp_tokens - 1) // mtp_q_len)                # 其他mtp层的qlen=1
                # 其次：计算 MTP Block 的总耗时
                mtp_cost_ns += round(layer0_cost_time_ns / in_len * mtp_q_len)                  # 980，第一层mtp
                mtp_cost_ns += (layer0_cost_time_ns // in_len * (num_mtp_tokens - 1))        # 其他mtp层qlen=1

                layer.param_bytes += layer0_param_bytes
                # 激活值仅做近似计算
                layer.io_bytes += round(layer0_io_bytes / in_len * mtp_q_len)
                layer.io_bytes += round(layer0_io_bytes / in_len * (num_mtp_tokens - 1))

                # 最后计算mtp层的完成时间
                layer.end_time_ns = layer.start_time_ns + mtp_cost_ns
                layer.layer_cost_ns = mtp_cost_ns

    @staticmethod
    def _fill_rank_results(
        model: ModelConfig,
        op_results: dict[int, OperatorExecuteResult],
        ranks: list[RankResult],
        mem_capacity_gb: float = 64.0,
        noise_gb: float = 5,
    ) -> None:
        """根据 layer_cost_ms 和 repeat 调整 Rank/Layer/Op 的时间戳。"""
        
        for rank in ranks:
            # 调整该层内所有算子的时间，并统计所属层的内存占用
            rank.start_time_ns = min(
                (op_results[op_id].start_time_ns for op_id in rank.ops if op_id in op_results),
                default=0,
            )

            # 这里必须用layer的完成时间，以兼容mtp层
            rank.end_time_ns = max(
                (layer.end_time_ns for layer in rank.layers),
                default=0,
            )

            # 注：这里不能用所有算子的end-start，会因为mtp层算子end时间有问题
            rank.total_cost_ms = round(sum(
              layer.layer_cost_ns * layer.repeat for layer in rank.layers
            ) / 1_000_000.0, 2)

            rank.param_bytes = sum(
                layer.param_bytes * layer.repeat for layer in rank.layers
            )

            rank.io_bytes = sum(
                layer.io_bytes * layer.repeat for layer in rank.layers
            )

            rank.peak_mem_gb = (rank.param_bytes + rank.io_bytes) / 1024.0 / 1024.0 / 1024.0 + noise_gb 
            rank.noise_gb = noise_gb
            rank.mem_capacity_gb = mem_capacity_gb
            rank.oom = rank.peak_mem_gb > mem_capacity_gb

    @staticmethod
    def _fill_end_operator(
        op_results: dict[int, OperatorExecuteResult],
        overall_end_time: int,
    ):
        for _, r in op_results.items():
            if r.op_name == "end":
                r.start_time_ns = overall_end_time
                r.end_time_ns = overall_end_time
                break

    @staticmethod
    def _calc_mem_gb_cost(
        ranks: list[RankResult],
    ):
        """计算所有 rank 中各指标的最大值。"""
        if not ranks:
            return 0, 0, 0

        max_peak_gb = 0
        model_start_time = 10**18
        model_end_time = 0

        for rank in ranks:
            max_peak_gb = max(max_peak_gb, rank.peak_mem_gb)
            model_start_time = min(model_start_time, rank.start_time_ns)
            model_end_time = max(model_end_time, rank.end_time_ns)


        return max_peak_gb, model_start_time, model_end_time

    # ── response ─────────────────────────────────────────

    def _build_response(
        self,
        req: SimulateRequest,
        model: ModelConfig,
        op_results: dict[int, OperatorExecuteResult],
        chip: AIChipConfig,
        hw_name: str = "",
    ) -> SimulateResponse:
        wl = req.workloads[0] if req.workloads else WorkloadEntry()
        world_size = wl.parallel.world_size
        tp = wl.parallel.tp_size
        dp = wl.parallel.dp_size
        ep = wl.parallel.ep_size

        strategy_parts = []
        if tp > 1: strategy_parts.append(f"TP{tp}")
        if dp > 1: strategy_parts.append(f"DP{dp}")
        if wl.parallel.pp_size > 1: strategy_parts.append(f"PP{wl.parallel.pp_size}")
        if ep > 1: strategy_parts.append(f"EP{ep}")

        mem_capacity_gb = float(chip.spec_memory_size) if chip.spec_memory_size > 0 else 64.0
        noise_gb = chip.memory_noise if chip.memory_noise > 0 else 5

        ranks = self._build_layer_cost_time(model, op_results)
        self._adjust_layer_operator_start_and_end_time(model, op_results, ranks)
        self._fill_mtp_results(model, op_results, ranks, req)
        self._fill_rank_results(model, op_results, ranks, mem_capacity_gb, noise_gb)
        operators = self._build_operator_response(model, op_results)
        op_statistics = self._build_operator_statistics_response(model, op_results)

        (
            max_peak_mem_gb,
            overall_start_time,
            overall_end_time,
        ) = self._calc_mem_gb_cost(ranks)
        oom = max_peak_mem_gb > mem_capacity_gb

        self._fill_end_operator(op_results, overall_end_time)

        overall_cost_ms = round((overall_end_time - overall_start_time) / 1_000_000.0, 4)

        tpot_ms = (overall_cost_ms / wl.request.avg_accept_tokens) if wl.request.avg_accept_tokens > 0 else overall_cost_ms
        ttft_ms = overall_cost_ms
        tps = wl.request.batch_size / (tpot_ms / 1000.0) / world_size if tpot_ms > 0 else 0
        qps = 1000 / tps / wl.request.output_length

        logger.info(
            "response: tpot=%.2f ms, oom=%s, peak_mem=%.2f GB/%d NPUs",
            tpot_ms, oom, max_peak_mem_gb * world_size, world_size,
        )

        return SimulateResponse(
            hardware_name=hw_name,
            tpot_ms=round(tpot_ms, 2),
            prefill_latency_ms=round(ttft_ms, 2),
            decode_latency_per_token_ms=round(tpot_ms, 2),
            tps=round(tps, 2),
            qps=round(qps, 2),
            peak_mem_gb=max_peak_mem_gb,
            oom=oom,
            operators=operators,
            op_statistics=op_statistics,
            ranks=ranks,
            strategy="_".join(strategy_parts) if strategy_parts else "single",
            timestamp=datetime.now(timezone.utc).isoformat(),
            start_time_ns=overall_start_time,
            end_time_ns=overall_end_time,
        )
