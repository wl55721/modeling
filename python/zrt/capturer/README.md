# Captures

Uses `TorchDispatchMode` to intercept the full aten op sequence of a causal LM forward pass on **meta tensors**, and writes the result to a formatted Excel workbook.

No model weights need to be downloaded.

---

## Core Idea

### Main Working Flow

```
HF config.json
       ↓  AutoConfig.from_pretrained
   config object
       ↓  AutoModelForCausalLM.from_config  +  torch.device("meta")
   model structure (no weights, all meta tensors)
       ↓  TorchDispatchMode  +  ModuleTracker hooks
   aten op sequence (shape / dtype / module path)
       ↓  two-stage fusion + dataflow analysis
       ↓  openpyxl
   model_ops.xlsx  +  *_fusion_rules.json
```

### Meta Tensors

**Meta Tensors** carry only shape and dtype — they allocate no memory and perform no numerical computation. The entire trace has a tiny memory footprint; even DeepSeek-V3 (671B parameters) finishes in seconds.

### Fusion

The raw aten op sequences captured via `TorchDispatchMode` is too fine-grained. We fuse some of these ops into high-level fused ops — for example, fusing `pow, mean, add, rsqrt, mul, mul` into `RMSNorm`.

---

## Output Excel Explained

The workbook contains 6 sheets:

| Sheet | Description |
|-------|-------------|
| Model Config | Summary of model configuration |
| Fused Operators | Fused op sequence (main view, with fused I/O mapping) |
| Raw Operator Sequence | Full raw aten op sequence |
| Summary | Aggregated statistics grouped by fused op |
| By Layer | Aggregated statistics grouped by layer |
| Fusion Rules | Auto-discovered fusion patterns (with fused I/O mapping) |

And a `*_fusion_rules.json` file is also generated, recording the auto-discovered op fusion patterns.

---

### Sheet: Model Config

Overview of the model's key configuration (shown if present, skipped otherwise):

| Field | Description |
|-------|-------------|
| `model_id` | Model source |
| `model_type` | Architecture type |
| `hidden_size` | Hidden dimension |
| `num_hidden_layers` | Total layers / traced layers |
| `num_attention_heads` | Number of attention heads |
| `vocab_size` | Vocabulary size |
| `n_routed_experts` | MoE routed expert count (DeepSeek) |
| `num_local_experts` | MoE local expert count (Mixtral) |
| `q_lora_rank` / `kv_lora_rank` | MLA low-rank dimensions (DeepSeek) |

---

### Sheet: Fused Operators (main view)

The op sequence after two-stage automatic fusion. Each row contains:

- **Fused Input Shapes/Dtypes**: external input tensor info of the fused kernel
- **Input Sources**: which sub-op's input port each input comes from
- **Fused Output Shapes/Dtypes**: external output tensor info of the fused kernel
- **Output Sources**: which sub-op's output port each output is produced by

---

## Component Label Scheme

Labels in the `Component` column are inferred from naming patterns in the module path, independent of any specific model implementation:

| Label prefix | Meaning | Color |
|--------------|---------|-------|
| `attn_norm` | LayerNorm / RMSNorm before attention | green |
| `ffn_norm` | LayerNorm / RMSNorm after attention | green |
| `final_norm` | Final norm layer | green |
| `attn.q_proj` / `attn.k_proj` / … | Standard QKV and output projections | blue |
| `attn.q_a_proj` / `attn.kv_a_proj` / … | MLA low-rank projections (DeepSeek) | blue |
| `attn.score` | QK inner product | blue |
| `attn.softmax` | Softmax | blue |
| `attn.rope` | RoPE positional encoding | blue |
| `moe.gate.*` | MoE routing / gating | orange |
| `moe.shared.*` | MoE shared experts (DeepSeek) | yellow |
| `moe.experts.*` | MoE routed expert MLP | pink |
| `ffn.gate_proj` / `ffn.up_proj` / `ffn.down_proj` | Dense FFN projections | purple |
| `ffn.silu` / `ffn.mul` | Activation functions | purple |
| `embedding` | Token embedding | gray |
| `lm_head` | Language model output head | gray |

---
