export const DTYPES = ['fp32', 'fp16', 'bf16', 'fp8', 'fp4', 'int4', 'int8', 'int32', 'int64']

export const MODULE_GROUPS: { label: string; modules: string[] }[] = [
  { label: 'Attention', modules: ['attn', 'attn.input_layernorm', 'attn.x_proj', 'attn.indexer', 'attn.indexer.compressor', 'attn.compressor','attn.flash_attention', 'attn.o_proj'] },
  { label: 'MLP', modules: ['mlp', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.gate_up_proj', 'mlp.act_fn', 'mlp.down_proj'] },
  { label: 'MoE', modules: ['moe', 'moe.gate', 'moe.topk', 'moe.gate_topk', 'moe.combine', 'moe.dispatch', 'moe.routed_expert.gate_up_proj', 'moe.routed_expert.act_fn', 'moe.routed_expert.down_proj', 'moe.shared_expert.gate_up_proj', 'moe.shared_expert.act_fn', 'moe.shared_expert.down_proj'] },
  { label: 'Embedding', modules: ['embed'] },
  { label: 'MHC', modules: ['mhc', 'mhc_pre', 'mhc_post', 'mhc_head'] },
  { label: 'MTP', modules: ['mtp', 'mtp.embed', 'mtp.e_norm', 'mtp.h_norm', 'mtp.e_proj', 'mtp.h_proj', "mtp.block", 'mtp.mhc_head', 'mtp.lm_head'] },
  { label: 'Norm', modules: ['norm', "input_norm", "post_attn_norm"] },
  { label: 'Output', modules: ['lm_head'] },
]
