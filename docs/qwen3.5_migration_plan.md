# MiniMind 骨架迁移至 Qwen3.5 系列：实施计划

## 1. 现状与目标

**现状**：`minimind-3` 是从零训练的小模型，结构对齐 `Qwen3 / Qwen3-MoE`（GQA + QK-Norm + SwiGLU + RMSNorm + RoPE），导出时直接映射到 `Qwen3ForCausalLM / Qwen3MoeForCausalLM`（`scripts/convert_model.py`），借此接入 transformers / vLLM / SGLang / llama.cpp / ollama 生态。

**目标**：把主线结构切换到 Qwen3.5 文本骨架（`Qwen3_5ForCausalLM`，`model_type=qwen3_5_text`），保持"原生 PyTorch 训练 → 无损导出 HF 格式 → 复用生态"的工作流不变。

**已锁定决策（2026-09-04）**：MoE 严格对齐 `Qwen3_5MoeForCausalLM`（含 shared expert）；`full_attention_interval=4`；`rope_theta=1e7` 等超参与官方对齐。

**范围界定**：

- 只做文本骨架。Qwen3.5 的视觉塔（`Qwen3_5VisionModel`）、MTP 头（`mtp.*`）不在范围内；导出时 `Qwen3_5ForCausalLM` 会自动忽略这两部分权重键。
- tokenizer 保持 6400 词表不变，仅在需要时同步 chat template。
- 这是结构迁移，**现有 `.pth` 权重不可复用，需要重新预训练**。

## 2. Qwen3 → Qwen3.5 结构差异

| 组件 | Qwen3（当前） | Qwen3.5（目标） | 对 MiniMind 的影响 |
|---|---|---|---|
| 层类型 | 全部 softmax 注意力 | 混合层：3 层 Gated DeltaNet（线性注意力）+ 1 层 Gated Attention，由 `layer_types` / `full_attention_interval=4` 控制 | 新增线性注意力模块；`MiniMindBlock` 按层类型分派 |
| 全注意力 | GQA + QK-Norm | GQA + QK-Norm + **输出门控**：`q_proj` 输出维度翻倍，一半作 query、一半经 `sigmoid` 门控 attention 输出 | `Attention.q_proj` 形状变化，forward 增加门控 |
| RoPE | 全维度旋转 | **部分旋转** `partial_rotary_factor=0.25`，仅前 25% 的 head_dim 参与旋转；线性注意力层不使用 RoPE | `precompute_freqs_cis` 的 `dim` 改为 `head_dim * 0.25`；`apply_rotary_pos_emb` 需切分旋转/直通部分 |
| RMSNorm | `weight * norm(x)`，weight 初值 1 | **零中心** `(1 + weight) * norm(x)`，weight 初值 0 | 所有 RMSNorm（含 q_norm/k_norm）改写；影响权重初始化与导出映射 |
| 线性注意力 | 无 | Gated DeltaNet：`in_proj_qkv / in_proj_z / in_proj_b / in_proj_a` + 深度可分离因果 Conv1d（kernel=4）+ Q/K L2 归一化 + 门控 delta 规则递推 + `RMSNormGated` + `out_proj`；可学习参数 `A_log`、`dt_bias` | 全新模块，需要 chunk（训练/prefill）和 recurrent（decode）两套等价实现 |
| 推理缓存 | 每层 `(k, v)` | 混合缓存：全注意力层存 `(k, v)`；线性层存 `conv_state (B, conv_dim, 4)` 和 `recurrent_state (B, num_v_heads, d_k, d_v)`，大小固定 | 自定义 `generate` 与 `MiniMindModel.forward` 中 `past_key_values[0][0].shape[1]` 取 `start_pos` 的写法失效（第 0 层是线性层） |
| MoE | 无 shared expert | `Qwen3_5MoeForCausalLM` 沿用 Qwen2-MoE 块，**含 shared expert + shared_expert_gate** | 若要严格对齐 MoE 导出类，需要加回 shared expert（与 README 当前"移除 shared expert"的设计相悖，需决策） |
| head_dim | `hidden // heads` | 独立配置（官方为 256） | MiniMind 可继续用 96（768/8），仅需保证与 RoPE/mrope 配置一致 |
| RoPE 配置格式 | `rope_theta` 顶层字段 | `rope_parameters={rope_theta, partial_rotary_factor, mrope_section, mrope_interleaved}` | 导出脚本需写新格式；纯文本下 3 路位置 id 相同，mRoPE 退化为普通 RoPE，但 `mrope_section` 之和必须等于旋转维度的一半 |

参考实现：transformers `models/qwen3_5/modular_qwen3_5.py` 与 `models/qwen3_next/modular_qwen3_next.py`。

## 3. MiniMind-3.5 目标配置（建议）

以 `hidden_size=768, num_hidden_layers=8` 为基准，尽量保持参数量与当前 64M 持平：

| 字段 | 建议值 | 说明 |
|---|---|---|
| `layer_types` | `[L, L, L, F, L, L, L, F]` | 3:1，8 层里只有 2 层全注意力；见第 7 节风险 |
| `num_attention_heads / num_key_value_heads / head_dim` | 8 / 4 / 96 | 沿用现值 |
| `attn_output_gate` | `True` | `q_proj: 768 → 8*96*2 = 1536` |
| `partial_rotary_factor` | 0.25 | 旋转维度 24，`mrope_section=[4, 4, 4]`（之和 12 = 24/2） |
| `linear_num_key_heads / linear_num_value_heads` | 4 / 8 | 官方 V 头数是 K 头数的 2 倍（反向 GQA） |
| `linear_key_head_dim / linear_value_head_dim` | 96 / 96 | `key_dim=384, value_dim=768, conv_dim=1536` |
| `linear_conv_kernel_dim` | 4 | 官方默认 |
| `rope_theta` | 1e6（保留）或 1e7（对齐官方） | 与 YaRN 外推参数一起复核 |
| `rms_norm_eps` | 1e-6 | 不变 |

参数量估算（单层）：Gated DeltaNet ≈ 2.4M（`in_proj_qkv` 1.18M + `in_proj_z` 0.59M + `out_proj` 0.59M + 少量），Gated Attention ≈ 2.36M（`q_proj` 翻倍），与当前 Attention 1.77M 相比略增，总参数约 +4M，可接受。

## 4. 实施阶段

### 阶段 0：依赖与基线

- `requirements.txt`：`transformers` 从 `4.57.6` 升到含 `Qwen3_5*` 的 5.x 版本（Qwen3.5 于 2026-02-09 合入，需在实施时确认最低版本，建议 `>=5.2`）。仓库已有大量 `transformers>=5` 分支判断，升级后可顺势清理 4.x 兼容路径。
- 可选加速依赖 `flash-linear-attention`（`fla`）、`causal-conv1d`：仅用于生态推理侧；MiniMind 自身实现走纯 PyTorch，不引入硬依赖。
- 在切换前，用现有 Qwen3 结构固定一组基线：同 token 预算下的 pretrain loss 曲线、`eval_llm.py` 采样输出、tokens/s。后续所有对比以此为准。

### 阶段 1：配置层（`model/model_minimind.py::MiniMindConfig`）

- 新增字段：`layer_types`、`full_attention_interval`、`attn_output_gate`、`partial_rotary_factor`、`linear_num_key_heads`、`linear_num_value_heads`、`linear_key_head_dim`、`linear_value_head_dim`、`linear_conv_kernel_dim`。
- `layer_types` 缺省时按 `full_attention_interval` 自动生成，规则与 HF 一致：`(i+1) % interval == 0` 为全注意力。
- 顺带修正现有隐患：多处脚本传入 `max_seq_len=` 但配置类只认 `max_position_embeddings`，该参数目前被静默丢弃。
- 保留 `use_moe` 及 MoE 字段；若采纳 shared expert，增加 `shared_expert_intermediate_size`。

### 阶段 2：模型层（`model/model_minimind.py`）

按依赖顺序推进，每一步都要能跑通前向：

1. **RMSNorm 零中心化**：`(1 + weight) * norm(x)`，weight 零初始化。新增 `RMSNormGated`（norm 后乘 `silu(gate)`）供 DeltaNet 使用。
2. **RoPE 部分旋转**：`precompute_freqs_cis(dim=int(head_dim * partial_rotary_factor))`；`apply_rotary_pos_emb` 拆分 `[:rot_dim]` 旋转、`[rot_dim:]` 直通后拼接。YaRN 逻辑随 `dim` 变化自动适配，需复核 `inv_dim` 公式仍以旋转维度为准。
3. **Gated Attention**：`q_proj` 输出翻倍，`chunk` 成 query 与 gate；attention 输出 reshape 后乘 `sigmoid(gate)` 再过 `o_proj`。KV 缓存逻辑不变。
4. **GatedDeltaNet 模块**（新增类，命名 `linear_attn`，子模块名与 HF 完全一致以便 strict 导出）：
   - 投影：`in_proj_qkv`、`in_proj_z`、`in_proj_b`、`in_proj_a`、`out_proj`，均无 bias。
   - `conv1d`：`groups=conv_dim, kernel=4, padding=3, bias=False`，之后 `silu`，截断到 `seq_len`。
   - 门控：`beta = sigmoid(b)`，`g = -exp(A_log) * softplus(a + dt_bias)`，以 fp32 计算。
   - K/Q 头数少于 V 头数时 `repeat_interleave` 对齐。
   - 核心：移植 HF 的 `torch_chunk_gated_delta_rule`（chunk=64，训练与 prefill）与 `torch_recurrent_gated_delta_rule`（decode 单步）两个纯 PyTorch 函数，内部 fp32。
   - 输出：`RMSNormGated(core_out, z)` → `out_proj`。
   - padding 处理：进入模块前按 `attention_mask` 把 pad 位置置零（对应 HF `apply_mask_to_padding_states`），否则 batch 内右侧 padding 会污染递推状态。
5. **`MiniMindBlock` 分派**：按 `layer_types[layer_id]` 实例化 `self_attn` 或 `linear_attn`，forward 传入对应缓存对象。
6. **混合缓存与 `generate`**：
   - 缓存结构改为每层一个 tuple：全注意力层 `(k, v)`，线性层 `(conv_state, recurrent_state)`。
   - `start_pos` / `past_len` 不再从 `past_key_values[0][0].shape[1]` 取，改为取首个全注意力层的 KV 长度，或在缓存对象上显式维护 `seen_tokens`。
   - decode 时线性层调用 recurrent 分支，并更新 `conv_state`（滑窗左移一位）与 `recurrent_state`。
   - `return_kv` 分支和 `rollout_engine.py` 的 torch 后端只经由 `generate` 使用缓存，接口对外不变。
7. **MoE**（若继续维护 MoE 线）：
   - 方案 A：加回 shared expert（`shared_expert` + `shared_expert_gate`），导出到 `Qwen3_5MoeForCausalLM`，README 中"移除 shared expert"的表述需更新。
   - 方案 B：MoE 继续导出为 `Qwen3MoeForCausalLM`，但这样 MoE 线就不再使用 Qwen3.5 混合注意力，与主线分叉。
   - 建议先完成 Dense 主线，MoE 采用方案 A 作为后续独立 PR。

### 阶段 3：导出与回转（`scripts/convert_model.py`）

- `convert_torch2transformers`：`Qwen3Config/Qwen3ForCausalLM` → `Qwen3_5TextConfig/Qwen3_5ForCausalLM`；补齐 `layer_types`、`attn_output_gate`、`partial_rotary_factor`、`linear_*`、`rope_parameters` 字段；保留 `load_state_dict(strict=True)` 作为结构对齐的硬校验。
- 由于 MiniMind 内部命名与 HF 保持一致（`model.layers.{i}.linear_attn.*` / `self_attn.*`），Dense 导出不需要键名重映射；MoE 沿用现有 `gate_up_proj / down_proj` 堆叠逻辑。
- 现有对 `config.json` 的后处理（把 `rope_parameters` 删掉、写回顶层 `rope_theta`）与 Qwen3.5 格式冲突，需要移除或改写。
- `convert_transformers2torch`：反向路径不再需要特殊处理，但要验证 `Qwen3_5ForCausalLM` 加载时忽略的 `mtp.*` 不会导致回转缺键。
- 保存目录改为 `../minimind-3.5`（或按最终命名）。

### 阶段 4：训练链路

- **LoRA**（`model/model_lora.py`）：当前只给方阵 `nn.Linear` 挂 LoRA。迁移后 `q_proj` 变为 768→1536 不再是方阵；`o_proj`、`in_proj_z`、`out_proj` 是方阵会被自动命中。建议改为显式模块名列表（默认 `q_proj, k_proj, v_proj, o_proj, in_proj_qkv, out_proj`），避免行为随结构漂移。
- **混合精度**：delta 规则递推与门控在 fp32 计算，其余部分保持 bf16/fp16；fp16 + GradScaler 路径需专门跑一轮确认无 NaN。
- **`torch.compile` / DDP**：chunk 算法含变长切分与 pad，先确认 `torch.compile` 在 `trainer_utils.py` 里的用法不会频繁 recompile；不行则对 DeltaNet 局部 `disable`。
- **RL 训练脚本**（GRPO / PPO / Agent / DPO）：只依赖 `generate` 和 `forward(labels=...)`，理论上无需改动；但 `train_grpo.py` 里对 `prompt_inputs` 的左截断 + 左 padding 采样，需要验证线性层的 padding mask 处理正确。
- **蒸馏**（`train_distillation.py`）：教师/学生同构即可，无额外改动。

### 阶段 5：推理与生态

| 入口 | 改动 |
|---|---|
| `eval_llm.py`、`scripts/serve_openai_api.py`（原生路径） | 随 `MiniMindConfig` 新字段自动生效；`inference_rope_scaling` 仅作用于 2 层全注意力，效果需重新评估 |
| `scripts/web_demo.py`（HF 路径） | `AutoModelForCausalLM` 自动解析为 `Qwen3_5ForCausalLM`，无改动 |
| vLLM | 原生支持 `qwen3_5`，README 中的 `--model-impl transformers` 可去掉；需要 `fla` / `causal-conv1d` 才能走快路径 |
| SGLang | 原生支持 Qwen3.5 混合结构；`--attention-backend triton` 启动参数需按其 hybrid 后端要求复核 |
| llama.cpp | `convert_hf_to_gguf.py` 已支持 `Qwen3_5ForCausalLM`（GGUF 架构 `qwen35`）；`get_vocab_base_pre` 中复用 `qwen2` 的 tokenizer hack 保留 |
| ollama | Modelfile 无变化，依赖 GGUF 可用 |

### 阶段 6：文档与命名

- README / README_en 的"结构"章节、参数表、Qwen3 生态表述改为 Qwen3.5；新增混合层说明。
- `images/LLM-structure.jpg`、`LLM-structure-moe.jpg` 需重绘（人工任务，单独跟踪）。
- 模型命名建议 `minimind-3.5` / `minimind-3.5-moe`，权重文件名规则 `{weight}_{hidden_size}[_moe].pth` 不变，但与旧权重不兼容，需在 README 明示。
- 是否同步 Qwen3.5 官方 chat template（默认开启 thinking 等）单独评估；当前模板已支持 tools 与 `open_thinking`，非阻塞项。

## 5. 验证清单

| 层级 | 验证项 | 通过标准 |
|---|---|---|
| 单元 | chunk 与 recurrent 两种 delta 规则对同一序列输出一致 | 最大绝对误差 < 1e-4（fp32） |
| 单元 | 一次性 prefill 与逐 token decode（带混合缓存）的 logits 一致 | 同上 |
| 单元 | 带右 padding 的 batch 与单条无 padding 的输出一致 | 同上 |
| 导出 | `convert_torch2transformers` strict 加载成功；原生模型与 `Qwen3_5ForCausalLM` 同输入 logits 一致 | 最大绝对误差 < 1e-3（fp16） |
| 导出 | HF 格式 `generate` 与原生 `generate` 贪心输出一致 | 完全一致 |
| 训练 | 100M token 级别小规模预训练，loss 曲线与阶段 0 基线对比 | 不明显劣于基线（差距 < 0.05） |
| 训练 | full_sft → lora → dpo → grpo 各脚本 smoke run | 无报错、loss 正常下降 |
| 性能 | 训练 tokens/s 与显存 | 记录数值，见第 7 节预期 |
| 生态 | vLLM / SGLang / llama.cpp 三条链路各跑通一次对话 | 输出可读 |

所有临时验证脚本放在仓库外（`/tmp`），不进入提交。

## 6. 交付拆分

原计划分 6 个 PR，实际按用户要求在 PR #2 一次性落地了 1–6 项（含 MoE shared expert）。图片重绘与新权重训练见第 9 节。

## 7. 风险与待决策项

- **短序列收益有限**：训练截断长度多为 340–1024 token，线性注意力的复杂度优势在此区间几乎体现不出来；纯 PyTorch chunk 实现反而可能比 SDPA 慢。预期训练吞吐下降 10%–30%，需要以阶段 0 基线实测确认，并在 README 说明迁移动机是"结构对齐生态"而非提速。
- **8 层里只有 2 层全注意力**：小模型在检索类任务上可能退化。建议对 `full_attention_interval=2`（4 层全注意力）做一次消融再定默认值；官方最小的 Qwen3.5-0.8B 是 24 层 3:1。
- **权重不兼容**：旧的 `minimind-3` 权重无法迁移，需要完整重跑 pretrain 与 SFT 主线，是本次迁移最大的成本项。
- **transformers 5.x 升级的连带影响**：所有脚本要在新版本下回归一遍，尤其是 tokenizer 保存与 `config.json` 后处理逻辑。
- **MoE 设计取舍**：已决策加回 shared expert，严格对齐 `Qwen3_5MoeForCausalLM`。
- **YaRN 外推**：仅对 2 层全注意力生效，且旋转维度缩小为 24，`beta_fast/beta_slow` 的默认值需重新标定。

## 8. 验收结果（2026-09-04，CPU 环境）

阶段 1–4 已在 PR #2 一次性落地，验收项与结果：

| 验证项 | 结果 |
|---|---|
| chunk vs recurrent delta 规则一致性 | 通过，误差 ~1e-7 |
| prefill vs 逐 token decode；`generate` 有/无 cache | 通过，贪心输出完全一致 |
| 右 padding / 左 padding 批量生成与单条一致 | 通过 |
| 原生 vs `Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM` logits | 通过，误差 ~2e-6，键名严格对齐 |
| `convert_model.py` 导出 → `AutoModelForCausalLM` 回载 → 回转 `.pth` strict 加载 | 通过 |
| HF `generate` 与原生 `generate` 贪心一致 | 通过 |
| Dense / MoE 训练步（fp32、bf16 autocast），所有参数均有梯度 | 通过 |
| LoRA 训练 → save → merge → 基模 strict 加载 | 通过 |
| YaRN 路径前向 | 通过 |
| `torch.compile`（inductor）训练步，梯度与 eager 对比 | 通过，误差 ~5e-7，单图无中断 |
| `train_pretrain / train_full_sft / train_lora / train_dpo` 最小 smoke | 通过 |
| `eval_llm.py` 原生路径 | 通过 |
| CPU 相对吞吐（hidden 768，8 层，batch 4） | hybrid 比全注意力慢：T=340 约 1.5x，T=768 约 1.6x |

验收中修复的问题：

- `chunk/recurrent_gated_delta_rule` 中 `.to(dtype, memory_format=...)` 改为显式 `.contiguous()`。
- `MiniMindModel.forward` 中 RoPE 缓冲区的数据相关判断改为一次性 Python 标志，消除 `torch.compile` 图中断。

已知非阻塞项：`aot_eager` 调试后端下 `Linear → softplus → transpose` 反向会报 `view` 错误，属 PyTorch 侧问题，inductor 正常，HF 参考实现写法相同。

实测参数量：Dense 68.74M；MoE 248.08M-A113.60M。

## 9. 后续工作

按优先级排列，前三项是发布新权重前的必需项。

### P0：重训主线权重

- 旧 `minimind-3` 权重不可加载，需从零重跑 `pretrain_t2t → sft_t2t → rlaif / agent_rl`。
- 先用 `pretrain_t2t_mini` 跑一次短程对照：新结构 vs 老结构（`git checkout master` 的 `model_minimind.py`）在同 token 预算下的 loss 曲线与 tokens/s，作为 GPU 上的真实性能基线。
- 若 GPU 吞吐下降超过 30%，评估接入 `flash-linear-attention`（`fla`）与 `causal-conv1d` 作为可选快路径（检测到即用，缺失则回落纯 PyTorch）。

### P0：`full_attention_interval` 消融

- 用 mini 数据对比 interval=4（2 层全注意力）与 interval=2（4 层全注意力）的 loss 与简单检索题表现；默认值保持 4，若差距明显则在 README 给出建议。

### P0：生态链路实机验证

- vLLM：确认原生 `qwen3_5` 路径可加载 6400 词表小模型；README 已去掉 `--model-impl transformers`。
- SGLang：复核 `--attention-backend` 参数在 hybrid 结构下的取值。
- llama.cpp：验证 `convert_hf_to_gguf.py` 对 `Qwen3_5ForCausalLM` 的转换，含 `mrope_section`、tokenizer pre-hash hack；ollama Modelfile 随之验证。

### P1：训练链路补齐

- RL 脚本（GRPO / PPO / Agent）在 GPU 上跑通一轮，重点看左 padding rollout 与 `rollout_engine` SGLang 后端。
- `train_distillation.py` smoke。
- fp16 + GradScaler 路径专项检查（bf16 已验证）。
- DDP 多卡 smoke。

### P1：YaRN 重新标定

- 旋转维度由 96 降到 24，`beta_fast=32 / beta_slow=1` 的默认值需要在长文本上重新校准，或在 README 说明外推能力受限于 2 层全注意力。

### P2：文档与资产

- 重绘 `images/LLM-structure.jpg` 与 `LLM-structure-moe.jpg`。
- README 其余提到 `64M / 198M-A64M` 的位置（成本表、评测表）在新权重出来后统一更新。
- 评估是否同步 Qwen3.5 官方 chat template（当前模板已支持 tools 与 `open_thinking`）。

### P2：代码整理

- `trainer_utils.init_model` 读取权重固定使用 `../out`，忽略 `--save_dir`（历史遗留），可顺手修正。
- 清理各脚本中 `transformers<5` 的兼容分支。
- 将本次验收脚本整理为仓库内可复跑的最小测试（需用户同意后再加）。
