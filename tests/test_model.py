"""
MiniMind (Qwen3.5 骨架) 最小回归测试，CPU 可跑：
    python tests/test_model.py        # 或 python -m pytest tests/
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM, chunk_gated_delta_rule, recurrent_gated_delta_rule
from model.model_lora import apply_lora

torch.manual_seed(0)
GREEDY = dict(do_sample=False, temperature=1.0, top_p=1.0, top_k=0)


def small_config(**kwargs):
    return MiniMindConfig(hidden_size=256, num_hidden_layers=8, max_position_embeddings=256, **kwargs)


def test_default_config_matches_qwen35():
    cfg = MiniMindConfig()
    assert cfg.layer_types == ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"]
    assert cfg.rope_theta == 1e7 and cfg.partial_rotary_factor == 0.25 and cfg.attn_output_gate
    assert sum(cfg.mrope_section) == cfg.rotary_dim // 2
    assert MiniMindConfig(max_seq_len=64).max_position_embeddings == 32768


def test_chunk_and_recurrent_delta_rule_agree():
    B, T, Hk, Hv, d = 2, 17, 4, 8, 16
    q = torch.randn(B, T, Hk, d).repeat_interleave(Hv // Hk, dim=2)
    k = torch.randn(B, T, Hk, d).repeat_interleave(Hv // Hk, dim=2)
    v, g, beta = torch.randn(B, T, Hv, d), -torch.rand(B, T, Hv), torch.rand(B, T, Hv)
    o1, s1 = chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=True)
    o2, s2 = recurrent_gated_delta_rule(q, k, v, g, beta, output_final_state=True)
    assert (o1 - o2).abs().max() < 1e-4 and (s1 - s2).abs().max() < 1e-4


def test_prefill_matches_stepwise_decode():
    for use_moe in (False, True):
        model = MiniMindForCausalLM(small_config(use_moe=use_moe)).eval()
        x = torch.randint(0, 6400, (1, 12))
        full = model(x, use_cache=True).logits[:, -1]
        past, last = None, None
        for i in range(x.shape[1]):
            out = model(x[:, i:i + 1], past_key_values=past, use_cache=True)
            past, last = out.past_key_values, out.logits[:, -1]
        assert (full - last).abs().max() < 1e-3


def test_generate_cache_equals_no_cache():
    model = MiniMindForCausalLM(small_config()).eval()
    prompt = torch.randint(3, 6400, (1, 6))
    y1 = model.generate(prompt, max_new_tokens=8, use_cache=True, **GREEDY)
    y2 = model.generate(prompt, max_new_tokens=8, use_cache=False, **GREEDY)
    assert torch.equal(y1, y2)


def test_padding_masks():
    model = MiniMindForCausalLM(small_config()).eval()
    a = torch.randint(3, 6400, (1, 8))
    right = torch.cat([a, torch.zeros(1, 4, dtype=torch.long)], 1)
    right_mask = torch.tensor([[1] * 8 + [0] * 4])
    assert (model(a).logits - model(right, attention_mask=right_mask).logits[:, :8]).abs().max() < 1e-3
    b = torch.randint(3, 6400, (1, 5))
    batch = torch.cat([a, torch.cat([torch.zeros(1, 3, dtype=torch.long), b], 1)], 0)
    left_mask = torch.tensor([[1] * 8, [0] * 3 + [1] * 5])
    yb = model.generate(batch, attention_mask=left_mask, max_new_tokens=5, **GREEDY)
    assert torch.equal(yb[1, 8:], model.generate(b, max_new_tokens=5, **GREEDY)[0, 5:])


def test_training_step_all_params_get_grad():
    for use_moe in (False, True):
        model = MiniMindForCausalLM(small_config(use_moe=use_moe)).train()
        x = torch.randint(0, 6400, (2, 40))
        out = model(x, labels=x)
        (out.loss + out.aux_loss).backward()
        assert all(p.grad is not None for p in model.parameters())


def test_lora_targets_named_modules():
    model = MiniMindForCausalLM(small_config())
    apply_lora(model, rank=4)
    names = [n for n, m in model.named_modules() if hasattr(m, 'lora')]
    assert any(n.endswith('self_attn.q_proj') for n in names)
    assert any(n.endswith('linear_attn.in_proj_qkv') for n in names)
    assert not any(n.endswith('mlp.gate') for n in names)


def test_export_to_qwen35_classes():
    try:
        from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM, Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM
    except ImportError:
        print("skip: transformers 版本不含 Qwen3.5 类")
        return
    from scripts.convert_model import _qwen35_common_config, _remap_moe_experts
    for use_moe in (False, True):
        cfg = small_config(use_moe=use_moe)
        mm = MiniMindForCausalLM(cfg).eval()
        common = _qwen35_common_config(cfg)
        if use_moe:
            qm = Qwen3_5MoeForCausalLM(Qwen3_5MoeTextConfig(
                **common, num_experts=cfg.num_experts, num_experts_per_tok=cfg.num_experts_per_tok,
                moe_intermediate_size=cfg.moe_intermediate_size, shared_expert_intermediate_size=cfg.shared_expert_intermediate_size,
                norm_topk_prob=cfg.norm_topk_prob, router_aux_loss_coef=cfg.router_aux_loss_coef))
            sd = _remap_moe_experts(mm.state_dict(), cfg)
        else:
            qm = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**common, intermediate_size=cfg.intermediate_size))
            sd = mm.state_dict()
        expected = qm.state_dict()
        assert not [k for k in expected if not k.startswith('mtp.') and k not in sd]
        assert not [k for k in sd if k not in expected]
        qm.load_state_dict(sd, strict=False)
        x = torch.randint(0, 6400, (1, 8))
        with torch.no_grad():
            assert (mm(x).logits.float() - qm(x).logits.float()).abs().max() < 1e-3


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn(); print(f'PASS {name}')
