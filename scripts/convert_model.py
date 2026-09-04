import os
import sys
import json

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, merge_lora

warnings.filterwarnings('ignore', category=UserWarning)

try:
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    from transformers import Qwen3_5MoeTextConfig, Qwen3_5MoeForCausalLM
except ImportError as e:
    raise ImportError("导出 Qwen3.5 需要 transformers>=5.2，请先升级：pip install -U 'transformers>=5.2'") from e


def _qwen35_common_config(lm_config):
    return {
        "vocab_size": lm_config.vocab_size,
        "hidden_size": lm_config.hidden_size,
        "num_hidden_layers": lm_config.num_hidden_layers,
        "num_attention_heads": lm_config.num_attention_heads,
        "num_key_value_heads": lm_config.num_key_value_heads,
        "head_dim": lm_config.head_dim,
        "max_position_embeddings": lm_config.max_position_embeddings,
        "rms_norm_eps": lm_config.rms_norm_eps,
        "tie_word_embeddings": lm_config.tie_word_embeddings,
        "hidden_act": lm_config.hidden_act,
        "attention_bias": False,
        "attn_output_gate": lm_config.attn_output_gate,
        "layer_types": list(lm_config.layer_types),
        "full_attention_interval": lm_config.full_attention_interval,
        "linear_conv_kernel_dim": lm_config.linear_conv_kernel_dim,
        "linear_key_head_dim": lm_config.linear_key_head_dim,
        "linear_value_head_dim": lm_config.linear_value_head_dim,
        "linear_num_key_heads": lm_config.linear_num_key_heads,
        "linear_num_value_heads": lm_config.linear_num_value_heads,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": lm_config.rope_theta,
            "partial_rotary_factor": lm_config.partial_rotary_factor,
            "mrope_interleaved": lm_config.mrope_interleaved,
            "mrope_section": list(lm_config.mrope_section),
        },
        "bos_token_id": lm_config.bos_token_id,
        "eos_token_id": lm_config.eos_token_id,
    }


def _remap_moe_experts(state_dict, lm_config):
    new_sd = {k: v for k, v in state_dict.items() if 'experts.' not in k or 'gate.weight' in k or 'shared_expert' in k}
    for l in range(lm_config.num_hidden_layers):
        p = f'model.layers.{l}.mlp.experts'
        new_sd[f'{p}.gate_up_proj'] = torch.cat([
            torch.stack([state_dict[f'{p}.{e}.gate_proj.weight'] for e in range(lm_config.num_experts)]),
            torch.stack([state_dict[f'{p}.{e}.up_proj.weight'] for e in range(lm_config.num_experts)])
        ], dim=1)
        new_sd[f'{p}.down_proj'] = torch.stack([state_dict[f'{p}.{e}.down_proj.weight'] for e in range(lm_config.num_experts)])
    return new_sd


def _save_tokenizer(transformers_path):
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)
    # transformers>=5 需要显式 tokenizer_class，否则 AutoTokenizer 无法回载
    tokenizer_config_path = os.path.join(transformers_path, "tokenizer_config.json")
    json.dump(
        {**json.load(open(tokenizer_config_path, 'r', encoding='utf-8')), "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}},
        open(tokenizer_config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False
    )


def convert_torch2transformers_minimind(torch_path, transformers_path, dtype=torch.float16):
    MiniMindConfig.register_for_auto_class()
    MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    lm_model = MiniMindForCausalLM(lm_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    lm_model = lm_model.to(dtype)
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    _save_tokenizer(transformers_path)
    print(f"模型已保存为 Transformers-MiniMind 格式: {transformers_path}")


def convert_torch2transformers(torch_path, transformers_path, dtype=torch.float16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    common_config = _qwen35_common_config(lm_config)
    if not lm_config.use_moe:
        qwen_config = Qwen3_5TextConfig(**common_config, intermediate_size=lm_config.intermediate_size)
        qwen_model = Qwen3_5ForCausalLM(qwen_config)
    else:
        qwen_config = Qwen3_5MoeTextConfig(
            **common_config,
            num_experts=lm_config.num_experts,
            num_experts_per_tok=lm_config.num_experts_per_tok,
            moe_intermediate_size=lm_config.moe_intermediate_size,
            shared_expert_intermediate_size=lm_config.shared_expert_intermediate_size,
            norm_topk_prob=lm_config.norm_topk_prob,
            router_aux_loss_coef=lm_config.router_aux_loss_coef,
        )
        qwen_model = Qwen3_5MoeForCausalLM(qwen_config)
        expected = qwen_model.state_dict()
        if any(k.endswith('experts.gate_up_proj') for k in expected):
            state_dict = _remap_moe_experts(state_dict, lm_config)

    missing, unexpected = qwen_model.load_state_dict(state_dict, strict=False)
    missing = [k for k in missing if not k.startswith('mtp.')]
    if missing or unexpected:
        raise RuntimeError(f'导出权重未严格对齐 Qwen3.5：missing={missing} unexpected={unexpected}')
    qwen_model = qwen_model.to(dtype)
    qwen_model.save_pretrained(transformers_path)
    model_params = sum(p.numel() for p in qwen_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    _save_tokenizer(transformers_path)
    print(f"模型已保存为 Transformers 格式: {transformers_path}")


def convert_transformers2torch(transformers_path, torch_path):
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    torch.save({k: v.cpu().half() for k, v in model.state_dict().items() if not k.startswith('mtp.')}, torch_path)
    print(f"模型已保存为 PyTorch 格式: {torch_path}")


def convert_merge_base_lora(base_torch_path, lora_path, merged_torch_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lm_model = MiniMindForCausalLM(lm_config).to(device)
    state_dict = torch.load(base_torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    apply_lora(lm_model)
    merge_lora(lm_model, lora_path, merged_torch_path)
    print(f"LoRA 已合并并保存为基模结构 PyTorch 格式: {merged_torch_path}")


def convert_jinja_to_json(jinja_path):
    with open(jinja_path, 'r') as f: template = f.read()
    escaped = json.dumps(template)
    print(f'"chat_template": {escaped}')


def convert_json_to_jinja(json_file_path, output_path):
    with open(json_file_path, 'r') as f: config = json.load(f)
    template = config['chat_template']
    with open(output_path, 'w') as f: f.write(template)
    print(f"模板已保存为 jinja 文件: {output_path}")


if __name__ == '__main__':
    lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, max_seq_len=8192, use_moe=False)

    # convert torch to transformers
    torch_path = f"../out/full_sft_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    transformers_path = '../minimind-3.5'
    convert_torch2transformers(torch_path, transformers_path)

    # # merge lora
    # base_torch_path = f"../out/full_sft_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    # lora_path = f"../out/lora_identity_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    # merged_torch_path = f"../out/merge_identity_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    # convert_merge_base_lora(base_torch_path, lora_path, merged_torch_path)

    # convert_transformers2torch(transformers_path, torch_path)
    # convert_json_to_jinja('../model/tokenizer_config.json', '../model/chat_template.jinja')
    # convert_jinja_to_json('../model/chat_template.jinja')
