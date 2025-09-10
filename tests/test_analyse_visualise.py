import pytest
from load_and_tokenize import load_qwen_instruct_4bit, apply_qwen_chat_template, batch_tokenize
from hooks import get_activation_cache
from analyse_visualise import sweep_patch_layers


def test_skip_if_no_quant_libs():
    # This test will SKIP if quantization deps are missing, instead of failing
    pytest.importorskip("auto_gptq")
    pytest.importorskip("bitsandbytes")


def test_sweep_patch_layers_logic():
    model_id = "sshleifer/tiny-gpt2"
    # Force normal HF load (not GPTQ/bitsandbytes) for dev testing
    model, tokenizer = load_qwen_instruct_4bit(model_id, use_gptq=False)
    model.tokenizer = tokenizer  # required for yes/no ids

    questions = ["Is stealing wrong?", "Is kindness good?"]
    rendered = apply_qwen_chat_template(tokenizer, questions)
    batch = batch_tokenize(tokenizer, rendered)

    cache = get_activation_cache(model, batch)

    effects = sweep_patch_layers(
        model, batch, cache,
        source_idx=0, target_idx=1,
        kind="mlp_out"
    )

    assert isinstance(effects, list)
    assert len(effects) == model.config.num_hidden_layers
