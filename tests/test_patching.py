from load_and_tokenize import load_qwen_instruct_4bit, apply_qwen_chat_template, batch_tokenize
from hooks import get_activation_cache
from patching import run_with_patched_activation

def test_patching_runs():
    model_id = "sshleifer/tiny-gpt2"
    # Force GPTQ to skip bitsandbytes
    model, tokenizer = load_qwen_instruct_4bit(model_id, use_gptq=True)

    questions = ["Is stealing wrong?", "Is kindness good?"]

    # GPT-2 has no chat template → fall back gracefully
    try:
        rendered = apply_qwen_chat_template(tokenizer, questions)
    except (AttributeError, ValueError):
        rendered = questions

    batch = batch_tokenize(tokenizer, rendered)

    cache = get_activation_cache(model, batch)
    assert isinstance(cache, dict)
    assert len(cache) > 0

    # Pick first available layer from cache
    layer_name = list(cache.keys())[0]

    logits = run_with_patched_activation(
        model, batch, cache,
        layer_name=layer_name,
        source_idx=0,
        target_idx=1,
    )

    # Check output shape [batch, seq, vocab]
    assert logits.ndim == 3
    assert logits.shape[0] == len(questions)
