from load_and_tokenize import load_qwen_instruct_4bit, apply_qwen_chat_template, batch_tokenize
from hooks import get_activation_cache

def test_activation_cache():
    model_id = "sshleifer/tiny-gpt2"
    model, tokenizer = load_qwen_instruct_4bit(model_id, use_gptq=False)

    # GPT-2 has no chat template → fallback
    try:
        rendered = apply_qwen_chat_template(tokenizer, ["Is murder wrong?"])
    except AttributeError:
        rendered = ["Is murder wrong?"]

    batch = batch_tokenize(tokenizer, rendered)

    cache = get_activation_cache(model, batch)

    assert isinstance(cache, dict)
    assert any("mlp_in" in k for k in cache.keys())
    assert any("resid_out" in k for k in cache.keys())

    for v in cache.values():
        assert v.ndim in (2, 3)  # some hooks give [batch, seq, hidden], others may flatten
