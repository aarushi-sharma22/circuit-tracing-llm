from src.load_and_tokenize import load_qwen_instruct_4bit

def test_tokenize_and_render():
    model_id = "sshleifer/tiny-gpt2"  # tiny model for quick tests
    model, tokenizer = load_qwen_instruct_4bit(model_id)
    inputs = tokenizer("The capital of France is Paris.", return_tensors="pt")
    assert "input_ids" in inputs
    assert inputs["input_ids"].shape[1] > 0
