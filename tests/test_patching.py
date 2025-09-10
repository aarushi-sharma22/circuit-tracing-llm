from src.patching import patch_hidden_states
from src.load_and_tokenize import load_qwen_instruct_4bit

def test_patching_runs():
    model_id = "sshleifer/tiny-gpt2"
    model, tokenizer = load_qwen_instruct_4bit(model_id)
    patch_hidden_states(model, tokenizer, "The capital of France is Paris.", patch_token="Paris")
