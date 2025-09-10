from src.hooks import get_activation_cache
from src.load_and_tokenize import load_qwen_instruct_4bit

def test_activation_cache():
    model_id = "sshleifer/tiny-gpt2"
    model, tokenizer = load_qwen_instruct_4bit(model_id)
    capture_activation_cache(model, tokenizer, "The capital of France is Paris.")
