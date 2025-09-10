from src.analyse_visualise import sweep_patch_layers
from src.load_and_tokenize import load_qwen_instruct_4bit

def test_sweep_patch_layers_logic():
    model_id = "sshleifer/tiny-gpt2"
    # Force normal HF load (not GPTQ/bitsandbytes) for dev testing
    model, tokenizer = load_qwen_instruct_4bit(model_id)
    sweep_patch_layers(model, tokenizer, "The capital of France is Paris.", layer_range=[0])
