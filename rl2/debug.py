import multiprocessing
import torch

MODEL = "/media/blazingbhavneek/Common/Code/sglangServer/Infer/Qwen/Qwen3-0.6B"

def run():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda:0")
    inner = model.model

    # Hook to capture what cos/sin the model actually passes to a layer
    captured = {}
    original_forward = inner.layers[0].forward

    def hooked_forward(hidden_states, **kwargs):
        pe = kwargs.get("position_embeddings")
        if pe is not None:
            captured["cos_shape"] = pe[0].shape
            captured["sin_shape"] = pe[1].shape
        return original_forward(hidden_states, **kwargs)

    inner.layers[0].forward = hooked_forward

    ids = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")
    with torch.no_grad():
        model(ids, use_cache=False)

    print("cos shape from real forward:", captured.get("cos_shape"))
    print("sin shape from real forward:", captured.get("sin_shape"))

    # Also check what rotary_emb returns directly
    pos = torch.arange(5).unsqueeze(0).cuda()
    dummy = torch.zeros(1, 5, 128, dtype=torch.bfloat16, device="cuda")
    cos, sin = inner.rotary_emb(dummy, pos)
    print("rotary_emb output cos shape:", cos.shape)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    run()
