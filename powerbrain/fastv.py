"""FastV-style visual token pruning inside the language model (Chen et al., ECCV 2024), as a training-free baseline
for whole-image inputs: after `layer` decoder layers, only the `keep` visual tokens (a count, or as a float the fraction
of the input's visual tokens) that the last token attends to most
(mean over heads, attention of decoder layer `layer` - 1) stay in the sequence; the others are dropped for all
remaining layers. Text tokens are never dropped. Qwen3-VL adds multi-level visual features to the hidden states after
its first three decoder layers (DeepStack), so `layer` must be at least 3.

    model = core.MLLM(path, lora=adapter); fastv.install(model, layer=3, keep=576); model.answer(content, n_options)
"""
from __future__ import annotations

import torch


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def last_row_attention(block, hidden, position_embeddings):
    """Attention of the last token over the whole sequence inside decoder layer `block`, mean over heads: (batch, seq).
    Computed for that one query row only, so it costs next to nothing (the stock attention does not return weights)."""
    attn = block.self_attn
    h = block.input_layernorm(hidden)
    b, t, _ = h.shape
    q = attn.q_norm(attn.q_proj(h[:, -1:]).view(b, 1, -1, attn.head_dim)).transpose(1, 2)  # (b, heads, 1, d)
    k = attn.k_norm(attn.k_proj(h).view(b, t, -1, attn.head_dim)).transpose(1, 2)  # (b, kv heads, t, d)
    cos, sin = (v.unsqueeze(1) for v in position_embeddings)  # (b, 1, t, d)
    q = q * cos[:, :, -1:] + rotate_half(q) * sin[:, :, -1:]
    k = k * cos + rotate_half(k) * sin
    k = k.repeat_interleave(attn.num_key_value_groups, dim=1)
    scores = (q @ k.transpose(-1, -2)) * attn.scaling  # (b, heads, 1, t)
    return torch.softmax(scores.float(), dim=-1).mean(1)[:, 0]


def install(mllm, layer=3, keep=576):
    """Adds the pruning hooks to a core.MLLM in place. Returns a dict that, after each forward pass, holds
    "kept" (visual tokens kept) and "visual" (visual tokens in the input); its "keep" entry can be changed between calls."""
    assert layer >= 3, "prune after the DeepStack layers"
    top = mllm.model
    layers = top.model.language_model.layers
    state = {"keep": keep}

    def remember_input(module, args, kwargs):
        ids = kwargs["input_ids"]
        assert ids.shape[0] == 1, "one question at a time"
        state["visual_mask"] = ids[0] == mllm.image_token_id
        state["index"] = None

    def rank(module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        row = last_row_attention(module, hidden, kwargs["position_embeddings"])[0]
        visual = state["visual_mask"].nonzero()[:, 0]
        state["visual"] = int(visual.numel())
        keep = state["keep"]
        if isinstance(keep, float):  # a fraction of the visual tokens of this input
            keep = max(1, int(round(keep * visual.numel())))
        if visual.numel() <= keep:
            state["kept"] = int(visual.numel())
            return
        best = visual[row[visual].topk(keep).indices]
        mask = ~state["visual_mask"]
        mask[best] = True
        state["index"] = mask.nonzero()[:, 0]
        state["kept"] = keep

    def prune(module, args, kwargs):
        index = state["index"]
        if index is None:
            return None
        hidden = args[0] if args else kwargs["hidden_states"]
        full = state["visual_mask"].shape[0]
        if hidden.shape[1] == full:  # the first pruned layer receives the full sequence
            hidden = hidden[:, index]
        kwargs = dict(kwargs)
        cos, sin = kwargs["position_embeddings"]
        kwargs["position_embeddings"] = (cos[:, index], sin[:, index])
        mask = kwargs.get("attention_mask")
        if mask is not None:
            kwargs["attention_mask"] = mask[:, :, index][:, :, :, index]
        if kwargs.get("position_ids") is not None:
            kwargs["position_ids"] = kwargs["position_ids"][..., index]
        if kwargs.get("cache_position") is not None:
            kwargs["cache_position"] = kwargs["cache_position"][index]
        if args:
            return (hidden, *args[1:]), kwargs
        kwargs["hidden_states"] = hidden
        return args, kwargs

    top.register_forward_pre_hook(remember_input, with_kwargs=True)
    layers[layer - 1].register_forward_pre_hook(rank, with_kwargs=True)
    for block in layers[layer:]:
        block.register_forward_pre_hook(prune, with_kwargs=True)
    return state
