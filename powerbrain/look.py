"""Look-on-doubt (P21): one model, one adapter, an answer in up to two steps.

  step 1   no image. Input: a branch-state token, the evidence list of the detector branch and the question. The model
           answers with an option letter, or with X when the list alone does not allow a reliable answer.
  step 2   only after X. The dialogue continues with a thumbnail of the whole image and the model gives the final
           answer. The computation of step 1 is reused (key-value cache); only the new tokens are processed.

Branch-state token: one learned input token. A small network maps the label-free state scalars of the detector branch
(branchstate.py: how far the image lies from the inputs the branch was trained on) to the embedding space of the
language model. It lets step 1 tell an empty list on a normal image from an empty list on an image the branch cannot
handle, without running the vision encoder.

Training targets (looklabel.py): step 1 answers the reference letter when the rule reading of the out-of-fold list is
correct, otherwise X; step 2 always answers the reference letter.

Evidence tokenizer (P22, list wording 4 of core.evidence_content): every candidate enters as a learned class token, its
confidence as two digits, and a learned position token, about four tokens instead of a line of text. The class tokens
are one free embedding per class; the position token is a small network of the box corners. A read-back task (write
the candidates out as text) is mixed into training so that the language model learns what the tokens say.
"""
from __future__ import annotations

import torch
from torch import nn

import core

LOOK = "X"
STATE_SLOT = "<|box_end|>"  # a special token of the tokenizer; the branch-state token is written at its position
CLASS_SLOT = core.EVIDENCE_SLOT  # "<|box_start|>": the class token of a candidate (list wording 4)
BOX_SLOT = "<|quad_start|>"  # the position token of a candidate (list wording 4)
READBACK = "请把以上检测候选逐行写成文字。"
READBACK_SHARE = 0.2  # share of the training samples that are read-back tasks when the list is tokenised
STATE_DIM = 4
STATE_CENTRE = 4.5  # the state scalars are log squared distances; clean images sit around 4, failing ones above 6
INSTRUCTION = "请只回答选项字母；如果仅凭检测清单无法可靠判断，回答 X。"
STEP2_TEXT = "以上是整幅图像的缩略图。请结合图像回答选项字母。"
# open-ended questions (openform.py): the question keeps its own answer instruction, the way out is appended
OPEN_INSTRUCTION = "如果仅凭检测清单无法可靠判断，只回答 X。"
STEP2_OPEN = "以上是整幅图像的缩略图。请结合图像回答上面的问题。"


class _Size:
    """Stands in for an image where only its size is needed (step 1 never opens the image)."""

    def __init__(self, width, height):
        self.size = (width, height)


def look_question(item):
    if "answer_text" in item:
        return item["question"] + OPEN_INSTRUCTION
    assert item["question"].endswith("请只回答选项字母。"), item["id"]
    return item["question"][: -len("请只回答选项字母。")] + INSTRUCTION


def asks_to_look(text):
    """A written step-1 answer that is a request for the image."""
    return text.strip().startswith(LOOK)


def step1_content(item, given):
    head = [{"type": "text", "text": "检测分支状态：" + STATE_SLOT + "\n"}] if given.get("state") is not None else []
    evidence = core.evidence_content(_Size(item["width"], item["height"]), {**given, "thumb_tokens": 0, "mark": False})
    return head + evidence + [{"type": "text", "text": look_question(item)}]


def messages(item, given, step, image=None):
    """The dialogue up to the point where the model has to answer. `image` is needed for step 2 only."""
    out = [{"role": "user", "content": step1_content(item, given)}]
    if step == 2:
        thumb = core.resize_to_token_cap(image, given.get("thumb_tokens") or core.THUMB_TOKENS)
        out += [{"role": "assistant", "content": [{"type": "text", "text": LOOK}]},
                {"role": "user", "content": [{"type": "image", "image": thumb}, {"type": "text", "text": STEP2_OPEN if "answer_text" in item else STEP2_TEXT}]}]
    return out


class StateProjector(nn.Module):
    """State scalars of the detector branch -> one input token of the language model."""

    def __init__(self, dim, hidden=256):
        super().__init__()
        self.base = nn.Parameter(torch.zeros(dim))
        self.mlp = nn.Sequential(nn.Linear(STATE_DIM, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    @torch.no_grad()
    def init_base(self, embed_table, tokenizer, word="状态"):
        """The token starts as the mean input embedding of a neutral word; the dependence on the scalars is learned."""
        ids = tokenizer.encode(word, add_special_tokens=False)
        self.base.copy_(embed_table[torch.tensor(ids, device=embed_table.device)].float().mean(0))

    def forward(self, state):
        return (self.base + self.mlp(state.float().reshape(-1, STATE_DIM) - STATE_CENTRE)).reshape(1, -1)


class SlotInjector:
    """Writes given vectors into the input embeddings at the positions of placeholder tokens (works for text-only and
    for multimodal forward passes, in training and in generation). `values`: {placeholder token id: [n, dim] vectors for
    its n occurrences in reading order, or [1, dim] to be repeated}."""

    def __init__(self, embedding_module):
        self.values = None
        embedding_module.register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        if not self.values or args[0].shape[-1] == 1:  # nothing to write, or one newly generated token (the prompt came in the first call)
            return None
        out = None
        for slot_id, vectors in self.values.items():
            mask = args[0] == slot_id
            n = int(mask.sum())
            if not n:
                continue
            assert vectors.shape[0] in (1, n), (slot_id, vectors.shape, n)
            out = output.clone() if out is None else out
            out[mask] = vectors.to(out.dtype).expand(n, -1)
        return out


def readback_text(given, width, height):
    """The candidates written out as text lines (what the read-back task has to produce)."""
    pct = lambda v, total: int(round(100 * v / total))
    return "\n".join(
        f"{k + 1}. {d['name']}，置信度 {d['conf']:.2f}，位置（左 {pct(d['box'][0], width)}%，上 {pct(d['box'][1], height)}%，右 {pct(d['box'][2], width)}%，下 {pct(d['box'][3], height)}%）"
        for k, d in enumerate(given["dets"]))


def readback_messages(item, given):
    evidence = core.evidence_content(_Size(item["width"], item["height"]), {**given, "thumb_tokens": 0, "mark": False})
    return [{"role": "user", "content": evidence + [{"type": "text", "text": READBACK}]}]


def slot_inputs(given, width, height):
    """(class indices [K], box corners as fractions [K, 4]) for the token slots of a tokenised list."""
    import evtok

    dets = given["dets"]
    cls = torch.tensor([evtok.CLASSES.index(d["cls"]) for d in dets], dtype=torch.long)
    box = torch.tensor([[d["box"][0] / width, d["box"][1] / height, d["box"][2] / width, d["box"][3] / height] for d in dets], dtype=torch.float32).reshape(-1, 4)
    return cls, box.clamp(0.0, 1.0)


class EvidenceTokenizer(nn.Module):
    """Class and position tokens of the candidates (list wording 4)."""

    def __init__(self, dim, n_classes, hidden=512, bands=8):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(n_classes, dim))
        self.box_base = nn.Parameter(torch.zeros(dim))
        self.mlp = nn.Sequential(nn.Linear(4 * 2 * bands, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.register_buffer("freq", (2.0 ** torch.arange(bands)) * torch.pi, persistent=False)

    @torch.no_grad()
    def init_from_names(self, embed_table, tokenizer, names, seed=0):
        """Class tokens start at the mean input embedding of the class name plus a class-specific random direction of half
        its norm: names that share most characters (as several defect classes do) must not start as the same token."""
        g = torch.Generator().manual_seed(seed)
        for k, name in enumerate(names):
            ids = tokenizer.encode(name, add_special_tokens=False)
            mean = embed_table[torch.tensor(ids, device=embed_table.device)].float().mean(0).cpu()
            noise = torch.randn(mean.shape, generator=g)
            self.cls[k] = (mean + 0.5 * mean.norm() * noise / noise.norm()).to(self.cls.device)
        ids = tokenizer.encode("位置", add_special_tokens=False)
        self.box_base.copy_(embed_table[torch.tensor(ids, device=embed_table.device)].float().mean(0))

    def forward(self, cls, box):
        x = box.float()[..., None] * self.freq
        x = torch.cat([x.sin(), x.cos()], dim=-1).flatten(1)
        return self.cls[cls], self.box_base + self.mlp(x)
