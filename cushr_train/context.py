#!/usr/bin/env python3


import inspect

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

ENCODERS = {}

# Byte values are shifted up by one so that 0 can mean "padding" and never
# collides with a real character.
PAD_CHAR = 0
CHAR_OFFSET = 1


def register(name):
    def wrap(cls):
        cls.name = name
        ENCODERS[name] = cls
        return cls
    return wrap


def get(name, **kw):
    if name not in ENCODERS:
        raise KeyError(f"unknown encoder {name!r}; available: {sorted(ENCODERS)}")
    cls = ENCODERS[name]
    # Keep only the kwargs this encoder actually declares. train.py passes the
    # union of every encoder's options, so a new encoder can take `base_dim` or
    # `heads` without char_bilstm having to grow parameters it ignores.
    accepted = inspect.signature(cls.__init__).parameters
    return cls(**{k: v for k, v in kw.items() if k in accepted})


@register("char_bilstm")
class CharBiLSTMEncoder(nn.Module):
    # takes a batch of raw character sequences
    # outputs fixed size context vector for each node/candidate word

    def __init__(self, n_chars=256, char_dim=32, hidden=128, layers=2,
                 out_dim=96, dropout=0.0):
        super().__init__()
        self.out_dim = out_dim

        # convert raw character integers into dense vector
        # CHAR_OFFSET is 1 to ensure that no character ever encodes to a 0
        # 0 in raw byte value means null padding space
        self.emb = nn.Embedding(n_chars + CHAR_OFFSET, char_dim, padding_idx=PAD_CHAR)

        # from the pytorch library torch.nn which is standard lib for building neural network
        # LSTM is type of RNN to process data and remember context as you maintain memory

        # char_dim is the length of vector for every character = 32
        # hidden is length of vector it will remember at every stage = 128
        # num_layers is how many LSTMs stacked = 2
        # bidirectional to create 2 LSTMs in parallel running L-R and R-L, final output will be 256
        # batch_first means to format data in the form of (batch size=64 sent, sequence length=40 char/sent, features=32 nums/char)
        # dropout if layers > 1 else 0.0. drouput is a technique to randomly turn off neurons to prevent memorizing/overfitting
            # applies dropout value if 2+ layers
        self.lstm = nn.LSTM(char_dim, hidden, num_layers=layers,
                            bidirectional=True, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)

        # each endpoint has forward/backward lstm vector so 128+128 = 256
        # 2 endpoints -> 512 = 4*hidden = size of tensor entering final linear layer
        self.span = nn.Linear(4 * hidden, out_dim)

    def forward(self, chars, char_len, node_sent, span_start, span_end,
                char_ok, base=None):
        """chars [B, L] int64 (0 = pad); returns [n_nodes, out_dim].

        `base` (the featurizer's node vectors) is accepted and ignored so that
        this and the lattice-attention encoders share one call signature and
        node_vectors needs no branch -- the same trick BiaffineEdgeScorer.forward
        plays with `ids` (model.py)."""
        h = self.emb(chars)
        lengths = char_len.detach().cpu().clamp(min=1)
        packed = pack_padded_sequence(h, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        H, _ = pad_packed_sequence(out, batch_first=True, total_length=chars.shape[1])

        # first and last character together give me the left context + word and right context + word
        first = H[node_sent, span_start]  # [n, 2*hidden]
        last = H[node_sent, span_end - 1]  # [n, 2*hidden]

        # concatenate boundary char states together
        ctx = self.span(torch.cat([first, last], dim=-1))
        # Boundary nodes carry no signal, matching every other featurizer.
        return ctx.masked_fill(~char_ok.unsqueeze(-1), 0.0)


# put model together
class ContextualBiaffine(nn.Module):

    def __init__(self, featurizer, encoder, scorer):
        super().__init__()

        # takes existing pre computed features thru featurizer
        self.featurizer = featurizer
        self.encoder = encoder
        self.scorer = scorer


    def node_vectors(self, feats, ids, batch):
        base = (self.featurizer(feats, ids) if self.featurizer is not None
                else feats)
        # `base` is passed so an encoder can attend over the featurized nodes
        # themselves. This is not a cycle: base depends only on feats/ids, and
        # the concatenation below keeps an un-attended copy, so an attention
        # encoder only has to learn a correction.
        ctx = self.encoder(batch["chars"], batch["char_len"],
                           batch["node_sent"], batch["span_start"],
                           batch["span_end"], batch["char_ok"], base)

        # concatenates base features with new context-aware features generated by encoder
        return torch.cat([base, ctx], dim=-1)

    # pass in the concatenated vector to scorer (Biaffine) to predict probability that node A
    # connects to B in the final segmentation graph
    def forward(self, feats, src, dst, ids=None, batch=None):
        if batch is None:
            raise ValueError(
                "ContextualBiaffine needs the batch dict for character "
                "context; call model(feats, src, dst, ids, batch)")
        return self.scorer.edge_scores(self.node_vectors(feats, ids, batch),
                                       src, dst)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# runs the neural network once over the entire corpus in mini batches
# extracts final 192 dim vectors for each node and freeze them into numpy arr
@torch.no_grad()
def materialize_contextual(model, store, device, out, sent_chunk=256):
    from dataset import collate

    model.eval()
    n_sent = len(store.sent_off) - 1
    for lo in range(0, n_sent, sent_chunk):
        ids = np.arange(lo, min(lo + sent_chunk, n_sent), dtype=np.int64)
        b = collate(store, ids)
        t = {
            "chars": torch.as_tensor(b["chars"], dtype=torch.int64, device=device),
            "char_len": torch.as_tensor(b["char_len"], dtype=torch.int64, device=device),
            "node_sent": torch.as_tensor(b["node_sent"], dtype=torch.int64, device=device),
            "span_start": torch.as_tensor(b["span_start"], dtype=torch.int64, device=device),
            "span_end": torch.as_tensor(b["span_end"], dtype=torch.int64, device=device),
            "char_ok": torch.as_tensor(b["char_ok"], dtype=torch.bool, device=device),
        }
        feats = torch.as_tensor(np.asarray(b["feats"]), dtype=torch.float32,
                                device=device)
        nid = (torch.as_tensor(np.asarray(b["ids"]), dtype=torch.int64,
                               device=device)
               if b.get("ids") is not None else None)
        out[b["global_node"]] = model.node_vectors(feats, nid, t).cpu().numpy()
    model.train()
    return out


# ---------------------------------------------------------------------------
# Lattice self-attention (TransLIST-style), with a four-position span bias.
#
# The biaffine scorer is first-order: an edge score is a function of exactly two
# node vectors, so the model cannot express "this candidate is wrong given a word
# ten characters away". TransLIST gets that from self-attention over all
# candidate words of a sentence, biased by how their spans sit relative to one
# another. This is the same idea, applied to lattice nodes.
#
# The output is still a per-node vector that depends on the sentence but never on
# the path, so the materialise-then-decode contract holds and the C++/CUDA
# decoders are untouched (see dataset.py collate and train.py write_materialized).
# ---------------------------------------------------------------------------


def span_groups(counts, budget):
    """Partition sentence indices so each group's padded attention fits.

    Cost of a group is len(group) * max_nodes(group)**2. Sentences are sorted by
    node count first, so a group is also near-uniform in length and wastes little
    padding. Attention never crosses a sentence boundary, so the grouping cannot
    change the result -- only the peak memory. That is what lets training
    (batch 64) and materialisation (chunk 256) produce the same vectors.

    "The same" is exact in arithmetic but not bitwise in float32: padding width
    and group size change the reduction order inside attention. Measured by
    test_lattice_attn.py, the disagreement is ~1e-6 in float32 and ~1.7e-15 in
    float64 -- i.e. rounding, not leakage. Run that test with --double after any
    change here; a real bug survives the dtype change, rounding does not.
    """
    order = sorted(range(len(counts)), key=lambda i: counts[i])
    groups, cur, cur_max = [], [], 0
    for i in order:
        c = max(1, int(counts[i]))
        m = max(cur_max, c)
        if cur and (len(cur) + 1) * m * m > budget:
            groups.append(cur)
            cur, cur_max = [i], c
        else:
            cur.append(i)
            cur_max = m
    if cur:
        groups.append(cur)
    return groups


class _AttnBlock(nn.Module):
    """Pre-LN self-attention + FFN, with a per-head additive attention bias."""

    def __init__(self, d_model, heads, ffn, dropout):
        super().__init__()
        self.heads, self.dh = heads, d_model // heads
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ffn), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(ffn, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        B, N, D = x.shape
        h = self.n1(x)
        shape = (B, N, self.heads, self.dh)
        q = self.q(h).view(shape).transpose(1, 2)
        k = self.k(h).view(shape).transpose(1, 2)
        v = self.v(h).view(shape).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, N, D)))
        return x + self.drop(self.ff(self.n2(x)))


@register("lattice_attn")
class LatticeAttentionEncoder(nn.Module):
    """Global self-attention over the candidate nodes of one sentence.

    Every candidate attends to every other candidate in its sentence. The
    attention logits carry a learned bias built from the four signed distances
    between the two candidates' character spans, which is what lets the model
    read the relation between them:

        es < 0                  i ends before j starts  -- i may precede j
        es >= 0 and se <= 0     the spans OVERLAP       -- mutually exclusive
        ss == 0                 same head               -- competing splits
                                                           from one start point

    One scalar distance cannot express that family; four can. Sanskrit needs it
    because sandhi makes neighbouring words share characters, so "adjacent" is
    not `end_i + 1 == start_j` but a family of relations.
    """

    use_chars = False

    def __init__(self, base_dim=96, n_chars=256, char_dim=32, hidden=128,
                 layers=2, out_dim=96, dropout=0.1, heads=8, max_rel=64,
                 pair_budget=1_000_000, span_bias=True, ffn_mult=2):
        super().__init__()
        if hidden % heads:
            raise ValueError(f"--ctx-hidden {hidden} must divide by heads {heads}")
        self.out_dim, self.d_model, self.heads = out_dim, hidden, heads
        self.max_rel, self.pair_budget = max_rel, int(pair_budget)

        # lattice_attn_char keeps the proven char-BiLSTM span vector and feeds it
        # to attention alongside the featurizer output; lattice_attn replaces it.
        self.char_enc = (CharBiLSTMEncoder(n_chars=n_chars, char_dim=char_dim,
                                           hidden=hidden, layers=layers,
                                           out_dim=out_dim)
                         if self.use_chars else None)
        self.in_proj = nn.Linear(base_dim + (out_dim if self.use_chars else 0),
                                 hidden)
        self.blocks = nn.ModuleList([_AttnBlock(hidden, heads, ffn_mult * hidden,
                                                dropout)
                                     for _ in range(layers)])
        self.norm_out = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, out_dim)

        # Four tables, one per distance, each emitting a per-head scalar. A
        # linear fusion of four embeddings is a sum of four fused tables, so the
        # fusion matrix is folded in and nothing wider than [G, N, N, heads] is
        # ever materialised. Zero-init: training starts as plain attention and
        # grows the geometry rather than fighting a random bias.
        self.span_tables = None
        if span_bias:
            self.span_tables = nn.ModuleList(
                [nn.Embedding(2 * max_rel + 1, heads) for _ in range(4)])
            for t in self.span_tables:
                nn.init.zeros_(t.weight)

    def _bias(self, S, E, allow):
        """S, E: [G, N] span start / inclusive end. -> [G, heads, N, N] float."""
        G, N = S.shape
        neg = torch.zeros(G, 1, N, N, device=S.device)
        # Disallowed KEYS are -inf: padding slots, and boundary nodes whose spans
        # are fabricated (dataset.py clamps them when char_ok is false).
        neg = neg.masked_fill(~allow[:, None, None, :], float("-inf"))
        # ... but every row keeps its own diagonal, so no softmax row is entirely
        # -inf. An all--inf row yields NaN, which the residual stream then spreads
        # through the whole batch silently.
        eye = torch.eye(N, dtype=torch.bool, device=S.device)
        neg = neg.masked_fill(eye[None, None], 0.0)
        if self.span_tables is None:
            return neg.expand(G, self.heads, N, N)

        m = self.max_rel
        d = (S[:, :, None] - S[:, None, :],      # ss
             S[:, :, None] - E[:, None, :],      # se
             E[:, :, None] - S[:, None, :],      # es
             E[:, :, None] - E[:, None, :])      # ee
        bias = 0
        for table, dk in zip(self.span_tables, d):
            bias = bias + table(dk.clamp(-m, m) + m)
        return bias.permute(0, 3, 1, 2) + neg

    def forward(self, chars, char_len, node_sent, span_start, span_end,
                char_ok, base=None):
        if base is None:
            raise ValueError(f"{type(self).__name__} needs the featurizer's node "
                             "vectors; ContextualBiaffine passes them as `base`")
        x = base
        if self.char_enc is not None:
            x = torch.cat([x, self.char_enc(chars, char_len, node_sent,
                                            span_start, span_end, char_ok)], -1)
        x = self.in_proj(x)

        dev = x.device
        n_sent = chars.shape[0]
        counts = torch.bincount(node_sent, minlength=n_sent)
        starts = torch.zeros_like(counts)
        starts[1:] = counts.cumsum(0)[:-1]
        slot = torch.arange(node_sent.numel(), device=dev) - starts[node_sent]

        # Which group each sentence landed in, so a group's nodes are one mask.
        groups = span_groups(counts.tolist(), self.pair_budget)
        of_sent = torch.empty(n_sent, dtype=torch.int64, device=dev)
        for gi, g in enumerate(groups):
            of_sent[torch.as_tensor(g, dtype=torch.int64, device=dev)] = gi
        node_group = of_sent[node_sent]

        out = x.new_zeros(x.shape[0], self.d_model)
        for gi, g in enumerate(groups):
            gt = torch.as_tensor(g, dtype=torch.int64, device=dev)
            idx = (node_group == gi).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            within = torch.empty(n_sent, dtype=torch.int64, device=dev)
            within[gt] = torch.arange(len(g), device=dev)
            row, col = within[node_sent[idx]], slot[idx]
            G, N = len(g), int(counts[gt].max())

            X = x.new_zeros(G, N, self.d_model)
            X[row, col] = x[idx]
            S = torch.zeros(G, N, dtype=torch.int64, device=dev)
            E = torch.zeros(G, N, dtype=torch.int64, device=dev)
            ok = torch.zeros(G, N, dtype=torch.bool, device=dev)
            S[row, col] = span_start[idx]
            E[row, col] = span_end[idx] - 1        # inclusive tail
            ok[row, col] = char_ok[idx]

            mask = self._bias(S, E, ok)
            for blk in self.blocks:
                X = blk(X, mask)
            out[idx] = X[row, col]

        ctx = self.out_proj(self.norm_out(out))
        # Boundary nodes carry no signal, matching every other featurizer.
        return ctx.masked_fill(~char_ok.unsqueeze(-1), 0.0)


@register("lattice_attn_char")
class LatticeAttentionCharEncoder(LatticeAttentionEncoder):
    """lattice_attn plus the char-BiLSTM span vector as attention input.

    The closest analogue to TransLIST, which reads characters and candidate words
    together. Keeps the +0.0695 F1 the char encoder already proved rather than
    asking attention over candidates to rediscover the raw sandhied surface.
    """

    use_chars = True
