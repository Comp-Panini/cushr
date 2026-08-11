#!/usr/bin/env python3
"""Path-level reranker: a BiLSTM over the whole candidate analysis.

WHY A SEQUENCE MODEL AND NOT THE COUNT FEATURES
------------------------------------------------
The obvious reranker scores role counts (n nominatives, n finite verbs, ...)
plus the base Viterbi score. That was measured before it was built, and it is
capped: of the 21.40 points of S+M headroom inside the K=16 beam, only 20.1%
is separable by role counts -- elsewhere a correct candidate carries exactly
the same count vector as an incorrect one in the same beam, and no weighting of
tied features can break a tie. Ceiling 56.57% S+M against 52.26 today.

So counts are skipped. This model reads the candidate as a SEQUENCE -- word by
word, in order, with each word's morphology -- so `nom acc verb` and
`acc nom verb` are different inputs where the counts are identical. That is the
distinction the count features provably cannot make.

WHAT IT CAN AND CANNOT REACH
-----------------------------
recall@16 = 73.67 S+M and recall@64 = 76.52 against an ORACLE of 77.65. The
beam nearly always contains the right answer, so this is a ranking problem, not
a coverage problem. No reranker can exceed recall@K, and that is the honest
bound on everything below.

THE ARCHITECTURE, AND THE ONE NON-OBVIOUS PIECE
------------------------------------------------
Per candidate path: embed each word's morph tag and lemma, run a 1-layer
BiLSTM, mean-pool, project to a scalar. The final score is

    score = w_base * base_viterbi_score + bilstm_scalar + b

with `w_base` initialised to 1 and the BiLSTM's output projection zeroed, so
**at step 0 the reranker is exactly the current decoder**. Every point it moves
is measured against a real baseline rather than an untrained scramble, and the
run asserts this identity at epoch 0 rather than trusting it.

Exactly one factor may be zero-initialised; see PathRanker.__init__ for the
deadlock that results from zeroing two.

A collapse to "just trust the base score" would look like success on the loss
curve while changing nothing, so `|ctx|` -- the mean absolute contribution of
the BiLSTM branch -- is reported every epoch. Zero there means the sequence
model is inert whatever the loss says.

Training is listwise: softmax over the K candidates of one sentence, loss
-log(sum of probability on the correct ones). Several candidates can be correct
(they are distinct node sequences that agree on form/lemma/cng), hence a
partial-label sum rather than a single-label cross-entropy. Sentences with no
correct candidate in the beam carry no signal and are dropped -- their count is
reported, since it is the recall@K miss rate seen from the inside.

TRAINING DATA. The train split's own candidate lists, which
`make_rerank_data.py` verified are representative: train recall@1 exceeds dev
by 0.67 points, so the base scorer has not memorised them and no jackknifed
decoding is needed. (Measured with random sampling; truncating each split to
its first N sentences instead shows a spurious +4.02, because sentence ids run
in corpus order.)
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn


class PathRanker(nn.Module):
    def __init__(self, n_morph, n_lemma, emb=32, hidden=64, lemma_emb=32,
                 dropout=0.3, use_lemma=True):
        super().__init__()
        self.m_emb = nn.Embedding(n_morph, emb)
        # Optional. The hypothesis is about the MORPHOLOGY sequence, and the
        # morph table is 848 entries against a lemma vocabulary two orders
        # bigger -- so the lemma branch is most of the parameters and most of
        # the overfitting risk while being the part the hypothesis does not
        # need. --no-lemma runs the clean version; dev decides.
        self.l_emb = nn.Embedding(n_lemma, lemma_emb) if use_lemma else None
        # The lemma vocabulary is large (~10^5) against ~10^5 training
        # sentences, so this is where the model will memorise if anywhere.
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(emb + (lemma_emb if use_lemma else 0), hidden,
                            batch_first=True,
                            bidirectional=True)
        self.out = nn.Linear(2 * hidden, 1)
        # Start as an exact copy of the base decoder: out(pooled) == 0, so
        # score == w_base * base_score.
        #
        # Zero-init EXACTLY ONE factor. An earlier version also had a scalar
        # w_ctx initialised to 0 multiplying this layer, which deadlocked: the
        # gradient w.r.t. w_ctx is proportional to out(pooled) = 0 and the
        # gradient w.r.t. out.weight is proportional to w_ctx = 0, so both
        # stayed at 0 for ever and the reranker silently remained the base
        # decoder. With the scalar gone, d(loss)/d(out.weight) is proportional
        # to `pooled`, which is non-zero, and training moves from step 1.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.w_base = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, morph, lemma, lengths, base, return_ctx=False):
        """morph/lemma [N, T] padded; lengths [N]; base [N] -> scores [N]."""
        x = self.m_emb(morph)
        if self.l_emb is not None:
            x = torch.cat([x, self.l_emb(lemma)], -1)
        x = self.drop(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        mask = (torch.arange(h.shape[1], device=h.device)[None, :]
                < lengths[:, None]).unsqueeze(-1).float()
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        ctx = self.out(self.drop(pooled)).squeeze(-1)
        s = self.w_base * base + ctx + self.bias
        return (s, ctx) if return_ctx else s


class Data:
    """Candidate lists grouped by sentence, with per-word ids attached."""

    def __init__(self, npz_path, morph_ids, lemma_ids, max_len=32):
        d = np.load(npz_path)
        self.off = d["cand_off"]
        self.nodes = d["cand_nodes"]
        self.sent = d["cand_sent"]
        self.score = d["cand_score"]
        self.label = d["cand_label"]
        self.n_sent = int(self.sent.max()) + 1 if len(self.sent) else 0
        self.morph_ids, self.lemma_ids, self.max_len = morph_ids, lemma_ids, max_len
        # candidate index ranges per sentence; cand_sent is non-decreasing
        bounds = np.searchsorted(self.sent, np.arange(self.n_sent + 1))
        self.groups = [(int(bounds[i]), int(bounds[i + 1]))
                       for i in range(self.n_sent)]
        has = np.zeros(self.n_sent, dtype=bool)
        np.logical_or.at(has, self.sent, self.label)
        self.usable = np.nonzero(has)[0]

    def batch(self, sent_idx, dev):
        lo = [self.groups[s][0] for s in sent_idx]
        hi = [self.groups[s][1] for s in sent_idx]
        cands = [c for a, b in zip(lo, hi) for c in range(a, b)]
        seqs = [self.nodes[self.off[c]:self.off[c + 1]][:self.max_len]
                for c in cands]
        T = max(1, max(len(s) for s in seqs))
        morph = np.zeros((len(seqs), T), dtype=np.int64)
        lemma = np.zeros((len(seqs), T), dtype=np.int64)
        lens = np.ones(len(seqs), dtype=np.int64)
        for i, s in enumerate(seqs):
            if len(s):
                morph[i, :len(s)] = self.morph_ids[s]
                lemma[i, :len(s)] = self.lemma_ids[s]
                lens[i] = len(s)
        gsz = [b - a for a, b in zip(lo, hi)]
        return (torch.as_tensor(morph, device=dev),
                torch.as_tensor(lemma, device=dev),
                torch.as_tensor(lens),
                torch.as_tensor(self.score[cands], device=dev),
                torch.as_tensor(self.label[cands], device=dev),
                gsz)


def listwise_loss(scores, labels, gsz):
    """-log sum p(correct), summed over sentences that have a correct candidate."""
    loss, i, n = scores.new_zeros(()), 0, 0
    for g in gsz:
        s, y = scores[i:i + g], labels[i:i + g]
        i += g
        if not bool(y.any()):
            continue
        lp = torch.log_softmax(s, 0)
        loss = loss - torch.logsumexp(lp[y], 0)
        n += 1
    return loss / max(1, n), n


@torch.no_grad()
def evaluate(model, data, dev, batch=64):
    """Top-1 accuracy after reranking, plus the base and beam references."""
    model.eval()
    hit = base_hit = ceil_hit = total = 0
    ctx_mag = n_cand = 0.0
    for lo in range(0, data.n_sent, batch):
        idx = list(range(lo, min(lo + batch, data.n_sent)))
        m, l, ln, bs, y, gsz = data.batch(idx, dev)
        sc, cx = model(m, l, ln, bs, return_ctx=True)
        ctx_mag += float(cx.abs().sum())
        n_cand += len(cx)
        i = 0
        for g in gsz:
            s, yy, b = sc[i:i + g], y[i:i + g], bs[i:i + g]
            i += g
            total += 1
            hit += bool(yy[int(s.argmax())])
            base_hit += bool(yy[int(b.argmax())])
            ceil_hit += bool(yy.any())
    model.train()
    # mean |context contribution|: the direct check that the BiLSTM branch is
    # alive. If this is 0 the reranker has silently collapsed to the base
    # decoder, which is what the dead zero-init used to do.
    return (100 * hit / total, 100 * base_hit / total, 100 * ceil_hit / total,
            ctx_mag / max(1, n_cand))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="rerank_k16_train.npz")
    ap.add_argument("--dev", default="rerank_k16_dev.npz")
    ap.add_argument("--test", default="rerank_k16_test.npz")
    ap.add_argument("--raw", default="../data/cushr_data_g95.npz")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--no-lemma", action="store_true",
                    help="morphology sequence only -- drops ~95%% of the "
                         "parameters and tests the hypothesis in isolation")
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=2,
                    help="stop after this many epochs with no dev improvement")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reranker.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("cpu")
    z = np.load(args.raw)
    morph_ids = np.asarray(z["node_features"]).ravel().astype(np.int64)
    lemma_ids = np.asarray(z["node_lemma_id"]).ravel().astype(np.int64)

    tr = Data(args.train, morph_ids, lemma_ids)
    dv = Data(args.dev, morph_ids, lemma_ids)
    print(f"train {tr.n_sent:,} sentences ({len(tr.usable):,} with a correct "
          f"candidate = {100*len(tr.usable)/tr.n_sent:.2f}%)")
    print(f"dev   {dv.n_sent:,} sentences ({len(dv.usable):,} usable)")
    print(f"  {tr.n_sent - len(tr.usable):,} train sentences dropped: the beam "
          f"holds no correct\n  candidate, so they carry no ranking signal.\n")

    model = PathRanker(int(morph_ids.max()) + 1, int(lemma_ids.max()) + 1,
                       hidden=args.hidden, dropout=args.dropout,
                       use_lemma=not args.no_lemma).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    print(f"  {sum(p.numel() for p in model.parameters()):,} reranker parameters"
          f"  (dropout {args.dropout}, weight decay {args.weight_decay})\n")

    a, b, c, cm = evaluate(model, dv, dev)
    print(f"epoch 0 (untrained)  dev top-1 {a:6.2f}   base {b:6.2f}   "
          f"beam ceiling {c:6.2f}   |ctx| {cm:.4f}")
    assert abs(a - b) < 1e-6, (
        "an untrained reranker must reproduce the base decoder exactly -- "
        "out(pooled) starts at 0, so any difference means the score path is "
        "wrong (argmax over base score must equal argmax over total score)")

    rng = np.random.default_rng(args.seed)
    best, stale = -1.0, 0
    for ep in range(1, args.epochs + 1):
        order = rng.permutation(tr.usable)
        tot, nb, t0 = 0.0, 0, time.time()
        for lo in range(0, len(order), args.batch):
            idx = order[lo:lo + args.batch].tolist()
            m, l, ln, bs, y, gsz = tr.batch(idx, dev)
            loss, n = listwise_loss(model(m, l, ln, bs), y, gsz)
            if not n:
                continue
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
            if nb % 500 == 0:
                print(f"    ep{ep} {lo:,}/{len(order):,} loss {tot/nb:.4f}",
                      flush=True)
        a, b, c, cm = evaluate(model, dv, dev)
        print(f"epoch {ep}  loss {tot/max(1,nb):.4f}  dev top-1 {a:6.2f}  "
              f"(base {b:6.2f}, ceiling {c:6.2f})  "
              f"w_base {float(model.w_base.detach()):.3f}  |ctx| {cm:.4f}  "
              f"{time.time()-t0:.0f}s")
        if a > best:
            best = a
            torch.save({"state": model.state_dict(),
                        "n_morph": int(morph_ids.max()) + 1,
                        "n_lemma": int(lemma_ids.max()) + 1,
                        "hidden": args.hidden,
                        "use_lemma": not args.no_lemma}, args.out)
            print(f"    saved {args.out}")
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"    no dev gain for {stale} epochs -- stopping early. "
                      f"Train loss can keep falling while dev decays;\n"
                      f"    the saved checkpoint is the best dev epoch, not "
                      f"the last.")
                break

    print(f"\nbest dev top-1 (gold-path match): {best:.2f}")
    if args.test:
        te = Data(args.test, morph_ids, lemma_ids)
        model.load_state_dict(torch.load(args.out)["state"])
        a, b, c, cm = evaluate(model, te, dev)
        print(f"test  top-1 {a:6.2f}   base {b:6.2f}   beam ceiling {c:6.2f}"
              f"   ({a-b:+.2f})")
        print("\n  These are GOLD-PATH match, the reranker's own objective.")
        print("  The reported metric is S+M against the external reference --")
        print("  run eval_slm.py --rerank to get it; the two differ by the")
        print("  SHR/DCS convention gap, which the maps handle separately.")


if __name__ == "__main__":
    main()
