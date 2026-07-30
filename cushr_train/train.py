#!/usr/bin/env python3

# loop that actually learns the weights
# MAIN stuff

import argparse
import json
import time

import numpy as np
import torch

import learned_featurizers as LF
from dataset import LatticeStore, collate
from model import BiaffineEdgeScorer
from viterbi import viterbi, path_score, gold_score, predicted_nodes


def to_torch(batch, dev):
    t = {}
    for k in ("src", "dst", "level_ptr", "source", "sink",
              "gold_edge", "gold_edge_ptr", "node_sent"):
        t[k] = torch.as_tensor(np.asarray(batch[k]), dtype=torch.int64, device=dev)
    t["feats"] = torch.as_tensor(batch["feats"], dtype=torch.float32, device=dev)
    if batch.get("ids") is not None:
        t["ids"] = torch.as_tensor(batch["ids"], dtype=torch.int64, device=dev)
    t["gold_node"] = torch.as_tensor(np.asarray(batch["gold_node"]),
                                     dtype=torch.bool, device=dev)
    t["num_nodes"] = batch["num_nodes"]
    t["num_edges"] = batch["num_edges"]
    return t


def batches(ids, size, shuffle, rng=None):
    ids = np.asarray(ids)
    if shuffle:
        assert rng is not None, "shuffle=True requires an rng"
        ids = ids[rng.permutation(len(ids))]
    for i in range(0, len(ids), size):
        yield ids[i:i + size]


def save_model(path, model, feat_dim, hidden, featurizer_name):
    """Write the model archive. Called on every dev improvement, not just at
    the end, so an interrupted run still leaves its best weights on disk."""
    p = model.to_numpy()
    np.savez(path, feat_dim=np.int32(feat_dim), hidden=np.int32(hidden),
             src_proj=p["src_proj"], dst_proj=p["dst_proj"], bias=p["bias"],
             featurizer_name=np.array(featurizer_name))


def f1_counts(pred_sets, gold_sets):
    tp = fp = fn = pm = 0
    for p, g in zip(pred_sets, gold_sets):
        ps, gs = set(p), set(g)
        tp += len(ps & gs)
        fp += len(ps - gs)
        fn += len(gs - ps)
        pm += int(ps == gs)
    return tp, fp, fn, pm


@torch.no_grad()
def evaluate(model, store, ids, dev, batch_size=128):
    model.eval()
    tp = fp = fn = pm = n = 0
    for chunk in batches(ids, batch_size, shuffle=False):
        b = collate(store, chunk)
        t = to_torch(b, dev)
        w = model(t["feats"], t["src"], t["dst"], t.get("ids"))
        pe, pmask, _ = viterbi(t, w)
        pred_local = predicted_nodes(t, pe, pmask)
        gn = np.asarray(b["global_node"])
        pred = [[int(gn[x]) for x in p] for p in pred_local]
        gold = [store.gold_nodes[store.gold_off[s]:store.gold_off[s + 1]].tolist()
                for s in chunk]
        a, c, d, e = f1_counts(pred, gold)
        tp += a; fp += c; fn += d; pm += e; n += len(chunk)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    model.train()
    return {"precision": prec, "recall": rec, "f1": f1,
            "perfect_match": pm / n if n else 0.0, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./cache")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-train", type=int, default=0,
                    help="cap training sentences (smoke runs)")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--out", default="model_biaffine.npz")
    ap.add_argument("--log", default="train_log.json")
    # --- learned featurizer options (used only if the cache carries node_ids)
    ap.add_argument("--learned", default="hybrid",
                    help="learned featurizer name (learned_featurizers.py), "
                         "or 'none' to score the precomputed columns directly "
                         "even when the cache carries id columns")
    ap.add_argument("--node-dim", type=int, default=96,
                    help="width of the learned node vector")
    ap.add_argument("--word-dropout", type=float, default=0.1,
                    help="probability of replacing a form/lemma id with <UNK> "
                         "during training; keeps the non-identity pathway "
                         "predictive so out-of-vocabulary input degrades "
                         "gracefully")
    ap.add_argument("--materialize", default="",
                    help="after training, write the frozen node vectors here "
                         "as a dense featurized .npz that the ordinary "
                         "(C++/GPU) pipeline can consume")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)

    store = LatticeStore(args.cache)
    train_ids = store.trainable("train")
    dev_ids = store.trainable("dev")
    test_ids = store.trainable("test")
    if args.limit_train:
        train_ids = train_ids[:args.limit_train]

    # feat_dim comes from the data, not a constant: the featurizer that built
    # this cache decided it.
    feat_dim = store.feat_dim
    print(f"device={dev}  train={len(train_ids)}  dev={len(dev_ids)}  "
          f"test={len(test_ids)}  feat_dim={feat_dim}  "
          f"featurizer={store.featurizer_name}")

    # A cache built with --emit-ids carries identity columns, which means the
    # node vector is produced by a learned featurizer rather than read straight
    # from the archive. Everything downstream of scoring is identical either
    # way; see learned_featurizers.materialize for how this collapses back to a
    # plain dense archive once training finishes.
    featurizer = None
    if store.node_ids is not None and args.learned != "none":
        n_forms, n_lemmas, n_preverbs = store.id_vocab_sizes
        featurizer = LF.get(args.learned, n_scalars=feat_dim,
                            n_forms=n_forms, n_lemmas=n_lemmas,
                            n_preverbs=n_preverbs,
                            out_dim=args.node_dim,
                            word_dropout=args.word_dropout).to(dev)
        scorer = BiaffineEdgeScorer(args.node_dim, args.hidden).to(dev)
        model = LF.LearnedBiaffine(featurizer, scorer).to(dev)
        print(f"learned featurizer {args.learned!r}: "
              f"forms={n_forms:,} lemmas={n_lemmas:,} preverbs={n_preverbs:,} "
              f"node_dim={args.node_dim} word_dropout={args.word_dropout}")
        print(f"  embedding params: "
              f"{sum(p.numel() for p in featurizer.embedding_parameters()):,}")
        # Embedding gradients are sparse -- a batch touches only the rows it
        # gathered -- so materialising a dense gradient over the whole table
        # every step would dominate the step time. SparseAdam handles the
        # tables; AdamW handles everything else.
        opt = torch.optim.AdamW(
            list(featurizer.dense_parameters()) + list(scorer.parameters()),
            lr=args.lr, weight_decay=args.weight_decay)
        emb_opt = torch.optim.SparseAdam(featurizer.embedding_parameters(),
                                         lr=args.lr)
    else:
        model = BiaffineEdgeScorer(feat_dim, args.hidden).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
        emb_opt = None
    print(f"parameters: {model.num_params()}")

    # What gets exported as the .bin is always the biaffine scorer, operating on
    # whatever width the node vector ends up being: feat_dim for a precomputed
    # featurizer, node_dim once a learned one has been materialised.
    scorer_module = model.scorer if featurizer is not None else model
    export_dim = args.node_dim if featurizer is not None else feat_dim
    # Record the composed name, so a model file says which learned featurizer
    # produced the vectors it expects rather than only naming the scalar block.
    export_name = (f"{store.featurizer_name}+{args.learned}"
                   if featurizer is not None else store.featurizer_name)

    history = []
    best_f1, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tot_loss, tot_active, nb, nsent = 0.0, 0, 0, 0
        for chunk in batches(train_ids, args.batch, True, rng):
            # step 1: collate. pack 64 sent into 1 batch
            b = collate(store, chunk)

            #step 2: one score per edge
            t = to_torch(b, dev)
            w = model(t["feats"], t["src"], t["dst"], t.get("ids"))

            # step 3: cost augmentation, reward landing on a non gold node
            # we want model to be confident that gol = best
            # add margin of 1 to any wrong edge
            # this way if inflated path beats gold, it is too close and weights need to be fixed
            # ensure that gold beats by certain margin
            cost = args.margin * (~t["gold_node"][t["dst"]]).to(w.dtype)

            # run viterbi to get best path with cost augmented scores
            # you either get:
            # 1) best wrong path - need to optimize
            # 2) gold path - good
            pe, pmask, _ = viterbi(t, (w + cost).detach())

            s_pred = path_score(w, pe, pmask)

            # hamming is the number of nodes on predicted path not on gold path * 1.0 margin
            hamming = (cost[pe] * pmask.to(w.dtype)).sum(-1)
            s_gold = gold_score(w, t["gold_edge"], t["gold_edge_ptr"], len(chunk))

            # compute loss using rectified linear unit which makes negative value = 0
            # if the next best path you got is a lot smaller than gold score, RELU = 0
            # if you get gold path meaning score, margin is 0 so RELU = 0
            loss_vec = torch.relu(hamming + s_pred - s_gold)
            loss = loss_vec.mean()

            opt.zero_grad(set_to_none=True)
            if emb_opt is not None:
                emb_opt.zero_grad(set_to_none=True)

            # step 6: backpropagate to trace the loss thru w src, dest, bias -> compute a gradient
            loss.backward()
            # Only the dense parameters are clipped: clip_grad_norm_ needs a
            # dense .grad and would fail on the sparse embedding gradients.
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters()
                 if p.grad is not None and not p.grad.is_sparse], 5.0)

            # step 7: update the weights with a little nudge
            opt.step()
            if emb_opt is not None:
                emb_opt.step()

            tot_loss += float(loss.detach()) * len(chunk)
            tot_active += int((loss_vec > 0).sum())
            nsent += len(chunk)
            nb += 1

        rec = {"epoch": ep,
               "train_loss": tot_loss / max(nsent, 1),
               "active_frac": tot_active / max(nsent, 1),
               "sec": time.time() - t0}
        if ep % args.eval_every == 0 or ep == args.epochs:
            # evaluate after each epoch
            # decode to get best path, get F1 (best compared to gold), select the model of all the 10 epochs that gives best result
            # evaluate on test
            rec["dev"] = evaluate(model, store, dev_ids, dev)
            if rec["dev"]["f1"] > best_f1:
                best_f1 = rec["dev"]["f1"]
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
                # Checkpoint on every improvement rather than only at the end.
                # A long CPU run can be killed by an external time limit, and
                # without this the entire run's work is lost; with it the best
                # model so far is already on disk and usable.
                save_model(args.out, scorer_module, export_dim, args.hidden,
                           export_name)
        history.append(rec)
        # Same reasoning for the log: keep the epoch history durable as it goes.
        with open(args.log, "w") as f:
            json.dump({"args": vars(args), "featurizer": store.featurizer_name,
                       "feat_dim": int(feat_dim), "history": history}, f, indent=2)
        d = rec.get("dev")
        print(f"epoch {ep:>2}  loss {rec['train_loss']:.4f}  "
              f"active {rec['active_frac']:.3f}  {rec['sec']:.1f}s"
              + (f"  dev_F1 {d['f1']:.4f}  dev_PM {d['perfect_match']:.4f}"
                 if d else ""))

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, store, test_ids, dev)
    print(f"\ntest: F1 {test['f1']:.4f}  P {test['precision']:.4f}  "
          f"R {test['recall']:.4f}  PM {test['perfect_match']:.4f}  n={test['n']}")

    save_model(args.out, scorer_module, export_dim, args.hidden, export_name)
    print(f"wrote {args.out}")
    with open(args.log, "w") as f:
        json.dump({"args": vars(args), "featurizer": store.featurizer_name,
                   "feat_dim": int(feat_dim),
                   "history": history, "test": test}, f, indent=2)
    print(f"wrote {args.log}")

    if featurizer is not None and args.materialize:
        write_materialized(args, store, featurizer, dev, export_dim)


def write_materialized(args, store, featurizer, dev, export_dim):
    """Freeze a learned featurizer into an ordinary dense featurized .npz.

    This is what keeps embedding tables out of the C++ and CUDA paths. The
    tables no longer change after training, so every node has one fixed vector;
    writing those out produces exactly the artifact the precomputed featurizers
    produce, and the rest of the pipeline (prepare.py, export_weights.py,
    BiaffineScorer, the GPU drivers) consumes it with no changes at all.
    """
    import os

    n = store.node_features.shape[0]
    scratch = args.materialize + ".tmp.npy"
    print(f"materialising {n:,} node vectors -> ({n:,}, {export_dim}) ...")
    feats = np.lib.format.open_memmap(scratch, mode="w+", dtype=np.float32,
                                      shape=(n, export_dim))
    LF.materialize(featurizer, store.node_features, store.node_ids, dev,
                   out=feats)
    feats.flush()

    cache = args.cache
    load = lambda name: np.load(os.path.join(cache, name + ".npy"), mmap_mode="r")
    save = dict(
        node_features=feats,
        node_word_length=load("node_word_length"),
        row_ptr=load("row_ptr"), col_idx=load("col_idx"),
        topo_level=load("topo_level"),
        sentence_offsets=load("sentence_offsets"),
        gold_path_mask=load("gold_path_mask"),
        gold_path_nodes=load("gold_path_nodes"),
        gold_path_offsets=load("gold_path_offsets"),
        feat_dim=np.int32(export_dim),
        featurizer_name=np.array(f"{store.featurizer_name}+{args.learned}"),
    )
    np.savez(args.materialize, **save)
    del feats, save
    os.remove(scratch)
    print(f"wrote {args.materialize}  "
          f"({os.path.getsize(args.materialize) / 1e6:.0f} MB)")
    # The embedding tables are not needed to decode, but keep them so a run is
    # reproducible and so the learned vectors can be inspected later.
    torch.save(featurizer.state_dict(), args.materialize + ".featurizer.pt")


if __name__ == "__main__":
    main()
