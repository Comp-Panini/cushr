# Learned Scoring Function

A trained `BiaffineEdgeScorer` that produces edge scores.


## Pipeline

```
python prepare.py --npz ../data/new_cushr_data_fixed_USE_THIS.npz
python smoke_test.py                    # validates collate + viterbi
python baseline.py                      
python train.py --epochs 10 --batch 64
python export_weights.py --bin model_biaffine.bin --edge-scores edge_score.npy
python check_export.py

# use it in the CPU decoder:
../cushr_cpu/cushr_evaluate ../data/new_cushr_data_fixed_USE_THIS.npz \
    --scorer biaffine --model model_biaffine.bin --K 10
```

## The model

```
score(e = (u, v)) = <W_s x(u), W_d x(v)> + b
x(v) = [ node_features[v] (43 morph one-hots) , log1p(word_length[v]) ]
```

## Results

Top-1, word-level:

| scorer | F1 | P | R | 
|---|---|---|---|
| uniform | 0.4857 | 0.4217 | 0.5726 | 
| length | 0.4848 | 0.4209 | 0.5716 | 
| log_linear | 0.5666 | 0.5276 | 0.6118 | 
| biaffine | 0.7904 | 0.8175 | 0.7650 | 

![Segmentation accuracy by scorer](scorer_f1.png)


### Training curve

![Training loss vs epoch](training_loss.png)

### Max margin scores

![Hinge loss distribution](hinge_histogram.png)