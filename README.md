# Coverage, not granularity: measurements of mini-batch GNN training

Code and raw measurements for the study of subgraph-wise mini-batch training of graph neural networks (Cluster-GCN sampling) on ogbn-arxiv, Reddit, ogbn-products and Yelp. The central experiment holds the fraction of the graph covered by each training step constant while the number of METIS partitions varies, and shows that the accuracy loss usually attributed to finer partitioning follows coverage instead.

All numbers reported in the paper are computed from the files in `results/` by the scripts in `scripts/`.

## Layout

- `notebooks/` – Jupyter notebooks that produced the measurements, in the order they were run: `osnovni_a100` (neighborhood growth, full-graph against sampled training), `dopuna_a100_sve` and `dopuna_a100_dodatni_seedovi` (sampling depth and width, Cluster-GCN sweep, overlap factor), `clustergcn_pokrivenost` (constant coverage on ogbn-arxiv), `prosirenje_a100_*` (four graphs, numerical precision, input pipeline), `clustergcn_int16` (16-bit edge indices on ogbn-arxiv), `pokrivenost_gustina` (constant coverage on Reddit and ogbn-products), `int16_products`, `rns_arxiv` (random node sampling).
- `results/` – per-run CSV files, one directory per seed, plus aggregated tables under `agregat/`.
- `scripts/` – aggregation and significance tests (`agregacija_*.py`, `test_znacajnosti*.py`), the memory prediction analysis (`analiza_lanac_memorije.py`), the hardware calibration (`kalibracija_a100.py`) and two small checks of index types and batch assembly.

## Environment

Measurements were taken on one NVIDIA A100 (40 GB) with 16 allocated processor cores, PyTorch 2.6.0, CUDA 12.4 and PyTorch Geometric. The datasets are downloaded by the notebooks through the `ogb` and `torch_geometric` loaders into a `podaci/` directory next to the notebook.

```
pip install -r requirements.txt
```

## Reproducing

1. Run a notebook on a machine with a CUDA device. Each notebook writes one CSV per seed into its results directory and skips rows that already exist, so an interrupted session can be resumed. Seeds are fixed in the first cell.
2. Aggregate: `python3 scripts/agregacija_pokrivenost_gustina.py`, `python3 scripts/agregacija_int16.py`, `python3 scripts/agregacija_rns.py` and so on, each writing an `agregat/` directory.
Scripts in `scripts/` expect to sit next to the `results_*` and `rezultati_*` directories, as in the original project; copy or symlink `results/*` one level up, or adjust the `HERE` path at the top of each script.

## Statistics

Every configuration whose accuracy enters a comparison was trained ten times with seeds 42, 123, 7, 1, 2, 3, 4, 5, 6 and 8. Differences are tested with a paired exact permutation test (sign-flip test) over the seeds. With n paired observations the smallest two-sided p-value such a test can produce is 2/2^n, so three repetitions cannot bring any difference below the 0.05 level.

## Citation

Paper under review at the XVIII International SAUM Conference, Niš, 2026. Citation details will be added here on acceptance.
