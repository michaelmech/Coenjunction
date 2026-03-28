# Coenjunction

Mutual information and transfer entropy estimation using copula entropy.

## Academic foundation

This repository's mutual-information estimator was inspired by the paper bundled in this repo:

- Yves-Laurent Kom Samo, *Inductive Mutual Information Estimation: A Convex Maximum-Entropy Copula Approach* (MIND), arXiv:2102.13182v3.
- Local copy: `2102.13182v3.pdf`

If you use this implementation in academic work, please cite that paper.

```bibtex
@article{komsamo2021mind,
  title   = {Inductive Mutual Information Estimation: A Convex Maximum-Entropy Copula Approach},
  author  = {Kom Samo, Yves-Laurent},
  journal = {arXiv preprint arXiv:2102.13182},
  year    = {2021},
  url     = {https://arxiv.org/abs/2102.13182}
}
```

## Project layout

- `coenjunction/`: reusable Python modules.
- `notebooks/`: exploratory notebooks (`*.ipynb`).

## Main APIs

- `coenjunction.CopulaEntropyEstimator`
- `coenjunction.estimate_mi_from_ce`
- `coenjunction.calculate_transfer_entropy`
- `coenjunction.calculate_transfer_entropy_with_edge_lag`
