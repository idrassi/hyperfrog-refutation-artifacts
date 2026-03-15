# Artifacts for HyperFrog's Formal Miner Is Almost Uniform

Artifacts accompanying the paper `HyperFrog's Formal Miner Is Almost Uniform`.

Paper details:

- IDRASSI, M. (2026). *HyperFrog's Formal Miner Is Almost Uniform*. Zenodo. https://doi.org/10.5281/zenodo.18976980
- PDF in this repository: `HyperFrog_refutation.pdf`

Repository URL cited by the paper:

- `https://github.com/idrassi/hyperfrog-refutation-artifacts`

## Contents

- `HyperFrog_refutation.pdf`: current paper PDF
- `hyperfrog_refutation_experiments.py`: standalone reproduction script
- `bulk_metrics_100k.csv`: bulk statistics on 100,000 uniform samples
- `topology_metrics_10k.csv`: component and cycle-rank diagnostics on 10,000 samples
- `practical_largest_component_metrics_10k.csv`: keep-largest-component diagnostics on 10,000 samples
- `summary.json`: main theorem/experiment summary used in the paper
- `practical_summary.json`: summary for the keep-largest-component experiment
- `figure_predicate_diagnostics.png`: figure reproduced in the paper

## Environment

The script expects Python 3 with the packages listed in `requirements.txt`.

Install dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

## Reproducing the artifacts

Run:

```bash
python3 hyperfrog_refutation_experiments.py
```

This regenerates:

- `bulk_metrics_100k.csv`
- `topology_metrics_10k.csv`
- `practical_largest_component_metrics_10k.csv`
- `summary.json`
- `practical_summary.json`
- `figure_predicate_diagnostics.png`

## Notes

- The reproduction targets the paper-mode specification analyzed in the paper.
- The experiments use fixed seeds embedded in the script for reproducibility.
- The figure file `figure_predicate_diagnostics.png` is the one referenced by the LaTeX paper source.
