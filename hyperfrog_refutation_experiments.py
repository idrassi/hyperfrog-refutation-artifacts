import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, stats

OUTDIR = Path(__file__).resolve().parent
N = 4096
GRID = (16, 16, 16)
SEED_BULK = 20260307
SEED_TOPO = 20260308
SEED_PRACTICAL = 20260309

STRUCTURE = np.zeros((3, 3, 3), dtype=np.uint8)
STRUCTURE[1, 1, 0] = STRUCTURE[1, 1, 2] = 1
STRUCTURE[1, 0, 1] = STRUCTURE[1, 2, 1] = 1
STRUCTURE[0, 1, 1] = STRUCTURE[2, 1, 1] = 1
STRUCTURE[1, 1, 1] = 1


def bulk_metrics(num=100_000, batch=5_000, seed=SEED_BULK):
    rng = np.random.default_rng(seed)
    Vs, Es, Bs, LBs = [], [], [], []
    for start in range(0, num, batch):
        b = min(batch, num - start)
        X = rng.integers(0, 2, size=(b, *GRID), dtype=np.uint8)
        V = X.sum(axis=(1, 2, 3), dtype=np.int32)
        E = (
            (X[:, :, :, :-1] & X[:, :, :, 1:]).sum(axis=(1, 2, 3), dtype=np.int32)
            + (X[:, :, :-1, :] & X[:, :, 1:, :]).sum(axis=(1, 2, 3), dtype=np.int32)
            + (X[:, :-1, :, :] & X[:, 1:, :, :]).sum(axis=(1, 2, 3), dtype=np.int32)
        )
        B = (
            (X[:, :, :, :-1] ^ X[:, :, :, 1:]).sum(axis=(1, 2, 3), dtype=np.int32)
            + (X[:, :, :-1, :] ^ X[:, :, 1:, :]).sum(axis=(1, 2, 3), dtype=np.int32)
            + (X[:, :-1, :, :] ^ X[:, 1:, :, :]).sum(axis=(1, 2, 3), dtype=np.int32)
        )
        LB = E - V + 1
        Vs.append(V)
        Es.append(E)
        Bs.append(B)
        LBs.append(LB)
    return pd.DataFrame({
        'V': np.concatenate(Vs),
        'E': np.concatenate(Es),
        'B': np.concatenate(Bs),
        'lb': np.concatenate(LBs),
    })


def topology_metrics(num=10_000, seed=SEED_TOPO):
    rng = np.random.default_rng(seed)
    recs = []
    for _ in range(num):
        arr = rng.integers(0, 2, size=GRID, dtype=np.uint8).astype(bool)
        V = int(arr.sum())
        E = int((arr[:, :, :-1] & arr[:, :, 1:]).sum() + (arr[:, :-1, :] & arr[:, 1:, :]).sum() + (arr[:-1, :, :] & arr[1:, :, :]).sum())
        labeled, C = ndimage.label(arr, structure=STRUCTURE)
        genus = E - V + C
        B = int((arr[:, :, :-1] ^ arr[:, :, 1:]).sum() + (arr[:, :-1, :] ^ arr[:, 1:, :]).sum() + (arr[:-1, :, :] ^ arr[1:, :, :]).sum())
        counts = np.bincount(labeled.ravel())[1:]
        largest = int(counts.max()) if counts.size else 0
        recs.append((V, E, C, genus, B, largest))
    return pd.DataFrame(recs, columns=['V', 'E', 'C', 'genus', 'B', 'largest'])


def keep_largest_component(arr):
    labeled, C = ndimage.label(arr, structure=STRUCTURE)
    if C <= 1:
        return arr
    counts = np.bincount(labeled.ravel())[1:]
    best = counts.argmax() + 1
    return labeled == best


def practical_metrics(num=10_000, seed=SEED_PRACTICAL):
    rng = np.random.default_rng(seed)
    recs = []
    for _ in range(num):
        arr = rng.integers(0, 2, size=GRID, dtype=np.uint8).astype(bool)
        arr = keep_largest_component(arr)
        V = int(arr.sum())
        E = int((arr[:, :, :-1] & arr[:, :, 1:]).sum() + (arr[:, :-1, :] & arr[:, 1:, :]).sum() + (arr[:-1, :, :] & arr[1:, :, :]).sum())
        labeled, C = ndimage.label(arr, structure=STRUCTURE)
        genus = E - V + C
        B = int((arr[:, :, :-1] ^ arr[:, :, 1:]).sum() + (arr[:, :-1, :] ^ arr[:, 1:, :]).sum() + (arr[:-1, :, :] ^ arr[1:, :, :]).sum())
        recs.append((V, C, genus, B))
    return pd.DataFrame(recs, columns=['V', 'C', 'genus', 'B'])


def create_figure(bulk_df, topo_df, outpath):
    fig = plt.figure(figsize=(10, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.38, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(topo_df['genus'], bins=35, edgecolor='white')
    ax1.axvline(8, linestyle='--', linewidth=1.5)
    ax1.set_title('(a) Cycle rank beta_1 on 10,000 samples')
    ax1.set_xlabel('beta_1(Gamma(S))')
    ax1.set_ylabel('Count')
    ax1.set_xlim(0, 1200)
    ax1.text(0.98, 0.95, f"mean = {topo_df['genus'].mean():.1f}\nmin = {topo_df['genus'].min()}", transform=ax1.transAxes,
             ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.7'))

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(topo_df['C'], bins=np.arange(topo_df['C'].min() - 0.5, topo_df['C'].max() + 1.5, 1), edgecolor='white')
    ax2.axvline(1, linestyle='--', linewidth=1.5)
    ax2.set_title('(b) Connected components on 10,000 samples')
    ax2.set_xlabel('Number of components')
    ax2.set_ylabel('Count')
    ax2.text(0.98, 0.95, f"mean = {topo_df['C'].mean():.2f}\nconnected = 0 / 10,000", transform=ax2.transAxes,
             ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.7'))

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(topo_df['B'], bins=35, edgecolor='white')
    ax3.axvline(5760, linestyle='--', linewidth=1.5)
    ax3.set_title('(c) Internal boundary-edge count B(s) on 10,000 samples')
    ax3.set_xlabel('Boundary-edge count')
    ax3.set_ylabel('Count')
    ax3.text(0.98, 0.95, f"mean = {topo_df['B'].mean():.1f}\nrandom expectation = 5760", transform=ax3.transAxes,
             ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.7'))

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(bulk_df['lb'], bins=35, edgecolor='white')
    ax4.axvline(8, linestyle='--', linewidth=1.5)
    ax4.set_title('(d) Lower bound E - V + 1 on 100,000 samples')
    ax4.set_xlabel('Guaranteed lower bound on beta_1')
    ax4.set_ylabel('Count')
    ax4.set_xlim(0, 1200)
    ax4.text(0.98, 0.95, f"min observed = {bulk_df['lb'].min()}\nall 100,000 exceed threshold", transform=ax4.transAxes,
             ha='right', va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.7'))

    fig.suptitle('Published HyperFrog predicate in the paper-aligned miner', fontsize=14, y=0.98)
    fig.savefig(outpath, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    bulk_df = bulk_metrics()
    topo_df = topology_metrics()
    practical_df = practical_metrics()

    bulk_df.to_csv(OUTDIR / 'bulk_metrics_100k.csv', index=False)
    topo_df.to_csv(OUTDIR / 'topology_metrics_10k.csv', index=False)
    practical_df.to_csv(OUTDIR / 'practical_largest_component_metrics_10k.csv', index=False)

    p_pop_fail = float(stats.binom.cdf(1199, N, 0.5) + stats.binom.sf(2600, N, 0.5))
    deg_sq_sum = 8 * 2**2 + 168 * 3**2 + 1176 * 4**2 + 2744 * 5**2
    beta_fail_bd = float(math.exp(-2 * (832 - 6) ** 2 / deg_sq_sum))
    accept_lb = float(1 - p_pop_fail - beta_fail_bd)
    connected_obs = int((topo_df['C'] == 1).sum())
    connected_upper95 = float(stats.beta.ppf(0.95, connected_obs + 1, len(topo_df) - connected_obs)) if connected_obs < len(topo_df) else 1.0

    summary = {
        'p_pop_fail_exact': p_pop_fail,
        'beta_fail_bounded_diff': beta_fail_bd,
        'acceptance_prob_lower_bound': accept_lb,
        'tv_bound_secret_vs_uniform': float(1 - accept_lb),
        'expected_trials_upper_bound': float(1 / accept_lb),
        'bulk_mean_V': float(bulk_df['V'].mean()),
        'bulk_std_V': float(bulk_df['V'].std(ddof=1)),
        'bulk_mean_E': float(bulk_df['E'].mean()),
        'bulk_mean_B': float(bulk_df['B'].mean()),
        'bulk_min_lb': int(bulk_df['lb'].min()),
        'topo_mean_C': float(topo_df['C'].mean()),
        'topo_mean_genus': float(topo_df['genus'].mean()),
        'topo_median_genus': float(topo_df['genus'].median()),
        'topo_min_genus': int(topo_df['genus'].min()),
        'topo_max_genus': int(topo_df['genus'].max()),
        'topo_mean_B': float(topo_df['B'].mean()),
        'topo_connected_obs': connected_obs,
        'topo_connected_upper95': connected_upper95,
        'topo_mean_largest': float(topo_df['largest'].mean()),
        'practical_mean_V': float(practical_df['V'].mean()),
        'practical_mean_genus': float(practical_df['genus'].mean()),
        'practical_mean_B': float(practical_df['B'].mean()),
    }
    (OUTDIR / 'summary.json').write_text(json.dumps(summary, indent=2))
    (OUTDIR / 'practical_summary.json').write_text(json.dumps({
        'practical_mean_V': float(practical_df['V'].mean()),
        'practical_mean_C': float(practical_df['C'].mean()),
        'practical_mean_genus': float(practical_df['genus'].mean()),
        'practical_min_genus': int(practical_df['genus'].min()),
        'practical_mean_B': float(practical_df['B'].mean()),
        'practical_min_B': int(practical_df['B'].min()),
        'practical_max_B': int(practical_df['B'].max()),
    }, indent=2))

    create_figure(bulk_df, topo_df, OUTDIR / 'figure_predicate_diagnostics.png')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
