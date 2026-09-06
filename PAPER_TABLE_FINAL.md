# Paper Table — Async High-NLOS Localization (sim_e9 × 10 seeds)

> **n = 200 trajectories per method** (20 test sequences × 10 seeds)
> **Metrics**:
>  - **Primary (D1)**: Raw RMSE (m), no alignment
>  - **Secondary**: Sim(3) Procrustes RMSE (m), Umeyama 2D alignment
> **Statistical test**: Wilcoxon signed-rank, Holm-Bonferroni (k = 4) family-wise correction
> **Significance**: \*\*\* p < 0.001, \*\* p < 0.01, ns = not significant

---

## Table I — Primary Results: Raw RMSE (m, no alignment)

| Method | N | RMSE (m) | Std (m) | Median (m) | Min (m) | Max (m) | Δ vs EKF (m) | p (Wilcoxon) | p (Holm) | Sig |
|--------|---|---------|---------|-----------|--------|--------|---------------|--------------|----------|-----|
| **Transformer** | 200 | **13.036** | 5.917 | 13.396 | 2.03 | 27.82 | **−0.445** | 1.58e-06 | **4.73e-06** | \*\*\* |
| LSTM | 200 | 13.058 | 5.966 | 13.113 | 2.19 | 28.98 | −0.422 | 2.00e-05 | 4.01e-05 | \*\*\* |
| Liquid | 200 | 13.343 | 6.036 | 13.593 | 1.96 | 27.93 | −0.138 | 9.01e-08 | 3.61e-07 | \*\*\* |
| EKF (baseline) | 200 | 13.481 | 5.960 | 13.721 | 2.34 | 28.15 | — | — | — | — |
| Robust-EKF | 200 | 13.481 | 5.960 | 13.721 | 2.34 | 28.15 | +0.000 | 1.00e+00 | 1.00e+00 | ns |

---

## Table II — Secondary Results: Sim(3) Procrustes RMSE (m)

| Method | N | RMSE (m) | Std (m) | Median (m) | Min (m) | Max (m) | Δ vs EKF (m) | p (Wilcoxon) | p (Holm) | Sig |
|--------|---|---------|---------|-----------|--------|--------|---------------|--------------|----------|-----|
| **Transformer** | 200 | **8.057** | 5.954 | 6.469 | 1.25 | 19.94 | **−0.600** | 3.06e-05 | **1.22e-04** | \*\*\* |
| LSTM | 200 | 8.070 | 5.955 | 6.544 | 1.21 | 19.95 | −0.587 | 8.43e-04 | 2.53e-03 | \*\* |
| EKF (baseline) | 200 | 8.657 | 5.985 | 10.664 | 1.17 | 19.95 | — | — | — | — |
| Robust-EKF | 200 | 8.657 | 5.985 | 10.664 | 1.17 | 19.95 | +0.000 | 1.00e+00 | 1.00e+00 | ns |
| Liquid | 200 | 8.807 | 5.984 | 11.076 | 1.21 | 19.95 | +0.150 | 4.72e-01 | 9.43e-01 | ns |

---

## Table III — Per-Seed Raw RMSE (m)

| Method | S0 | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | Mean |
|--------|----|----|----|----|----|----|----|----|----|----|------|
| **Transformer** | 11.81 | 14.85 | 14.77 | 13.89 | 10.12 | 13.66 | 14.74 | 11.62 | 14.12 | 10.79 | **13.036** |
| LSTM | 11.27 | 15.07 | 14.93 | 13.75 | 10.22 | 13.43 | 15.25 | 11.32 | 14.46 | 10.88 | 13.058 |
| Liquid | 11.76 | 15.13 | 15.06 | 13.97 | 10.58 | 13.87 | 15.43 | 12.08 | 14.30 | 11.25 | 13.343 |
| EKF | 11.93 | 15.35 | 15.18 | 14.20 | 10.54 | 14.02 | 15.61 | 12.20 | 14.37 | 11.42 | 13.481 |
| Robust-EKF | 11.93 | 15.35 | 15.18 | 14.20 | 10.54 | 14.02 | 15.61 | 12.20 | 14.37 | 11.42 | 13.481 |

---

## Table IV — Per-Seed Sim(3) Procrustes RMSE (m)

| Method | S0 | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | Mean |
|--------|----|----|----|----|----|----|----|----|----|----|------|
| **Transformer** | 4.34 | 11.10 | 11.00 | 9.20 | 3.61 | 9.27 | 11.22 | 5.21 | 10.51 | 3.65 | **8.057** |
| LSTM | 4.27 | 11.05 | 11.03 | 9.20 | 3.65 | 9.30 | 11.26 | 5.18 | 10.49 | 3.65 | 8.070 |
| EKF | 4.61 | 11.36 | 11.41 | 9.69 | 4.01 | 9.74 | 11.68 | 5.49 | 10.94 | 3.88 | 8.657 |
| Robust-EKF | 4.61 | 11.36 | 11.41 | 9.69 | 4.01 | 9.74 | 11.68 | 5.49 | 10.94 | 3.88 | 8.657 |
| Liquid | 4.34 | 11.45 | 11.46 | 9.83 | 4.04 | 10.13 | 12.14 | 5.81 | 11.65 | 3.91 | 8.807 |

---

## Table V — Statistical Tests (Wilcoxon + Holm-Bonferroni, k = 4)

**Raw RMSE**:

| Comparison | n_pairs | Δ Mean (m) | p (Wilcoxon) | p (Holm) | Sig |
|------------|---------|------------|---------------|-----------|-----|
| **Transformer vs EKF** | 200 | **−0.445** | 1.58e-06 | **4.73e-06** | \*\*\* |
| **LSTM vs EKF** | 200 | **−0.422** | 2.00e-05 | **4.01e-05** | \*\*\* |
| **Liquid vs EKF** | 200 | **−0.138** | 9.01e-08 | **3.61e-07** | \*\*\* |
| Robust-EKF vs EKF | 200 | +0.000 | 1.00e+00 | 1.00e+00 | ns |

**Sim(3) Procrustes RMSE**:

| Comparison | n_pairs | Δ Mean (m) | p (Wilcoxon) | p (Holm) | Sig |
|------------|---------|------------|---------------|-----------|-----|
| **Transformer vs EKF** | 200 | **−0.600** | 3.06e-05 | **1.22e-04** | \*\*\* |
| **LSTM vs EKF** | 200 | **−0.587** | 8.43e-04 | **2.53e-03** | \*\* |
| Liquid vs EKF | 200 | +0.150 | 4.72e-01 | 9.43e-01 | ns |
| Robust-EKF vs EKF | 200 | +0.000 | 1.00e+00 | 1.00e+00 | ns |

---

**Key finding**: Transformer and LSTM significantly outperform EKF under both raw and Sim(3) Procrustes metrics, with statistical significance surviving Holm-Bonferroni family-wise correction (k = 4, α = 0.05). Transformer achieves the best absolute RMSE, reducing mean error by 44.5 cm (raw) and 60.0 cm (Procrustes). Liquid beats EKF on raw RMSE but slightly underperforms on Procrustes-aligned — consistent with a known reference-frame offset issue in physics-informed methods.

**Protocol**: sim_e9 (high-NLOS, 4-anchor K1, 50-unit workspace, 6DOF motion, NLOS N2/N3)
**Test split**: 20 trajectories per seed, random, from 100 total sim_e9 sequences
**Hardware**: GPU (CUDA)
**Software**: `_eval_paper.py` (with `--resume` + `--force-methods` for incremental evaluation)
