# Adaptive Treatment Recommendation System — Multi-Armed Bandit

> A clinical treatment-selection problem modeled as a Multi-Armed Bandit (MAB). Simulates 1,000 synthetic patients across 7 medicines with hidden success probabilities, and compares Greedy, Epsilon-Greedy (ε = 0.01 / 0.10 / 0.50), and UCB1 strategies to recommend the safest real-world deployment policy.

> **Course**: Deep Reinforcement Learning — Lab Assignment 1 (BITS Pilani, WILP Division) | **Group ID**: 176

---

## Output

![MAB Comparison Chart](outputs/Team_176_MAB_comparison.png)

---

## Features

- **Synthetic Patient Dataset** — 1,000 patient records generated deterministically from a group-specific seed (`G = 176`), each with a `patient_id` and cyclic `severity_score` (1–5)
- **Group-Derived Environment Parameters**:
  - Number of medicines `K = (G mod 3) + 5 = 7`
  - Hidden success probability per medicine `P_i = 0.4 + ((G + i) mod 6) × 0.07`
  - Optimal medicine: **Medicine 3** (P = 0.75)
- **Greedy Strategy** — round-robin warm-up (10 pulls per arm) followed by pure exploitation of the empirically best arm
- **Epsilon-Greedy Strategy** — evaluated at ε ∈ {0.01, 0.10, 0.50}, balancing random exploration against exploitation of the current best arm
- **UCB1 Strategy** — confidence-bound arm selection using `Q(a) + sqrt(2·ln(t) / N(a))`
- **Utility-Based Reward** — `utility = clinical_outcome × (1 − severity / 10)`, so reward accounts for both treatment success and patient severity
- **Comparative Visualisation** — side-by-side plots of cumulative reward over all 1,000 patients and a zoomed view of the first 200 patients (convergence detail)
- **Comparative Summary Table & Analytical Q&A** — final cumulative reward and identified best arm per strategy, plus answers to four deployment-focused analytical questions

---

## Architecture / Pipeline

```
GROUP_NUMBER = 176
        │
        ▼  compute_group_params(G)
        │     → K = 7 medicines
        │     → True hidden success probabilities P_i (per medicine)
        │
        ▼  create_patient_dataframe(n_patients=1000)
        │     → patient_id, severity_score (cyclic 1–5)
        │
        ├──────────────┬───────────────────┬─────────────────────┐
        ▼              ▼                   ▼                     ▼
   run_greedy()   run_epsilon_greedy()  run_epsilon_greedy()  run_ucb1()
   (warm-up +     (ε = 0.01, 0.10,      (per ε value)         (confidence
    exploit)       0.50)                                       bonus)
        │              │                   │                     │
        └──────────────┴───────────────────┴─────────────────────┘
                                  │
                                  ▼
                  Comparative Plot + Summary Table
                  → outputs/Team_176_MAB_comparison.png
                  → Task 5 analytical Q&A printed to console
```

---

## Project Structure

```
.
├── mab_treatment_recommendation.py   # Complete MAB pipeline (dataset, 3 strategies, plots, analysis)
├── outputs/
│   └── Team_176_MAB_comparison.png   # Generated comparison chart (cumulative reward)
├── requirements.txt
└── .gitignore
```

---

## Dataset Design

### Group-Specific Parameters (G = 176)

| Parameter | Formula | Value (G = 176) |
|---|---|---|
| Number of medicines K | `(G mod 3) + 5` | 7 |
| Hidden probability P_i | `0.4 + ((G + i) mod 6) × 0.07` | see table below |
| Patient severity | `(patient_id mod 5) + 1` | 1 to 5 (cyclic) |
| Utility score | `outcome × (1 − severity / 10)` | 0.0 to 0.9 |
| Random seeds | `random.seed(G)`, `np.random.seed(G)` | 176 |

### Hidden Success Probabilities

| Medicine | Formula | P_i |
|---|---|---|
| Medicine 0 | 0.4 + ((176+0) mod 6) × 0.07 | 0.61 |
| Medicine 1 | 0.4 + ((176+1) mod 6) × 0.07 | 0.68 |
| Medicine 2 | 0.4 + ((176+2) mod 6) × 0.07 | 0.75 |
| Medicine 3 | 0.4 + ((176+3) mod 6) × 0.07 | 0.40 |
| Medicine 4 | 0.4 + ((176+4) mod 6) × 0.07 | 0.47 |
| Medicine 5 | 0.4 + ((176+5) mod 6) × 0.07 | 0.54 |
| Medicine 6 | 0.4 + ((176+6) mod 6) × 0.07 | 0.61 |

> Optimal medicine (highest P): **Medicine 2** (P = 0.75)

### Patient DataFrame Schema

| Column | Description | Populated by |
|---|---|---|
| `patient_id` | Patient index (0–999) | Static |
| `severity_score` | Disease severity (1–5), cyclic | Static |
| `assigned_medicine` | Medicine selected by algorithm | Algorithm |
| `clinical_outcome` | Binary recovery (0/1) | Algorithm |
| `utility_score` | `outcome × (1 − severity/10)` | Algorithm |

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/adaptive-treatment-mab.git
cd adaptive-treatment-mab
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Run the complete pipeline:

```bash
python mab_treatment_recommendation.py
```

This will:

1. Print execution metadata (timestamp, hostname)
2. Compute group-specific parameters (K=7 medicines, hidden probabilities)
3. Generate the 1,000-patient dataset and print the first 10 records
4. Run **Task 2 — Greedy** and print the cumulative reward and exploited medicine
5. Run **Task 3 — Epsilon-Greedy** for ε ∈ {0.01, 0.10, 0.50} and print results for each
6. Run **Task 4 — UCB1** and print the cumulative reward and pull counts
7. Generate the comparative plot (`outputs/Team_176_MAB_comparison.png`) showing:
   - Cumulative reward over all 1,000 patients
   - Zoomed convergence detail for the first 200 patients
8. Print the **Task 5** summary table (final cumulative reward and best arm per strategy)
9. Print answers to the four analytical questions and a written comparative summary

### Strategy Comparison (Results from this run)

| Strategy | Final Cumulative Reward | Best Arm Identified |
|---|---|---|
| Greedy (Exploit) | 498.70 | Medicine 3 |
| ε-Greedy (ε=0.01) | 494.45 | Medicine 2 |
| ε-Greedy (ε=0.10) | 494.40 | Medicine 2 |
| ε-Greedy (ε=0.50) | 347.62 | Medicine 2 |
| UCB1 | 430.10 | Medicine 2 |

> **True best medicine: Medicine 2 (P = 0.75)**

### Recommendation

**ε-Greedy (ε = 0.10)** is recommended for real-world hospital deployment — 90% of patients receive the current best-known treatment while 10% continuously validate and refine that choice. UCB1 is theoretically optimal (O(log n) regret) but its early exploration cost exposes more patients to suboptimal treatments at this horizon (n = 1000).

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Random sampling, reward/probability computation |
| `pandas` | Patient dataset and per-strategy result DataFrames |
| `matplotlib` | Comparative cumulative-reward plots |

---

## License

This project is provided for educational and academic demonstration purposes.
