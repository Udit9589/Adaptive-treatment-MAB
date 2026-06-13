"""
============================================================
 Adaptive Treatment Recommendation System — MAB
 BITS Pilani — Deep Reinforcement Learning — Lab Assignment 1
 Group ID: 176
============================================================

Part 1 — Multi-Armed Bandit (MAB)
Algorithms : Greedy, Epsilon-Greedy (eps = 0.01 / 0.10 / 0.50), UCB1
Dataset    : 1000 synthetic patient records

This script models a clinical treatment-selection problem as a
Multi-Armed Bandit. A hospital evaluating K medicines for a chronic
disease learns from patient outcomes over time and progressively
identifies the optimal medicine using three bandit strategies.
"""

# ── Printing timestamp and hostname ─────────────────────────────────────────
import datetime, socket

print("=" * 60)
print("Execution timestamp:", datetime.datetime.now())
print("VM hostname:", socket.gethostname())
print("=" * 60)

# ── Importing dependencies ──────────────────────────────────────────────────
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import math


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — Dataset Design (Task 1)
# ═══════════════════════════════════════════════════════════════════════════

# Reproducibility Seeds
# Both Python's built-in random and NumPy are seeded with the group number
# to guarantee identical results across every run.
GROUP_NUMBER = 176  # group id
random.seed(GROUP_NUMBER)
np.random.seed(GROUP_NUMBER)


def compute_group_params(G: int):
    """
    Derive all dataset constants deterministically from the group number G.

    K     = (G mod 3) + 5                     -> number of medicines
    P_i   = 0.4 + ((G + i) mod 6) * 0.07      -> hidden success probability
    """
    K = (G % 3) + 5  # K = (G mod 3) + 5
    probs = [0.4 + ((G + i) % 6) * 0.07  # P_i formula
             for i in range(K)]
    return K, probs


K, TRUE_PROBS = compute_group_params(GROUP_NUMBER)

print("=" * 60)
print(" ADAPTIVE TREATMENT RECOMMENDATION SYSTEM – MAB")
print("=" * 60)
print(f" Group Number (G)            : {GROUP_NUMBER}")
print(f" No. of Medicines (K)        : {K}")
print()
print(" Hidden Success Probabilities per Medicine:")
for i, p in enumerate(TRUE_PROBS):
    print(f"   Medicine {i} -> P_{i} = {p:.2f}")
print()
print(f" Optimal Medicine (highest P) : Medicine "
      f"{int(np.argmax(TRUE_PROBS))} "
      f"(P = {max(TRUE_PROBS):.2f})")
print("=" * 60)


# 1.2 Environment helper functions
def compute_severity(patient_id: int) -> int:
    """Disease severity (1-5), cyclic: severity = (patient_id mod 5) + 1"""
    return (patient_id % 5) + 1


def simulate_outcome(medicine_idx: int, probs: list) -> int:
    """Bernoulli draw: 1 = recovery, 0 = no recovery, using the hidden P_i."""
    return int(np.random.rand() < probs[medicine_idx])


def compute_utility(clinical_outcome: int, severity: int) -> float:
    """utility = outcome * (1 - severity / 10)"""
    return clinical_outcome * (1.0 - severity / 10.0)


# 1.3 Initialise the patient DataFrame (static columns)
def create_patient_dataframe(n_patients: int = 1000) -> pd.DataFrame:
    patient_ids = list(range(n_patients))
    severities = [compute_severity(pid) for pid in patient_ids]
    df = pd.DataFrame({
        "patient_id": patient_ids,
        "severity_score": severities,
        # Columns below are populated dynamically by each algorithm
        "assigned_medicine": np.nan,
        "clinical_outcome": np.nan,
        "utility_score": np.nan,
    })
    return df


# Create the base patient frame (reset seeds first so ordering is deterministic)
random.seed(GROUP_NUMBER)
np.random.seed(GROUP_NUMBER)
patient_df = create_patient_dataframe(n_patients=1000)

print("\nFirst 10 Patient Records (static columns only):")
print(patient_df.head(10).to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — Task 2: Greedy (Immediate Exploitation) Strategy
# ═══════════════════════════════════════════════════════════════════════════
#
# Policy: The strategy runs in two sequential phases.
#   Phase 1 (Exploration)  : assign every medicine exactly `init_trials`
#                             times via round-robin (K * init_trials pulls)
#   Phase 2 (Exploitation) : identify the best medicine from phase-1 data
#                             and prescribe it exclusively for all
#                             remaining patients
#
def run_greedy(K: int, probs: list, n_patients: int = 1000,
               init_trials: int = 10) -> dict:
    # ── Tracking variables ───────────────────────────────────────────────
    counts = np.zeros(K, dtype=int)        # pull counts per arm
    sum_outcomes = np.zeros(K)             # sum of clinical outcomes
    records = []                           # per-patient log
    cumulative_rewards = []                # running sum of utility scores
    cum_reward = 0.0

    for patient_id in range(n_patients):
        severity = compute_severity(patient_id)

        # ── Arm selection ────────────────────────────────────────────────
        if patient_id < K * init_trials:
            # Phase 1: round-robin exploration – ensure every arm is tried
            # init_trials times before exploitation begins.
            medicine = patient_id % K
        else:
            # Phase 2: pure exploitation – always choose the arm with the
            # highest empirical success rate observed so far.
            mean_outcomes = np.divide(sum_outcomes, counts,
                                       out=np.zeros_like(sum_outcomes),
                                       where=counts > 0)
            medicine = int(np.argmax(mean_outcomes))

        # ── Simulate outcome ─────────────────────────────────────────────
        outcome = simulate_outcome(medicine, probs)
        utility = compute_utility(outcome, severity)

        # ── Update arm statistics ────────────────────────────────────────
        counts[medicine] += 1
        sum_outcomes[medicine] += outcome

        # ── Accumulate reward ────────────────────────────────────────────
        cum_reward += utility
        cumulative_rewards.append(cum_reward)

        records.append({
            "patient_id": patient_id,
            "severity_score": severity,
            "assigned_medicine": medicine,
            "clinical_outcome": outcome,
            "utility_score": utility,
        })

    return {
        "df": pd.DataFrame(records),
        "cumulative_rewards": cumulative_rewards,
        "counts": counts,
        "strategy": "Greedy (Exploit)",
    }


# Re-seed before running to ensure fair comparison across strategies
np.random.seed(GROUP_NUMBER)
greedy_results = run_greedy(K, TRUE_PROBS)

print("=" * 60)
print(" TASK 2 – GREEDY (IMMEDIATE EXPLOITATION) STRATEGY")
print("=" * 60)
print(f" Total Cumulative Reward : "
      f"{greedy_results['cumulative_rewards'][-1]:.4f}")
print(f" Medicine Pull Counts    : "
      f"{dict(enumerate(greedy_results['counts']))}")

# Identify which medicine was ultimately exploited (most-pulled post warm-up)
exploit_med = int(np.argmax(greedy_results['counts']))
print(f" Exploited Medicine      : Medicine {exploit_med} "
      f"(P = {TRUE_PROBS[exploit_med]:.2f})")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — Task 3: Epsilon-Greedy (Controlled Clinical Trial)
# ═══════════════════════════════════════════════════════════════════════════
#
# Policy: At each patient step, one of two actions is taken.
#   Explore (probability eps)     : choose a medicine uniformly at random
#   Exploit (probability 1 - eps) : choose the current best-performing medicine
#
# Three exploration rates are evaluated: eps in {0.01, 0.10, 0.50}.
#
def run_epsilon_greedy(K: int, probs: list, epsilon: float,
                       n_patients: int = 1000) -> dict:
    # ── Tracking variables ───────────────────────────────────────────────
    counts = np.zeros(K, dtype=int)
    sum_outcomes = np.zeros(K)
    records = []
    cumulative_rewards = []
    cum_reward = 0.0

    for patient_id in range(n_patients):
        severity = compute_severity(patient_id)

        # ── Arm selection ────────────────────────────────────────────────
        if np.random.rand() < epsilon:
            # Exploration: uniformly random medicine
            medicine = np.random.randint(0, K)
        else:
            if counts.sum() == 0:
                # No data yet – pick randomly to avoid cold-start ties
                medicine = np.random.randint(0, K)
            else:
                mean_outcomes = np.divide(sum_outcomes, counts,  # Q(a) — safe divide
                                           out=np.zeros_like(sum_outcomes),
                                           where=counts > 0)
                medicine = int(np.argmax(mean_outcomes))

        # ── Simulate outcome ─────────────────────────────────────────────
        outcome = simulate_outcome(medicine, probs)
        utility = compute_utility(outcome, severity)

        # ── Update statistics ────────────────────────────────────────────
        counts[medicine] += 1
        sum_outcomes[medicine] += outcome

        cum_reward += utility
        cumulative_rewards.append(cum_reward)

        records.append({
            "patient_id": patient_id,
            "severity_score": severity,
            "assigned_medicine": medicine,
            "clinical_outcome": outcome,
            "utility_score": utility,
        })

    return {
        "df": pd.DataFrame(records),
        "cumulative_rewards": cumulative_rewards,
        "counts": counts,
        "strategy": f"eps-Greedy (eps={epsilon})",
    }


print("=" * 60)
print(" TASK 3 – EPSILON-GREEDY (CONTROLLED CLINICAL TRIAL)")
print("=" * 60)

epsilon_results = {}
for eps in [0.10, 0.01, 0.50]:
    np.random.seed(GROUP_NUMBER)
    res = run_epsilon_greedy(K, TRUE_PROBS, epsilon=eps)
    epsilon_results[eps] = res
    print(f" eps = {eps:.2f} -> Cumulative Reward : "
          f"{res['cumulative_rewards'][-1]:.4f} | "
          f"Pulls: {dict(enumerate(res['counts']))}")

print()
print(" Analysis:")
print("   eps = 0.01 : Very low exploration -> fast convergence to (possibly")
print("                sub-optimal) arm; misses better arms if warm-up unlucky.")
print("   eps = 0.10 : Balanced exploration/exploitation; steady improvement.")
print("   eps = 0.50 : 50% random exploration -> wastes many patients on poor")
print("                arms; cumulative reward is significantly lower.")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4 — Task 4: UCB1 (Confidence-Based) Strategy
# ═══════════════════════════════════════════════════════════════════════════
#
# Policy: Select the arm that maximises the UCB1 score:
#   UCB1(a) = Q(a) + sqrt( 2 * ln(t) / N(a) )
#
#   Q(a)                       : empirical mean clinical outcome for arm a
#   t                          : total patients seen so far (step count)
#   N(a)                       : number of times arm a has been pulled
#   sqrt(2 * ln(t) / N(a))     : exploration bonus, large when under-sampled
#
def run_ucb1(K: int, probs: list, n_patients: int = 1000) -> dict:
    # ── Tracking variables ───────────────────────────────────────────────
    counts = np.zeros(K, dtype=int)
    sum_outcomes = np.zeros(K)
    records = []
    cumulative_rewards = []
    cum_reward = 0.0

    for patient_id in range(n_patients):
        severity = compute_severity(patient_id)
        t = patient_id + 1  # 1-indexed total step count

        # ── Arm selection ────────────────────────────────────────────────
        if patient_id < K:
            # Initialisation: pull every arm once to avoid division by zero
            medicine = patient_id
        else:
            # UCB1 formula: mean outcome + exploration bonus
            mean_outcomes = sum_outcomes / counts  # Q(a)
            exploration_bonus = np.sqrt(2 * np.log(t) / counts)  # UCB term
            ucb_values = mean_outcomes + exploration_bonus
            medicine = int(np.argmax(ucb_values))

        # ── Simulate outcome ─────────────────────────────────────────────
        outcome = simulate_outcome(medicine, probs)
        utility = compute_utility(outcome, severity)

        # ── Update statistics ────────────────────────────────────────────
        counts[medicine] += 1
        sum_outcomes[medicine] += outcome

        cum_reward += utility
        cumulative_rewards.append(cum_reward)

        records.append({
            "patient_id": patient_id,
            "severity_score": severity,
            "assigned_medicine": medicine,
            "clinical_outcome": outcome,
            "utility_score": utility,
        })

    return {
        "df": pd.DataFrame(records),
        "cumulative_rewards": cumulative_rewards,
        "counts": counts,
        "strategy": "UCB1",
    }


np.random.seed(GROUP_NUMBER)
ucb_results = run_ucb1(K, TRUE_PROBS)

print("=" * 60)
print(" TASK 4 – UCB1 (CONFIDENCE-BASED) STRATEGY")
print("=" * 60)
print(f" Total Cumulative Reward : "
      f"{ucb_results['cumulative_rewards'][-1]:.4f}")
print(f" Medicine Pull Counts    : "
      f"{dict(enumerate(ucb_results['counts']))}")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5 — Task 5: Comparative Plot and Analysis
# ═══════════════════════════════════════════════════════════════════════════

# 5.1 Gather all curves for the comparison plot
all_strategies = {
    "Greedy (Exploit)": greedy_results["cumulative_rewards"],
    "eps-Greedy eps=0.01": epsilon_results[0.01]["cumulative_rewards"],
    "eps-Greedy eps=0.10": epsilon_results[0.10]["cumulative_rewards"],
    "eps-Greedy eps=0.50": epsilon_results[0.50]["cumulative_rewards"],
    "UCB1": ucb_results["cumulative_rewards"],
}

# Colour palette – distinct, accessible colours
COLORS = {
    "Greedy (Exploit)": "#E63946",     # vivid red
    "eps-Greedy eps=0.01": "#F4A261",  # amber
    "eps-Greedy eps=0.10": "#2A9D8F",  # teal
    "eps-Greedy eps=0.50": "#8338EC",  # violet
    "UCB1": "#023E8A",                 # deep blue
}

# 5.2 Plot – Cumulative Reward vs Number of Patients
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    f"Multi-Armed Bandit - Adaptive Treatment Recommendation\n"
    f"Group {GROUP_NUMBER} | K={K} Medicines | 1000 Patients",
    fontsize=14, fontweight="bold", y=1.02
)

patients = list(range(1, 1001))

# ── Left panel: full 1000-patient horizon ───────────────────────────────────
ax1 = axes[0]
for label, curve in all_strategies.items():
    ax1.plot(patients, curve, label=label, color=COLORS[label],
             linewidth=1.8, alpha=0.85)
ax1.set_title("Cumulative Reward over 1000 Patients", fontsize=12)
ax1.set_xlabel("Number of Patients", fontsize=11)
ax1.set_ylabel("Cumulative Utility Score (Reward)", fontsize=11)
ax1.legend(fontsize=9, loc="upper left")
ax1.grid(True, linestyle="--", alpha=0.4)

# ── Right panel: zoom on first 200 patients (convergence detail) ────────────
ax2 = axes[1]
for label, curve in all_strategies.items():
    ax2.plot(patients[:200], curve[:200], label=label,
             color=COLORS[label], linewidth=1.8, alpha=0.85)
ax2.set_title("First 200 Patients (Convergence Detail)", fontsize=12)
ax2.set_xlabel("Number of Patients", fontsize=11)
ax2.set_ylabel("Cumulative Utility Score (Reward)", fontsize=11)
ax2.legend(fontsize=9, loc="upper left")
ax2.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()

import os
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/Team_176_MAB_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print(" Plot saved -> outputs/Team_176_MAB_comparison.png")
print()

# 5.3 Summary table – final cumulative rewards
print("=" * 60)
print(" TASK 5 – COMPARATIVE SUMMARY TABLE")
print("=" * 60)
print(f" {'Strategy':<22} {'Final Cum. Reward':>18} {'Best Arm Identified':>20}")
print(" " + "-" * 62)

for label, curve in all_strategies.items():
    if label == "Greedy (Exploit)":
        df_ref = greedy_results["df"]
    elif label == "UCB1":
        df_ref = ucb_results["df"]
    elif "0.01" in label:
        df_ref = epsilon_results[0.01]["df"]
    elif "0.10" in label:
        df_ref = epsilon_results[0.10]["df"]
    else:
        df_ref = epsilon_results[0.50]["df"]

    # Best arm = most frequently assigned medicine in last 100 patients
    last100 = df_ref.tail(100)["assigned_medicine"].value_counts().idxmax()
    print(f" {label:<22} {curve[-1]:>18.4f} {'Medicine ' + str(int(last100)):>20}")

print()
print(f" True Best Medicine: Medicine {int(np.argmax(TRUE_PROBS))} "
      f"(P = {max(TRUE_PROBS):.2f})")
print()

# 5.4 Answering the four analytical questions
print("=" * 60)
print(" ANALYTICAL QUESTIONS")
print("=" * 60)

final_rewards = {k: v[-1] for k, v in all_strategies.items()}
best_strategy = max(final_rewards, key=final_rewards.get)

print(f"\n Q1. Highest cumulative reward at 1000 patients?")
print(f"    -> {best_strategy} ({final_rewards[best_strategy]:.4f})")

print(f"\n Q2. Fastest convergence to best medicine?")
print(f"    -> UCB1 - its confidence bonus forces systematic exploration of")
print(f"       all arms early on, then rapidly locks onto the best medicine.")

print(f"\n Q3. Most stable performance over time (least fluctuations)?")
print(f"    -> eps-Greedy (eps=0.10) - the constant 10% exploration rate")
print(f"       produces a smooth, nearly linear cumulative reward curve")
print(f"       once the best arm is identified.")

print(f"\n Q4. Safest real-world hospital deployment strategy?")
print(f"    -> eps-Greedy (eps=0.10) is recommended.")
print(f"       It provides a principled balance: 90% of patients receive the")
print(f"       current best-known treatment (ethical duty of care) while 10%")
print(f"       are used to continuously validate and potentially update that")
print(f"       choice. UCB1 is theoretically optimal but its aggressive early")
print(f"       exploration exposes more patients to suboptimal treatments.")

print()
print("=" * 60)
print(" COMPARATIVE SUMMARY (3-5 sentences)")
print("=" * 60)
print("""
The Greedy strategy converges quickly but is brittle - if its warm-up
phase (K * init_trials initial pulls across K arms) happens to favour a
sub-optimal arm, it will exploit that arm forever, limiting long-run
reward. The eps-Greedy variant with eps=0.50 wastes too many patients on
random choices and consistently yields the lowest cumulative reward.
eps-Greedy with eps=0.10 strikes a reliable balance, achieving high
cumulative reward with smooth, predictable growth. UCB1 is the
theoretical winner: its uncertainty-driven exploration identifies the
optimal medicine earliest and achieves a high cumulative reward, making
it the most mathematically principled approach; however, its higher
early exploration cost means eps-Greedy (eps=0.10) remains the
preferable choice under strict patient-welfare constraints.
""")
