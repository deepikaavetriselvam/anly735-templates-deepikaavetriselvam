"""
ANLY 735 - Replication Laboratory #2
Can the Model Keep Learning?

Proxy replication inspired by Klein et al. (2024),
"Plasticity Loss in Deep Reinforcement Learning: A Survey."

Research question:
Does prior training alter a neural network's capacity to learn
after the target relationship changes?

The experiment compares:
1. A continued learner previously trained on Task A.
2. A freshly initialized learner beginning at Task B.

Learning capacity is evaluated from the post-change trajectory,
not from the immediate performance drop alone.
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Reproducibility and experimental configuration
# ------------------------------------------------------------

SEEDS = [11, 22, 33, 44, 55]

N_TRAIN = 1200
N_TEST = 600

PRETRAIN_EPOCHS = 150
POSTCHANGE_EPOCHS = 150

LEARNING_RATE = 0.01
HIDDEN_UNITS = 64

# Evaluate early adaptation over a prespecified window.
EARLY_WINDOW = 20

# Adaptation threshold is defined relative to the fresh learner's
# final Task-B performance for each seed.
THRESHOLD_MULTIPLIER = 1.10

DEVICE = torch.device("cpu")


# ------------------------------------------------------------
# 2. Data-generating process
# ------------------------------------------------------------

def make_inputs(n, seed):
    """Generate a fixed two-dimensional input distribution."""
    generator = torch.Generator().manual_seed(seed)
    return torch.empty(n, 2).uniform_(-2.0, 2.0, generator=generator)


def task_a(x):
    """
    Initial nonlinear target relationship.
    """
    return (
        torch.sin(1.5 * x[:, 0])
        + 0.35 * x[:, 1] ** 2
        - 0.25 * x[:, 0] * x[:, 1]
    ).unsqueeze(1)


def task_b(x):
    """
    Changed nonlinear target relationship.

    The input distribution is unchanged; only the target mapping
    changes. This isolates adaptation to target nonstationarity.
    """
    return (
        torch.cos(1.5 * x[:, 0])
        - 0.35 * x[:, 1] ** 2
        + 0.25 * x[:, 0] * x[:, 1]
    ).unsqueeze(1)


# ------------------------------------------------------------
# 3. Neural-network model
# ------------------------------------------------------------

class RegressionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS),
            nn.ReLU(),
            nn.Linear(HIDDEN_UNITS, 1),
        )

    def forward(self, x):
        return self.network(x)


# ------------------------------------------------------------
# 4. Training and evaluation utilities
# ------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mse(model, x, y):
    model.eval()
    with torch.no_grad():
        prediction = model(x)
        return nn.functional.mse_loss(prediction, y).item()


def train_one_epoch(model, optimizer, x, y):
    model.train()
    optimizer.zero_grad()

    prediction = model(x)
    loss = nn.functional.mse_loss(prediction, y)

    loss.backward()
    optimizer.step()

    return loss.item()


def early_log_improvement(losses, window=EARLY_WINDOW):
    """
    Estimate early adaptation rate from the slope of log(MSE).

    More negative values indicate faster proportional reduction
    in Task-B error.
    """
    usable = np.asarray(losses[: window + 1], dtype=float)
    epochs = np.arange(len(usable))

    slope = np.polyfit(
        epochs,
        np.log(np.clip(usable, 1e-12, None)),
        1,
    )[0]

    return float(slope)


def epochs_to_threshold(losses, threshold):
    """
    Return the first post-change epoch at which MSE reaches
    the prespecified threshold. Returns NaN if never reached.
    """
    for epoch, loss in enumerate(losses):
        if loss <= threshold:
            return epoch

    return np.nan


# ------------------------------------------------------------
# 5. Single-seed experiment
# ------------------------------------------------------------

def run_seed(seed):
    set_seed(seed)

    # Independent train/test samples from the same input distribution.
    x_train = make_inputs(N_TRAIN, seed + 1000).to(DEVICE)
    x_test = make_inputs(N_TEST, seed + 2000).to(DEVICE)

    y_a_train = task_a(x_train).to(DEVICE)
    y_a_test = task_a(x_test).to(DEVICE)

    y_b_train = task_b(x_train).to(DEVICE)
    y_b_test = task_b(x_test).to(DEVICE)

    # --------------------------------------------------------
    # Phase 1: learn Task A
    # --------------------------------------------------------

    set_seed(seed)
    continued = RegressionNet().to(DEVICE)

    optimizer_a = torch.optim.Adam(
        continued.parameters(),
        lr=LEARNING_RATE,
    )

    for _ in range(PRETRAIN_EPOCHS):
        train_one_epoch(
            continued,
            optimizer_a,
            x_train,
            y_a_train,
        )

    task_a_before_change = mse(
        continued,
        x_test,
        y_a_test,
    )

    # Evaluate the trained model on Task B BEFORE any Task-B updates.
    # This measures the immediate effect of the change.
    continued_b_losses = [
        mse(continued, x_test, y_b_test)
    ]

    # --------------------------------------------------------
    # Fresh Task-B baseline
    # --------------------------------------------------------

    set_seed(seed + 10000)
    fresh = RegressionNet().to(DEVICE)

    fresh_b_losses = [
        mse(fresh, x_test, y_b_test)
    ]

    optimizer_continued = torch.optim.Adam(
        continued.parameters(),
        lr=LEARNING_RATE,
    )

    optimizer_fresh = torch.optim.Adam(
        fresh.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # Phase 2: both models learn Task B
    # --------------------------------------------------------

    trajectory_rows = []

    trajectory_rows.append(
        {
            "seed": seed,
            "epoch": 0,
            "condition": "Continued learner",
            "task_b_mse": continued_b_losses[0],
            "task_a_mse": mse(
                continued,
                x_test,
                y_a_test,
            ),
        }
    )

    trajectory_rows.append(
        {
            "seed": seed,
            "epoch": 0,
            "condition": "Fresh learner",
            "task_b_mse": fresh_b_losses[0],
            "task_a_mse": mse(
                fresh,
                x_test,
                y_a_test,
            ),
        }
    )

    for epoch in range(1, POSTCHANGE_EPOCHS + 1):

        train_one_epoch(
            continued,
            optimizer_continued,
            x_train,
            y_b_train,
        )

        train_one_epoch(
            fresh,
            optimizer_fresh,
            x_train,
            y_b_train,
        )

        continued_b = mse(
            continued,
            x_test,
            y_b_test,
        )

        fresh_b = mse(
            fresh,
            x_test,
            y_b_test,
        )

        continued_b_losses.append(continued_b)
        fresh_b_losses.append(fresh_b)

        trajectory_rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "condition": "Continued learner",
                "task_b_mse": continued_b,
                "task_a_mse": mse(
                    continued,
                    x_test,
                    y_a_test,
                ),
            }
        )

        trajectory_rows.append(
            {
                "seed": seed,
                "epoch": epoch,
                "condition": "Fresh learner",
                "task_b_mse": fresh_b,
                "task_a_mse": mse(
                    fresh,
                    x_test,
                    y_a_test,
                ),
            }
        )

    # --------------------------------------------------------
    # Diagnostic metrics
    # --------------------------------------------------------

    # Threshold is based on the fresh learner's final Task-B loss.
    threshold = (
        fresh_b_losses[-1]
        * THRESHOLD_MULTIPLIER
    )

    summary = {
        "seed": seed,

        # Stability / retention
        "task_a_mse_before_change":
            task_a_before_change,

        "task_a_mse_after_adaptation":
            mse(
                continued,
                x_test,
                y_a_test,
            ),

        # Immediate disruption
        "continued_task_b_initial_mse":
            continued_b_losses[0],

        "fresh_task_b_initial_mse":
            fresh_b_losses[0],

        # New learning
        "continued_task_b_final_mse":
            continued_b_losses[-1],

        "fresh_task_b_final_mse":
            fresh_b_losses[-1],

        "continued_task_b_improvement":
            continued_b_losses[0]
            - continued_b_losses[-1],

        "fresh_task_b_improvement":
            fresh_b_losses[0]
            - fresh_b_losses[-1],

        # Rate
        "continued_early_log_slope":
            early_log_improvement(
                continued_b_losses
            ),

        "fresh_early_log_slope":
            early_log_improvement(
                fresh_b_losses
            ),

        # Time to criterion
        "task_b_threshold": threshold,

        "continued_epochs_to_threshold":
            epochs_to_threshold(
                continued_b_losses,
                threshold,
            ),

        "fresh_epochs_to_threshold":
            epochs_to_threshold(
                fresh_b_losses,
                threshold,
            ),
    }

    return (
        pd.DataFrame(trajectory_rows),
        summary,
    )


# ------------------------------------------------------------
# 6. Run all seeds
# ------------------------------------------------------------

all_trajectories = []
all_summaries = []

for seed in SEEDS:
    print(f"Running seed {seed}...")

    trajectory, summary = run_seed(seed)

    all_trajectories.append(trajectory)
    all_summaries.append(summary)


trajectories = pd.concat(
    all_trajectories,
    ignore_index=True,
)

summary = pd.DataFrame(all_summaries)


# ------------------------------------------------------------
# 7. Aggregate results
# ------------------------------------------------------------

aggregate = (
    trajectories
    .groupby(["condition", "epoch"])
    .agg(
        mean_task_b_mse=("task_b_mse", "mean"),
        sd_task_b_mse=("task_b_mse", "std"),
        mean_task_a_mse=("task_a_mse", "mean"),
        sd_task_a_mse=("task_a_mse", "std"),
    )
    .reset_index()
)


# ------------------------------------------------------------
# 8. Save reproducibility evidence
# ------------------------------------------------------------

Path("results").mkdir(exist_ok=True)
Path("figures").mkdir(exist_ok=True)

trajectories.to_csv(
    "results/learning_trajectories.csv",
    index=False,
)

summary.to_csv(
    "results/seed_summary.csv",
    index=False,
)

aggregate.to_csv(
    "results/aggregate_trajectories.csv",
    index=False,
)


# ------------------------------------------------------------
# 9. Create post-change learning figure
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

for condition in [
    "Continued learner",
    "Fresh learner",
]:
    subset = aggregate[
        aggregate["condition"] == condition
    ]

    ax.plot(
        subset["epoch"],
        subset["mean_task_b_mse"],
        label=condition,
    )

    lower = np.maximum(
        subset["mean_task_b_mse"]
        - subset["sd_task_b_mse"],
        1e-12,
    )

    upper = (
        subset["mean_task_b_mse"]
        + subset["sd_task_b_mse"]
    )

    ax.fill_between(
        subset["epoch"],
        lower,
        upper,
        alpha=0.15,
    )

ax.set_yscale("log")
ax.set_xlabel("Post-change training epoch")
ax.set_ylabel("Task B test MSE (log scale)")
ax.set_title(
    "Post-change learning trajectories across five seeds"
)
ax.legend()
ax.grid(alpha=0.2)

fig.tight_layout()

fig.savefig(
    "figures/post_change_learning.png",
    dpi=300,
)

plt.close(fig)


# ------------------------------------------------------------
# 10. Print compact summary
# ------------------------------------------------------------

print("\nPer-seed diagnostic summary:\n")

print(
    summary[
        [
            "seed",
            "continued_task_b_initial_mse",
            "continued_task_b_final_mse",
            "fresh_task_b_final_mse",
            "continued_early_log_slope",
            "fresh_early_log_slope",
            "continued_epochs_to_threshold",
            "fresh_epochs_to_threshold",
            "task_a_mse_before_change",
            "task_a_mse_after_adaptation",
        ]
    ].round(5).to_string(index=False)
)

print("\nMean diagnostic values across seeds:\n")

numeric_summary = (
    summary
    .drop(columns=["seed"])
    .mean(numeric_only=True)
    .round(5)
)

print(numeric_summary.to_string())

print(
    "\nSaved:"
    "\n  results/learning_trajectories.csv"
    "\n  results/seed_summary.csv"
    "\n  results/aggregate_trajectories.csv"
    "\n  figures/post_change_learning.png"
)