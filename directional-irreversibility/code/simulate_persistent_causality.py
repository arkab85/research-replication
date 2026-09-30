from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIOS = {
    "direct": "Direct employment to credit transmission",
    "feedback": "Bidirectional employment credit feedback",
    "confounded": "Persistent common confounder without direct transmission",
}
HORIZONS = [1, 3, 6]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def simulate_scenario(scenario: str, n_households: int, periods: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for household_id in range(n_households):
        risk = rng.normal(0.0, 0.45)
        e = np.zeros(periods)
        c = np.zeros(periods)
        h = np.zeros(periods)
        soft = np.zeros(periods)
        common = np.zeros(periods)
        latent = np.zeros(periods)
        e[0] = rng.normal(0, 0.7)
        c[0] = 0.15 * risk + rng.normal(0, 0.7)
        h[0] = 0.10 * risk + rng.normal(0, 0.6)
        for t in range(1, periods):
            if scenario == "direct":
                common[t] = 0.55 * common[t - 1] + rng.normal(0, 0.55)
                e[t] = 0.65 * e[t - 1] + 0.20 * common[t] + rng.normal(0, 0.75)
                c[t] = 0.85 * c[t - 1] + 0.45 * e[t - 1] + 0.10 * common[t] + rng.normal(0, 0.75)
            elif scenario == "feedback":
                common[t] = 0.55 * common[t - 1] + rng.normal(0, 0.55)
                e[t] = 0.55 * e[t - 1] + 0.20 * common[t] + 0.20 * c[t - 1] + rng.normal(0, 0.75)
                c[t] = 0.80 * c[t - 1] + 0.45 * e[t - 1] + 0.10 * common[t] + rng.normal(0, 0.75)
            elif scenario == "confounded":
                latent[t] = 0.97 * latent[t - 1] + rng.normal(0, 0.55)
                e[t] = 0.20 * e[t - 1] + 0.85 * latent[t] + rng.normal(0, 0.65)
                # There is no e term here: latent[t-1] is the only common driver.
                c[t] = 0.92 * c[t - 1] + 0.33 * latent[t - 1] + rng.normal(0, 0.70)
            else:
                raise ValueError(f"Unknown scenario {scenario}")

            h[t] = 0.90 * h[t - 1] + 0.25 * c[t - 1] + 0.06 * e[t - 1] + 0.10 * common[t] + rng.normal(0, 0.55)

        # A soft-information signal is observed at t and may forecast the next outcome,
        # but it is not allowed to use future outcomes in the measurement equation.
        soft = 0.45 * e + 0.40 * c + 0.12 * h + rng.normal(0, 0.90, periods)
        delinquency_prob = sigmoid(-3.05 + 0.78 * c + 0.20 * h + 0.40 * np.r_[0.0, soft[:-1]] + 0.25 * risk)
        delinquency = rng.binomial(1, delinquency_prob)
        credit_score = 720 - 52 * c - 6 * risk + rng.normal(0, 12, periods)
        utilization = sigmoid(-0.25 + 0.55 * c + 0.15 * risk + rng.normal(0, 0.15, periods))
        home_value = 250_000 * np.exp(0.004 * np.arange(periods) - 0.035 * h + rng.normal(0, 0.035, periods))
        move_prob = sigmoid(-3.1 + 0.18 * e - 0.35 * h - 0.20 * c + rng.normal(0, 0.10, periods))
        moved = rng.binomial(1, move_prob)
        for t in range(periods):
            rows.append(
                {
                    "scenario": scenario,
                    "household_id": household_id,
                    "time_index": t,
                    "date": (pd.Timestamp("2017-01-01") + pd.offsets.MonthBegin(t)).strftime("%Y-%m-%d"),
                    "employment_stress": e[t],
                    "credit_stress": c[t],
                    "housing_stress": h[t],
                    "soft_info": soft[t],
                    "credit_score": credit_score[t],
                    "utilization": utilization[t],
                    "home_value": home_value[t],
                    "delinquency": delinquency[t],
                    "moved": moved[t],
                }
            )
    return pd.DataFrame(rows)


def design_matrix(frame: pd.DataFrame, x: str, y: str, horizon: int, lags: int = 3) -> pd.DataFrame:
    work = frame.sort_values(["household_id", "time_index"]).copy()
    grouped = work.groupby("household_id", sort=False)
    for lag in range(1, lags + 1):
        work[f"x_lag{lag}"] = grouped[x].shift(lag)
        work[f"y_lag{lag}"] = grouped[y].shift(lag)
    work["target"] = grouped[y].shift(-horizon)
    work = work.dropna(subset=["target"] + [f"x_lag{lag}" for lag in range(1, lags + 1)] + [f"y_lag{lag}" for lag in range(1, lags + 1)])
    return work


def fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    x_train_aug = np.column_stack([np.ones(len(x_train)), x_train])
    x_test_aug = np.column_stack([np.ones(len(x_test)), x_test])
    beta, *_ = np.linalg.lstsq(x_train_aug, y_train, rcond=None)
    return x_test_aug @ beta


def bootstrap_ci(values: np.ndarray, seed: int, reps: int = 250) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    draws = rng.integers(0, len(values), size=(reps, len(values)))
    means = values[draws].mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def directional_result(frame: pd.DataFrame, scenario: str, horizon: int, seed: int) -> dict[str, float | str | int]:
    forward = design_matrix(frame, "employment_stress", "credit_stress", horizon)
    reverse = design_matrix(frame, "credit_stress", "employment_stress", horizon)
    results: dict[str, float | str | int] = {"scenario": scenario, "horizon": horizon}

    def estimate(work: pd.DataFrame, prefix: str) -> tuple[float, np.ndarray]:
        base_cols = [f"y_lag{lag}" for lag in range(1, 4)]
        full_cols = base_cols + [f"x_lag{lag}" for lag in range(1, 4)]
        train = work[work["time_index"] <= 29]
        test = work[work["time_index"] >= 30]
        y_train = train["target"].to_numpy()
        y_test = test["target"].to_numpy()
        pred_base = fit_predict(train[base_cols].to_numpy(), y_train, test[base_cols].to_numpy())
        pred_full = fit_predict(train[full_cols].to_numpy(), y_train, test[full_cols].to_numpy())
        denominator = float(np.var(y_test))
        se_base = (y_test - pred_base) ** 2
        se_full = (y_test - pred_full) ** 2
        gain = float(np.mean(se_base - se_full) / denominator)
        per_household = pd.DataFrame({"household_id": test["household_id"].to_numpy(), "diff": se_base - se_full}).groupby("household_id")["diff"].mean().to_numpy() / denominator
        results[f"{prefix}_gain"] = gain
        return gain, per_household

    forward_gain, forward_hh = estimate(forward, "forward")
    reverse_gain, reverse_hh = estimate(reverse, "reverse")
    n = min(len(forward_hh), len(reverse_hh))
    asymmetry_by_household = forward_hh[:n] - reverse_hh[:n]
    ci_low, ci_high = bootstrap_ci(asymmetry_by_household, seed + horizon)
    results["asymmetry"] = forward_gain - reverse_gain
    results["ci_low"] = ci_low
    results["ci_high"] = ci_high
    return results


def soft_information_result(frame: pd.DataFrame, scenario: str, seed: int) -> dict[str, float | str]:
    work = frame.sort_values(["household_id", "time_index"]).copy()
    grouped = work.groupby("household_id", sort=False)
    for lag in [1, 2, 3]:
        work[f"c_lag{lag}"] = grouped["credit_stress"].shift(lag)
        work[f"h_lag{lag}"] = grouped["housing_stress"].shift(lag)
    work["target"] = grouped["delinquency"].shift(-1)
    work = work.dropna(subset=["target", "soft_info", "c_lag1", "c_lag2", "c_lag3", "h_lag1", "h_lag2", "h_lag3"])
    train = work[work["time_index"] <= 29]
    test = work[work["time_index"] >= 30]
    base = [f"c_lag{lag}" for lag in [1, 2, 3]] + [f"h_lag{lag}" for lag in [1, 2, 3]]
    full = base + ["soft_info"]
    pred_base = fit_predict(train[base].to_numpy(), train["target"].to_numpy(), test[base].to_numpy())
    pred_full = fit_predict(train[full].to_numpy(), train["target"].to_numpy(), test[full].to_numpy())
    var = float(np.var(test["target"].to_numpy()))
    return {
        "scenario": scenario,
        "baseline_brier": float(np.mean((test["target"].to_numpy() - pred_base) ** 2)),
        "soft_info_brier": float(np.mean((test["target"].to_numpy() - pred_full) ** 2)),
        "soft_info_gain": float((np.mean((test["target"].to_numpy() - pred_base) ** 2) - np.mean((test["target"].to_numpy() - pred_full) ** 2)) / var),
    }


def make_figures(panel: pd.DataFrame, direction: pd.DataFrame, soft: pd.DataFrame, out_dir: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.3, 4.1))
    direct = panel[panel["scenario"] == "direct"].groupby("time_index")[["employment_stress", "credit_stress", "housing_stress"]].mean()
    for col, label, color in [
        ("employment_stress", "Employment stress", "#1f4e79"),
        ("credit_stress", "Credit stress", "#c55a11"),
        ("housing_stress", "Housing stress", "#548235"),
    ]:
        series = (direct[col] - direct[col].mean()) / direct[col].std()
        ax.plot(series.index, series.values, linewidth=2, label=label, color=color)
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_title("Synthetic dynamic household finance panel")
    ax.set_xlabel("Month index")
    ax.set_ylabel("Standardized mean stress")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(out_dir / "figure_dynamic_path.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.3, 4.2))
    for scenario, label, color in [
        ("direct", "Direct transmission", "#1f4e79"),
        ("feedback", "Bidirectional feedback", "#c55a11"),
        ("confounded", "Persistent confounder", "#a61c00"),
    ]:
        d = direction[direction["scenario"] == scenario].sort_values("horizon")
        ax.plot(d["horizon"], d["asymmetry"], marker="o", linewidth=2, label=label, color=color)
        ax.fill_between(d["horizon"].to_numpy(), d["ci_low"].to_numpy(), d["ci_high"].to_numpy(), alpha=0.14, color=color)
    ax.axhline(0, color="#444444", linewidth=0.9)
    ax.set_title("Out-of-sample directional asymmetry by horizon")
    ax.set_xlabel("Forecast horizon h")
    ax.set_ylabel("Forward gain minus reverse gain")
    ax.set_xticks(HORIZONS)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figure_directional_asymmetry.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    plot = soft.copy()
    ax.bar(plot["scenario"].map({"direct": "Direct", "feedback": "Feedback", "confounded": "Confounded"}), plot["soft_info_gain"], color=["#1f4e79", "#c55a11", "#a61c00"])
    ax.axhline(0, color="#444444", linewidth=0.9)
    ax.set_title("Incremental soft-information gain for next-month delinquency")
    ax.set_ylabel("Normalized out-of-sample Brier gain")
    fig.tight_layout()
    fig.savefig(out_dir / "figure_soft_information.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=".")
    parser.add_argument("--n-households", type=int, default=2500)
    parser.add_argument("--periods", type=int, default=48)
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = []
    direction_rows = []
    soft_rows = []
    for idx, scenario in enumerate(SCENARIOS):
        panel = simulate_scenario(scenario, args.n_households, args.periods, seed=20260914 + idx * 17)
        panels.append(panel)
        for horizon in HORIZONS:
            direction_rows.append(directional_result(panel, scenario, horizon, seed=1200 + idx * 10))
        soft_rows.append(soft_information_result(panel, scenario, seed=2200 + idx))
    full_panel = pd.concat(panels, ignore_index=True)
    direction = pd.DataFrame(direction_rows)
    soft = pd.DataFrame(soft_rows)
    full_panel.to_csv(out_dir / "synthetic_panel.csv", index=False)
    direction.to_csv(out_dir / "directional_results.csv", index=False)
    soft.to_csv(out_dir / "soft_information_results.csv", index=False)
    pd.DataFrame(
        [
            {"scenario": key, "description": value}
            for key, value in SCENARIOS.items()
        ]
    ).to_csv(out_dir / "scenario_definitions.csv", index=False)
    make_figures(full_panel, direction, soft, out_dir)
    print(direction.to_string(index=False))
    print(soft.to_string(index=False))


if __name__ == "__main__":
    main()
