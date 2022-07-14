import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import tqdm

import sys
sys.path.append("../")
sys.path.insert(1, os.path.join(sys.path[0], '../'))
import core


def get_lambdas(scores, max_lambdas=10000):
    """Precompute table of threshold losses/sizes."""
    # Since risk is monotone, can just use lambda grid based on all scores.
    lambdas = np.unique(scores)
    np.random.shuffle(lambdas)
    lambdas = lambdas[:max_lambdas]
    indices = np.argsort(-lambdas)
    lambdas = lambdas[indices]
    return lambdas


def get_loss_and_size_tables(scores, losses, lambdas):
    """Precompute table of threshold losses/sizes."""
    loss_table = np.zeros((len(scores), len(lambdas)))
    size_table = np.zeros((len(scores), len(lambdas)))
    for i in tqdm.tqdm(range(len(lambdas)), desc="precomputing sizes and losses"):
        # Take index of lowest score less than this lambda.
        max_index = np.sum(scores <= lambdas[i], axis=1) - 1
        size_table[:, i] = max_index
        loss_table[:, i] = np.take_along_axis(losses, np.reshape(max_index, (-1, 1)), axis=1).squeeze()

    return loss_table, size_table


def compute_trial(loss_table, size_table, lambdas, alpha, num_calibration):
    """Compute risk and sizes for a random trial.

    Args:
        loss_table: [num_examples, num_lambdas] losses by lambda from small to large.
        size_table: [num_examples, num_lambdas] sizes by lambda from small to large.
        lambdas: [num_lambdas] lambda values from small to large.
        alpha: Target loss.
        num_calibration: Number of calibration points to use.

    Returns:
        lhat: Confidence score threshold.
        avg_loss: Average set loss.
        avg_size: Average set size.
    """
    # Split to calibration and test.
    perm = np.random.permutation(len(loss_table))
    loss_table, size_table = loss_table[perm], size_table[perm]
    calib_loss_table = loss_table[:num_calibration]
    valid_loss_table = loss_table[num_calibration:]
    valid_size_table = size_table[num_calibration:]

    # Compute threshold.
    lhat = core.get_lhat(calib_loss_table, lambdas, alpha)

    # Compute losses and size.
    avg_loss = valid_loss_table[:, np.argmax(lambdas == lhat)].mean()
    avg_size = valid_size_table[:, np.argmax(lambdas == lhat)].mean()

    return lhat, avg_loss, avg_size


def run_trials(scores, losses, alpha, num_trials, num_calibration=3000, max_lambdas=10000):
    lambdas = get_lambdas(scores, max_lambdas)
    loss_table, size_table = get_loss_and_size_tables(scores, losses, lambdas)

    results = []
    all_risk = []
    all_lhat = []
    for i in tqdm.tqdm(range(num_trials), desc="Running trials"):
        lhat, risk, size = compute_trial(loss_table, size_table, lambdas, alpha, num_calibration)
        results.append(pd.DataFrame(dict(risk=risk, size=size, alpha=alpha), index=[i]))
        all_risk.append(risk)
        all_lhat.append(lhat)
    df = pd.concat(results, axis=0, ignore_index=True)
    print("Average risk across trials = %2.3f" % np.mean(all_risk))
    print("Average threshold across trials = %2.3f" % np.mean(all_lhat))
    return df


def plot_histograms(df, alpha, output_dir):
    """Plot histogram of trial risks at tolerance alpha.

    Args:
        df: DataFrame with experiment data with columns "sizes" and "risk".
        alpha: Target risk tolerance.
        output_dir: Where to save histogram plot (as {alpha}_qa_histograms.pdf).
    """
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))
    risk = df["risk"].to_numpy()
    axs[0].hist(risk, bins=15, alpha=0.7, density=True)

    size = df["size"].to_numpy()
    axs[1].hist(size, bins=15, alpha=0.7, density=True)

    axs[0].set_xlabel("risk")
    axs[0].locator_params(axis="x", nbins=10)
    axs[0].axvline(x=alpha, c="#999999", linestyle="--", alpha=0.7)
    axs[0].set_ylabel("density")
    axs[1].set_xlabel("set size")
    axs[1].locator_params(axis="x", nbins=10)
    axs[1].set_yscale("log")
    sns.despine(top=True, right=True, ax=axs[0])
    sns.despine(top=True, right=True, ax=axs[1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{alpha}_qa_histograms.pdf"))


def main(args):
    conf_scores, f1_scores = np.load(args.input_file)

    # Add empty set.
    conf_scores = np.pad(conf_scores, ((0, 0), (1, 0)), constant_values=1e8)
    f1_scores = np.pad(f1_scores, ((0, 0), (1, 0)), constant_values=0)

    if len(conf_scores) <= args.num_calibration:
        raise ValueError("Num calibration greater than num points! " +
                         f"({args.num_calibration} >= {len(conf_scores)}")

    df = run_trials(
        scores=-conf_scores,
        losses=1 - f1_scores,
        alpha=args.alpha,
        num_trials=args.num_trials,
        num_calibration=args.num_calibration,
        max_lambdas=args.max_lambdas)

    plot_histograms(df, args.alpha, args.output_dir)


if __name__ == "__main__":
    sns.set(palette="pastel", font="serif")
    sns.set_style("white")
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="results/processed.npy")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--num_trials", type=int, default=1000)
    parser.add_argument("--num_calibration", type=int, default=2500)
    parser.add_argument("--max_lambdas", type=int, default=5000)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    main(args)
