"""Plotting functions for calibration and full reports."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def plot_pca_magistrales(E: np.ndarray, labels: np.ndarray, output_path: str,
                         title: str = "PCA - Magistrales") -> str:
    """PCA scatter colored by magistrale assignment."""
    pca = PCA(n_components=2, random_state=42)
    E_2d = pca.fit_transform(E[:min(len(E), 10000)])
    lab = labels[:min(len(labels), 10000)]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(E_2d[:, 0], E_2d[:, 1], c=lab, cmap="tab10",
                         alpha=0.4, s=3, rasterized=True)
    plt.colorbar(scatter, ax=ax, label="Magistrale")
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved PCA plot: {output_path}")
    return output_path


def plot_residual_histogram(residuals: np.ndarray, tau: float, output_path: str,
                            title: str = "Lattice Residual Histogram") -> str:
    """Histogram of lattice residuals with tau line."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residuals, bins=100, density=True, alpha=0.7, color="steelblue", edgecolor="none")
    ax.axvline(tau, color="red", linestyle="--", linewidth=2, label=f"tau={tau}")
    aligned_frac = float(np.mean(residuals < tau))
    ax.set_title(f"{title} (aligned: {aligned_frac:.1%})")
    ax.set_xlabel("Residual (log-space)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_wave_lengths(wave_lengths: List[int], output_path: str,
                      title: str = "Wave Length Distribution") -> str:
    """Histogram of wave lengths."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if wave_lengths:
        ax.hist(wave_lengths, bins=max(1, len(set(wave_lengths))),
                alpha=0.7, color="teal", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Wave Length (bins)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_temporal_scores(bin_centers: List[float], scores: List[float],
                         output_path: str, title: str = "S(t) Temporal Scores") -> str:
    """Plot S(t) scores over time."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(bin_centers, scores, color="steelblue", linewidth=1.5, alpha=0.8)
    ax.fill_between(bin_centers, 0, scores, alpha=0.2, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Time (pseudo-index)")
    ax.set_ylabel("Alignment Score S(t)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_band_predictions(predictions: List[Dict], output_path: str,
                          title: str = "Band Predictions (E6)") -> str:
    """Plot predicted vs actual band centers."""
    valid = [p for p in predictions if p["status"] == "ok"]
    if not valid:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No valid predictions", ha="center", va="center", fontsize=14)
        ax.set_title(title)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path
    
    fig, ax = plt.subplots(figsize=(8, 6))
    mags = [p["magistrale"] for p in valid]
    predicted = [p["predicted_center"] for p in valid]
    actual = [p["actual_center"] for p in valid]
    
    ax.scatter(actual, predicted, c="steelblue", s=60, zorder=5, label="Predictions")
    lims = [min(min(actual), min(predicted)), max(max(actual), max(predicted))]
    ax.plot(lims, lims, "k--", alpha=0.5, label="Perfect")
    
    for m, p, a in zip(mags, predicted, actual):
        ax.annotate(f"M{m}", (a, p), fontsize=8, alpha=0.7)
    
    ax.set_xlabel("Actual Band Center")
    ax.set_ylabel("Predicted Band Center")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_fsi_breakdown(fsi_data: Dict, output_path: str,
                       title: str = "FSI Breakdown") -> str:
    """Bar chart of FSI components."""
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Signal Rate", "Surprise (Z)", "Clustering\n(1-Entropy)"]
    values = [fsi_data["signal_rate_component"],
              fsi_data["surprise_component"],
              fsi_data["entropy_component"]]
    colors = ["steelblue", "coral", "teal"]
    
    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor="white")
    ax.axhline(fsi_data["fsi"], color="black", linestyle="--", label=f"FSI={fsi_data['fsi']:.3f}")
    ax.set_title(title)
    ax.set_ylabel("Component Value")
    ax.legend()
    ax.set_ylim(0, max(0.5, max(values) * 1.3))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_perturbation_curve(alphas: List[float], scores: List[float],
                            tipping_alpha: Optional[float], output_path: str,
                            title: str = "Perturbation Curve") -> str:
    """Plot score vs perturbation alpha."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(alphas, scores, "o-", color="steelblue", linewidth=2, markersize=6)
    if tipping_alpha is not None:
        ax.axvline(tipping_alpha, color="red", linestyle="--", label=f"Tipping alpha*={tipping_alpha:.2f}")
    ax.set_title(title)
    ax.set_xlabel("Perturbation Alpha")
    ax.set_ylabel("Alignment Score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_null_pvalues(null_results: Dict, output_path: str,
                      title: str = "P-values per Null Model") -> str:
    """Bar chart of p-values per null model."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    null_names = ["uniform", "gaussian_cov", "spacing"]
    p_values = []
    z_scores = []
    for name in null_names:
        if name in null_results:
            p_values.append(null_results[name]["p_value"])
            z_scores.append(null_results[name]["z_score"])
        else:
            p_values.append(1.0)
            z_scores.append(0.0)
    
    x = np.arange(len(null_names))
    bars = ax.bar(x, p_values, alpha=0.8, color=["steelblue", "coral", "teal"], edgecolor="white")
    ax.axhline(0.05, color="red", linestyle="--", label="alpha=0.05")
    ax.set_xticks(x)
    ax.set_xticklabels(["Uniform", "Gaussian\nCov", "Spacing"])
    ax.set_ylabel("p-value")
    ax.set_title(title)
    ax.legend()
    
    for i, (p, z) in enumerate(zip(p_values, z_scores)):
        ax.text(i, p + 0.02, f"z={z:.1f}", ha="center", fontsize=9)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_echo_vs_fresh(fresh_data: Dict, echo_data: Dict, output_path: str) -> str:
    """Comparison plot of echo vs fresh metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ["alignment", "fsi", "median_error"]
    labels_list = ["Global Alignment", "FSI", "Prediction Error"]
    
    for ax, metric, label in zip(axes, metrics, labels_list):
        fresh_val = fresh_data.get(metric, 0)
        echo_val = echo_data.get(metric, 0)
        
        bars = ax.bar(["Fresh", "Echo"], [fresh_val, echo_val],
                      color=["steelblue", "coral"], alpha=0.8, edgecolor="white")
        
        if fresh_val > 0:
            ratio = echo_val / fresh_val
            ax.set_title(f"{label}\nratio={ratio:.3f}")
        else:
            ax.set_title(label)
        ax.set_ylabel("Value")
    
    fig.suptitle("Echo vs Fresh Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
