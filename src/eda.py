"""
eda.py
------
Reusable EDA plotting functions for the Student Placement & Salary
Prediction System. Kept separate from the notebook so the same functions
can be reused/imported elsewhere (e.g. for generating report figures).
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")


# ---------------------------------------------------------------------
# Univariate
# ---------------------------------------------------------------------

def plot_univariate_numeric(df: pd.DataFrame, save_path: str = None):
    """Histograms + KDE for key numeric columns."""
    cols = ["cgpa", "iq_score", "internships", "projects", "backlogs"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color="#4C72B0")
        axes[i].set_title(f"Distribution of {col}")
    fig.delaxes(axes[-1])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=110, bbox_inches="tight")
    return fig


def plot_package_distribution(df: pd.DataFrame, save_path: str = None):
    """Package distribution - placed students only."""
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(df[df["placement_status"] == "Placed"]["package_lpa"],
                 kde=True, ax=ax, color="#55A868")
    ax.set_title("Package (LPA) Distribution - Placed Students Only")
    ax.set_xlabel("Package (LPA)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=110, bbox_inches="tight")
    return fig


def plot_placement_countplot(df: pd.DataFrame, save_path: str = None):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.countplot(x="placement_status", hue="placement_status", data=df, ax=ax, palette=["#C44E52", "#55A868"], legend=False)
    ax.set_title("Placement Status Count")
    for container in ax.containers:
        ax.bar_label(container)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=110, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------
# Bivariate
# ---------------------------------------------------------------------

def plot_bivariate_box(df: pd.DataFrame, save_path: str = None):
    """Boxplots: numeric feature vs placement_status."""
    cols = ["cgpa", "iq_score", "internships", "projects",
             "backlogs", "communication_score"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.boxplot(x="placement_status", y=col, hue="placement_status", data=df, ax=axes[i], legend=False,
                    palette=["#C44E52", "#55A868"])
        axes[i].set_title(f"{col} vs Placement")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=110, bbox_inches="tight")
    return fig


def plot_cgpa_vs_package(df: pd.DataFrame, save_path: str = None):
    fig, ax = plt.subplots(figsize=(7, 5))
    placed = df[df["placement_status"] == "Placed"]
    sns.scatterplot(x="cgpa", y="package_lpa", hue="branch", data=placed, ax=ax, alpha=0.6)
    ax.set_title("CGPA vs Package (Placed Students)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=110, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------
# Multivariate
# ---------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = None):
    numeric_cols = df.select_dtypes(include="number").columns
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=110, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    import os
    os.makedirs("reports", exist_ok=True)
    df = pd.read_csv("data/placement_data_cleaned.csv")

    plot_univariate_numeric(df, "reports/univariate_numeric.png")
    plot_package_distribution(df, "reports/package_distribution.png")
    plot_placement_countplot(df, "reports/placement_countplot.png")
    plot_bivariate_box(df, "reports/bivariate_box.png")
    plot_cgpa_vs_package(df, "reports/cgpa_vs_package.png")
    plot_correlation_heatmap(df, "reports/correlation_heatmap.png")

    print("All EDA plots saved to reports/")
