"""Creates heatmap and contour plots, saves to file"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def save_heatmap(
    df: pd.DataFrame,
    index: str,
    column: str,
    values: str,
    file_parts: tuple[str, str],
) -> None:
    """
    Process a dataframe into a heatmap that is saved to file

    Parameters
    ----------
    df: pd.DataFrame
        Contains dataframe to be plotted
    index: str
        dataframe column to represent index
    column: str
        dataframe column to represent column
    values: str
        dataframe column to represent values
    file_parts: tuple[str,str]
        prefix and suffix for '_heatmap_'
    """
    df = df.pivot(index=index, columns=column, values=values)

    fig = plt.figure(figsize=(8, 6))
    ax = sns.heatmap(df, annot=True, cmap="coolwarm_r")
    ax.invert_yaxis()
    plt.title("Heatmap")
    plt.xlabel("chain_strength")
    plt.ylabel("cost_constraint_ratio")
    plt.savefig(
        os.path.join("data", file_parts[0] + "_heatmap_" + file_parts[1]),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_contour_plot(
    df: pd.DataFrame,
    index: str,
    column: str,
    values: str,
    file_parts: tuple[str, str],
) -> None:
    """
    Process a dataframe into a contour plot that is saved to file

    Parameters
    ----------
    df: pd.DataFrame
        Contains dataframe to be plotted
    index: str
        dataframe column to represent index
    column: str
        dataframe column to represent column
    values: str
        dataframe column to represent values
    file_parts: tuple[str,str]
        prefix and suffix for '_heatmap_'
    """
    df = df.pivot(index=index, columns=column, values=values)

    x = df.columns.values
    y = df.index.values
    x, y = np.meshgrid(x, y)
    z = df.values

    fig = plt.figure(figsize=(8, 6))
    contour = plt.contourf(x, y, z, levels=10, cmap="coolwarm_r")
    plt.colorbar(contour)
    plt.title("Contour Plot")
    plt.xlabel("chain_strength")
    plt.ylabel("cost_constraint_ratio")
    plt.savefig(
        os.path.join("data", file_parts[0] + "_contours_" + file_parts[1]),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
