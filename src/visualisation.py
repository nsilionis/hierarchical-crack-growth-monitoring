import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

plt.rcParams.update({"text.usetex": True,
                     "font.family": "sans-serif",
                     "font.sans-serif": ["Computer Modern Serif"]})


def plot_scg_data(times, crack_lengths, save_fig_name=None):
    """
    Plotting function that plots the crack length
    as a function of time using the stochastic
    crack growth data.
    -----------
    Parameters
    ----------
    times : np.ndarray
        Array of time instants.
    crack_lengths : np.ndarray
        Array of crack lengths corresponding to the time instants.
    save_fig_name : str, optional
        If provided, the figure will be saved with this name.
        If None, the figure will not be saved.
    -----------
    Raises
    ------
    ValueError
        If the shapes of `times` and `crack_lengths` do not match.
    """

    if times.shape != crack_lengths.shape:
        raise ValueError("""The shapes of `times` and `crack_lengths`
                          must match.""")
    # Create figure and axis
    fig, ax = plt.subplots(1, 1)
    # Set colour map
    cmap = plt.get_cmap('tab20c')
    # Plot the crack lengths against time for each realisation
    for i in range(crack_lengths.shape[0]):
        # Find the index where padding (zeroes) begins.
        # Identify last non-zero element
        non_zero_indices = np.where(crack_lengths[i, :] > 0)[0]
        if len(non_zero_indices) > 0:
            # +1 to include the last non-zero element
            last_idx = non_zero_indices[-1] + 1
            # Plot only the non-padded part of the data
            ax.plot(times[i, :last_idx], crack_lengths[i, :last_idx],
                    color=cmap(i % 20), alpha=np.random.uniform(0.01, 0.1),
                    label=" ")
        else:
            # If the entire row is zeros, skip plotting
            continue
    # Set minor ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    # Set labels
    ax.set_xlabel(r'Time [years]')
    ax.set_ylabel(r'Crack length $\alpha$ [mm]')
    # Set x-axis to start from zero
    ax.set_xlim(left=0)
    # Add a dashed horizontal line at 155 mm
    ax.axhline(y=155, color=cmap(4), linestyle='--', alpha=0.8)
    # Add annotation for the critical crack length
    ax.annotate(r'$\alpha_{\mathrm{cr}} = 155 \ \mathrm{mm}$',
                xy=(0.1, 155),
                xytext=(0.1, 155 - 10)
                )
    # plt.legend(frameon=False)
    if save_fig_name is not None:
        # Get the root directory of the project
        dir_path = Path(__file__).resolve().parents[1]
        # Create the path to save the figure
        fname = dir_path / 'outputs' / save_fig_name
        # Raise an error if the directory does not exist
        if not fname.parent.exists():
            raise FileNotFoundError("""Directory {fname.parent}
                                    does not exist.""")
        # Save the figure
        plt.savefig(fname, bbox_inches='tight')
    plt.show()


def paris_params_pair_plot(paris_c, paris_m, save_fig_name=None):
    """
    Plot the Paris law parameters C and m
    using a pair plot.
    -----------
    Parameters
    ----------
    paris_c : np.ndarray
        Array of Paris law C coefficients.
    paris_m : np.ndarray
        Array of Paris law m coefficients.
    save_fig_name : str, optional
        If provided, the figure will be saved with this name.
        If None, the figure will not be saved.
    -----------
    Raises
    ------
    ValueError
        If the shapes of `paris_c` and `paris_m` do not match.
    """
    if paris_c.shape != paris_m.shape:
        raise ValueError("""The shapes of `paris_c` and `paris_m` \
                          must match.""")
    # Create DataFrame with the data
    df = pd.DataFrame({
        r'$\ln C$': np.log(paris_c).flatten(),
        r'$m$': paris_m.flatten()
    })
    # Calculate Pearson correlation coefficient
    corr_coef = df.corr().iloc[0, 1]
    # Create a custom PairGrid instead of using pairplot
    g = sns.PairGrid(df, diag_sharey=False, height=3, aspect=1)
    # Lower triangle: scatter plots with correlation text

    def scatter_with_corr(x, y, **kwargs):
        ax = plt.gca()
        ax.scatter(x, y, alpha=0.5, s=8., edgecolors="white", **kwargs)
        # Add correlation text in the plot
        ax.text(0.75, 0.95, f'$r = {corr_coef:.2f}$',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5,
                          edgecolor=None))
    g.map_lower(scatter_with_corr)
    # Upper triangle: KDE plots
    g.map_upper(sns.kdeplot, fill=True, levels=10, alpha=0.6, cmap="Blues")
    # Diagonal: KDE plots
    g.map_diag(sns.kdeplot, fill=True)
    # Tight layout
    plt.tight_layout()
    if save_fig_name is not None:
        # Get the root directory of the project
        dir_path = Path(__file__).resolve().parents[1]
        # Create the path to save the figure
        fname = dir_path / 'outputs' / save_fig_name
        # Raise an error if the directory does not exist
        if not fname.parent.exists():
            raise FileNotFoundError("""Directory {fname.parent}
                                    does not exist.""")
        # Save the figure
        plt.savefig(fname, bbox_inches='tight')
    # Show plot
    plt.show()
    return g
