import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from src.predictive_models import CrackGrowthPredictor

plt.rcParams.update({"text.usetex": True,
                     "font.family": "sans-serif",
                     "font.sans-serif": ["Computer Modern Serif"]})


def plot_scg_data(times, crack_lengths, save_fig_name=None):
    """
    Plotting function that plots the crack length
    as a function of time using the stochastic
    crack growth data.

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


def plot_initial_crack_length(initial_crack_length, save_fig_name=None):
    """
    Plot the initial crack length as a histogram.

    Parameters
    ----------
    initial_crack_length : np.ndarray
        Array of initial crack lengths.
    save_fig_name : str, optional
        If provided, the figure will be saved with this name.
        If None, the figure will not be saved.
    -----------
    Raises
    ------
    ValueError
        If `initial_crack_length` is not a 1D array.
    """
    if initial_crack_length.ndim != 1:
        raise ValueError("""`initial_crack_length` must be a 1D array.""")
    # Create figure and axis
    fig, ax = plt.subplots(1, 1)
    # Plot histogram
    cmap = plt.get_cmap('tab20c')
    sns.histplot(initial_crack_length, kde=True, ax=ax,
                 color=cmap(6), edgecolor='white', alpha=0.7,
                 stat='density')
    # Set ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    # Set labels
    ax.set_xlabel(r'$\alpha_0$ [mm]')
    ax.set_ylabel('Density')
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


def plot_avg_cycles(avg_cycles, save_fig_name=None):
    """
    Plot the average number of cycles per load realisation
    as a histogram.

    Parameters
    ----------
    avg_cycles : np.ndarray
        Array of average cycles per load realisation.
    save_fig_name : str, optional
        If provided, the figure will be saved with this name.
        If None, the figure will not be saved.
    -----------
    """
    # Create figure and axis
    fig, ax = plt.subplots(1, 1)
    # Plot histogram
    cmap = plt.get_cmap('tab20c')
    sns.histplot(np.mean(avg_cycles, axis=1), kde=True, ax=ax,
                 color=cmap(13), edgecolor='white', alpha=0.7,
                 stat='density')
    # Set ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    # Set labels
    ax.set_xlabel(r'$N_{\mathrm{avg}}$ [cycles/year]')
    ax.set_ylabel('Density')
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


def plot_stress_ranges(stress_ranges, save_fig_name=None):
    """
    Plot the stress ranges as a histogram.

    Parameters
    ----------
    stress_ranges : np.ndarray
        Array of stress ranges.
    save_fig_name : str, optional
        If provided, the figure will be saved with this name.
        If None, the figure will not be saved.
    -----------
    """
    # Create figure and axis
    fig, ax = plt.subplots(1, 1)
    # Plot histogram
    cmap = plt.get_cmap('tab20c')
    sns.histplot(stress_ranges[:, 0], kde=True, ax=ax,
                 color=cmap(4), edgecolor='white', alpha=0.7,
                 stat='density')
    # Set ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    # Set labels
    ax.set_xlabel(r'$\Delta S$ [MPa]')
    ax.set_ylabel('Density')
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


def plot_paris_predictions(paris_params, ds, navg, a0, times,
                           save_fig_name=None, figsize=(6, 4),
                           plot_individual=True, plot_grid=True):
    """
    Plot detailed Paris law predictions with various visualization options.

    Parameters
    ----------
    paris_params : tuple
        Tuple containing (C, m) Paris law parameters
    ds : float or array
        Stress range(s)
    navg : float or array
        Average number of cycles per time unit
    a0 : float or array
        Initial crack length(s)
    times : array
        Time points for prediction (can be 1D or 2D array)
    save_fig_name : str, optional
        Filename to save the figure
    figsize : tuple, optional
        Figure size
    plot_individual : bool, optional
        Whether to plot individual curves
    plot_grid : bool, optional
        Whether to show grid lines

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Create predictor
    predictor = CrackGrowthPredictor()

    # Convert paris_params to numpy arrays if needed
    C, m = paris_params
    if not isinstance(C, np.ndarray):
        C = np.array([C])
    if not isinstance(m, np.ndarray):
        m = np.array([m])

    # Ensure times is properly formatted
    if len(times.shape) > 1:
        # We have multiple time series
        time_array = times
    else:
        # We have a single time series - use it for all predictions
        time_array = np.tile(times, (len(C), 1))

    # Get predictions
    crack_lengths = []
    for i in range(len(C)):
        # Convert C to logC (natural logarithm)
        logc = np.log(C[i])

        # Use appropriate time array
        t_arr = time_array[i] if len(times.shape) > 1 else times

        # Get prediction
        cl = predictor.predict_crack_growth(
            logc=logc,
            m=m[i],
            ds=ds[i] if isinstance(ds, np.ndarray) else ds,
            navg=navg[i] if isinstance(navg, np.ndarray) else navg,
            a0=a0[i] if isinstance(a0, np.ndarray) else a0,
            times=t_arr
        )
        crack_lengths.append(cl)

    # Create figure and plot results
    fig, ax = plt.subplots(figsize=figsize)
    # Set colormap
    cmap = plt.get_cmap('tab20c')
    # Plot each prediction
    for i in range(len(crack_lengths)):
        t_arr = time_array[i] if len(times.shape) > 1 else times
        ax.plot(t_arr, crack_lengths[i], color=cmap(i),
                label=f"$\\alpha_{{{i+1}}}$")

    # Set labels and title
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Crack Length (mm)")
    # ax.set_title("Crack Growth Predictions")
    # Set x-axis to start from zero
    ax.set_xlim(left=0)
    # Set y-axis to start from zero and limit to 160 mm
    ax.set_ylim(bottom=0, top=160)
    # Add grid if requested
    if plot_grid:
        ax.grid(True, linestyle='--', alpha=0.4)

    # Add legend
    # Add legend in horizontal format at bottom left
    ax.legend(loc='lower center', ncol=min(5, len(crack_lengths)),
              frameon=True, framealpha=0.8, fontsize='small',
              bbox_to_anchor=(0.5, 0), borderaxespad=1)

    # Save figure if requested
    if save_fig_name:
        # Get the root directory of the project
        dir_path = Path(__file__).resolve().parents[1]
        # Create the path to save the figure
        save_path = dir_path / 'outputs' / save_fig_name
        # Raise an error if the directory does not exist
        if not save_path.parent.exists():
            raise FileNotFoundError(f"""Directory {save_path.parent}
                                    does not exist.""")
        plt.savefig(save_path, bbox_inches="tight")

    return fig, ax


def plot_parameter_sensitivity(base_c, base_m, base_ds, base_navg,
                               base_a0, times, c_variations=None,
                               m_variations=None, ds_variations=None,
                               figsize=(12, 4), cmap_name="Dark2",
                               save_fig_name=None):
    """
    Plot the sensitivity of crack growth to variations in Paris law parameters.

    Parameters
    ----------
    base_c : float
        Base value of Paris law parameter C
    base_m : float
        Base value of Paris law exponent m
    base_ds : float
        Base value of stress range
    base_navg : float
        Base value of average cycles per time unit
    base_a0 : float
        Base value of initial crack length
    times : array
        Time points for prediction
    c_variations : list, optional
        List of C parameter values to test
        (default is [0.5*base_c, base_c, 1.5*base_c])
    m_variations : list, optional
        List of m parameter values to test
        (default is [0.8*base_m, base_m, 1.2*base_m])
    ds_variations : list, optional
        List of stress range values to test
        (default is [0.7*base_ds, base_ds, 1.3*base_ds])
    figsize : tuple, optional
        Figure size
    cmap_name : str, optional
        Name of colormap to use
    save_fig_name : str, optional
        Filename to save the figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : numpy.ndarray
        Array of axes objects
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from src.predictive_models import CrackGrowthPredictor

    # Set default variations if not provided
    if c_variations is None:
        c_variations = [base_c * 0.5, base_c, base_c * 1.5]
    if m_variations is None:
        m_variations = [base_m * 0.8, base_m, base_m * 1.2]
    if ds_variations is None:
        ds_variations = [base_ds * 0.7, base_ds, base_ds * 1.3]

    # Create predictor
    predictor = CrackGrowthPredictor()

    # Plot parameter sensitivity
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    cmap = plt.get_cmap(cmap_name)

    # Effect of C
    for i, c in enumerate(c_variations):
        cl = predictor.predict_crack_growth(
            logc=np.log(c),
            m=base_m,
            ds=base_ds,
            navg=base_navg,
            a0=base_a0,
            times=times
        )
        axes[0].plot(times, cl, color=cmap(i), label=f"$\\ln C = \
                     {{{np.log(c):.2f}}}$")
    axes[0].set_title("Effect of rate parameter $C$")
    axes[0].set_xlabel("Time (years)")
    axes[0].set_ylabel("Crack Length (mm)")
    axes[0].grid(True, linestyle='--', alpha=0.4)
    axes[0].legend()

    # Effect of m
    for i, m in enumerate(m_variations):
        cl = predictor.predict_crack_growth(
            logc=np.log(base_c),
            m=m,
            ds=base_ds,
            navg=base_navg,
            a0=base_a0,
            times=times
        )
        axes[1].plot(times, cl, color=cmap(i), label=f"$m = {m:.2f}$")
    axes[1].set_title("Effect of exponent parameter $m$")
    axes[1].set_xlabel("Time (years)")
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(bottom=0, top=160)
    axes[1].grid(True, linestyle='--', alpha=0.4)
    axes[1].legend()

    # Effect of stress range
    for i, ds in enumerate(ds_variations):
        cl = predictor.predict_crack_growth(
            logc=np.log(base_c),
            m=base_m,
            ds=ds,
            navg=base_navg,
            a0=base_a0,
            times=times
        )
        axes[2].plot(times, cl, color=cmap(i), label=f"$\\Delta S = {ds:.2f}$")
    axes[2].set_title("Effect of stress range $\\Delta S$")
    axes[2].set_xlabel("Time (years)")
    axes[2].grid(True, linestyle='--', alpha=0.4)
    axes[2].legend()

    plt.tight_layout()

    # Save figure if requested
    if save_fig_name:
        # Get the root directory of the project
        dir_path = Path(__file__).resolve().parents[1]
        # Create the path to save the figure
        save_path = dir_path / 'outputs' / save_fig_name
        # Raise an error if the directory does not exist
        if not save_path.parent.exists():
            raise FileNotFoundError(f"""Directory {save_path.parent}
                                    does not exist.""")
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes
