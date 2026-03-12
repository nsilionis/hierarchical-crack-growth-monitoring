import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
import numpy as np
import jax
from pathlib import Path
import arviz as az
from scipy import stats
from src.predictive_models import CrackGrowthPredictor
from src.crack_growth_models import VariableStressParisErdogan

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
        fname = dir_path / 'figures' / save_fig_name
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
        fname = dir_path / 'figures' / save_fig_name
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
        fname = dir_path / 'figures' / save_fig_name
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
        fname = dir_path / 'figures' / save_fig_name
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
        fname = dir_path / 'figures' / save_fig_name
        # Raise an error if the directory does not exist
        if not fname.parent.exists():
            raise FileNotFoundError("""Directory {fname.parent}
                                    does not exist.""")
        # Save the figure
        plt.savefig(fname, bbox_inches='tight')
    plt.show()


def plot_paris_predictions(paris_params, ds, navg, a0, times,
                           save_fig_name=None, figsize=(6, 4),
                           plot_individual=True, plot_grid=True,
                           cmap_name='tab20c'):
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
    cmap_name : str, optional
        Name of colormap to use for individual curves

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
    cmap = plt.get_cmap(cmap_name)
    # Plot each prediction
    for i in range(len(crack_lengths)):
        t_arr = time_array[i] if len(times.shape) > 1 else times
        ax.plot(t_arr, crack_lengths[i], color=cmap(i),
                label=f"Component {{{i+1}}}")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in', top=False, right=False)
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
        ax.grid(True, linestyle='--', alpha=0.25)

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
        save_path = dir_path / 'figures' / save_fig_name
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
    axes[0].xaxis.set_minor_locator(AutoMinorLocator())
    axes[0].yaxis.set_minor_locator(AutoMinorLocator())
    axes[0].set_xlabel("Time (years)")
    axes[0].set_ylabel("Crack Length (mm)")
    axes[0].set_title("Effect of rate parameter $C$")
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
    axes[1].xaxis.set_minor_locator(AutoMinorLocator())
    axes[1].yaxis.set_minor_locator(AutoMinorLocator())
    axes[1].set_xlabel("Time (years)")
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(bottom=0, top=160)
    axes[1].set_title("Effect of exponent parameter $m$")
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
    axes[2].xaxis.set_minor_locator(AutoMinorLocator())
    axes[2].yaxis.set_minor_locator(AutoMinorLocator())
    axes[2].set_xlabel("Time (years)")
    axes[2].set_xlim(left=0)
    axes[2].set_ylim(bottom=0, top=160)
    axes[2].set_title("Effect of stress range $\\Delta S$")
    axes[2].grid(True, linestyle='--', alpha=0.4)
    axes[2].legend()

    plt.tight_layout()

    # Save figure if requested
    if save_fig_name:
        # Get the root directory of the project
        dir_path = Path(__file__).resolve().parents[1]
        # Create the path to save the figure
        save_path = dir_path / 'figures' / save_fig_name
        # Raise an error if the directory does not exist
        if not save_path.parent.exists():
            raise FileNotFoundError(f"""Directory {save_path.parent}
                                    does not exist.""")
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes


def plot_variable_stress_comparison(stress_periods=None, ds_array=None,
                                    times=None,  logc=None, m=None,
                                    navg=None, a0=None, cmap_name='Dark2',
                                    figsize=(10, 6), save_fig_name=None):
    """
    Plot a comparison between constant and
    variable stress Paris-Erdogan models.

    Parameters
    ----------
    stress_periods : list of tuples, optional
        List of (start_time, end_time, stress_level)
        tuples defining stress periods.
        Required if ds_array is not provided.
    ds_array : array, optional
        Array of stress values for each time interval.
        Required if stress_periods is not provided.
    times : array, optional
        Time points for prediction. If None, creates a
        default array from 0 to 3 years.
    logc : float, optional
        Natural logarithm of Paris law parameter C. Default is ln(5e-14).
    m : float, optional
        Paris law exponent. Default is 3.3.
    navg : float, optional
        Average cycles per time unit. Default is 2.8e6 (cycles/year).
    a0 : float, optional
        Initial crack length. Default is 40.0 mm.
    cmap_name : str, optional
        Name of colormap to use. Default is 'Dark2'.
    figsize : tuple, optional
        Figure size. Default is (10, 6).
    save_fig_name : str, optional
        If provided, the figure will be saved with this name.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axs : numpy.ndarray
        Array of axes objects
    """

    # Set default values
    if times is None:
        times = np.linspace(0, 3.0, 50)

    if logc is None:
        logc = np.log(5e-14)

    if m is None:
        m = 3.3

    if navg is None:
        navg = 2.8e6

    if a0 is None:
        a0 = 40.0

    # Generate ds_array from stress_periods if needed
    if ds_array is None:
        if stress_periods is None:
            # Default stress periods if neither ds_array \
            # nor stress_periods provided
            stress_periods = [
                (0.0, 0.6, 12.0),    # (start_time, end_time, stress_level)
                (0.6, 1.2, 20.0),
                (1.2, 1.8, 8.0),
                (1.8, 2.4, 25.0),
                (2.4, 3.0, 15.0),
            ]

        # Convert to array format required by VariableStressParisErdogan
        ds_array = []
        for i in range(len(times)-1):
            t_mid = (times[i] + times[i+1])/2  # midpoint of interval
            # Find which period contains this time
            for start, end, stress in stress_periods:
                if start <= t_mid < end:
                    ds_array.append(stress)
                    break

        ds_array = np.array(ds_array)

    # Create predictor with variable stress model
    predictor_var = CrackGrowthPredictor(
        model_class=VariableStressParisErdogan
        )

    # Predict crack growth with variable stress
    crack_length_var = predictor_var.predict_crack_growth(
        logc=logc,
        m=m,
        ds=ds_array,
        navg=navg,
        a0=a0,
        times=times
    )

    # For comparison, create a constant stress model with average stress
    avg_stress = np.mean(ds_array)
    predictor_const = CrackGrowthPredictor()  # Default is ParisErdogan

    # Predict crack growth with constant stress
    crack_length_const = predictor_const.predict_crack_growth(
        logc=logc,
        m=m,
        ds=avg_stress,
        navg=navg,
        a0=a0,
        times=times
    )

    # Get colormap colors
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i) for i in range(3)]

    # Plot the results with proper fig, ax structure
    fig, axs = plt.subplots(2, 1, figsize=figsize)
    # Plot stress profile
    axs[0].step(times[:-1], ds_array, where='post', color=colors[0],
                linewidth=2)
    axs[0].set_xlim(left=0, right=times[-1])
    axs[0].set_ylim(bottom=0)
    axs[0].xaxis.set_minor_locator(AutoMinorLocator())
    axs[0].yaxis.set_minor_locator(AutoMinorLocator())
    axs[0].tick_params(which='both', direction='in', top=True, right=True)
    axs[0].set_xlabel('Time (years)')
    axs[0].set_ylabel(r"Stress Range $\Delta S$ (MPa)")
    # axs[0].set_title('Variable Stress Profile')
    axs[0].grid(True, linestyle='--', alpha=0.7)

    # Plot crack growth
    axs[1].plot(times, crack_length_var, '-', color=colors[1], linewidth=2,
                label='Variable Stress')
    axs[1].plot(times, crack_length_const, '--', color=colors[2], linewidth=2,
                label='Constant Avg. Stress')
    axs[1].set_xlim(left=0, right=times[-1])
    axs[1].set_ylim(bottom=25, top=160)
    axs[1].xaxis.set_minor_locator(AutoMinorLocator())
    axs[1].yaxis.set_minor_locator(AutoMinorLocator())
    axs[1].tick_params(which='both', direction='in', top=True, right=True)
    axs[1].set_xlabel('Time (years)')
    axs[1].set_ylabel(r'Crack Length $\alpha$ (mm)')
    # axs[1].set_title('Crack Growth Comparison')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Save figure if requested
    if save_fig_name:
        # Get the root directory of the project
        dir_path = Path(__file__).resolve().parents[1]
        # Create the path to save the figure
        save_path = dir_path / 'figures' / save_fig_name
        # Raise an error if the directory does not exist
        if not save_path.parent.exists():
            raise FileNotFoundError(f"""Directory {save_path.parent}
                                    does not exist.""")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axs


def plot_stress_pattern_comparison(
        times=None, logc=None, m=None, navg=None, a0=None,
        min_stress=5.0, max_stress=25.0, patterns=None,
        cmap_name='Dark2', figsize=(10, 10), save_fig_name=None):
    """
    Plot the effect of different stress patterns on crack growth.

    Parameters
    ----------
    times : array, optional
        Time points for prediction. If None, creates a default array
        from 0 to 3 years.
    logc : float, optional
        Natural logarithm of Paris law parameter C. Default is ln(5e-14).
    m : float, optional
        Paris law exponent. Default is 3.2.
    navg : float, optional
        Average cycles per time unit. Default is 2.8e6 (cycles/year).
    a0 : float, optional
        Initial crack length. Default is 40.0 mm.
    min_stress : float, optional
        Minimum stress value for pattern generation.
    max_stress : float, optional
        Maximum stress value for pattern generation.
    patterns : list of str, optional
        List of patterns to generate:
        'increasing', 'decreasing', 'cyclical', 'random'.
        Default is all four patterns.
    cmap_name : str, optional
        Name of colormap to use. Default is 'Dark2'.
    figsize : tuple, optional
        Figure size. Default is (10, 10).
    save_fig_name : str, optional
        If provided, the figure will be saved with this name.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : numpy.ndarray
        Array of axes objects
    """

    # Set default values
    if times is None:
        times = np.linspace(0, 3.0, 50)

    if logc is None:
        logc = np.log(5e-14)

    if m is None:
        m = 3.2

    if navg is None:
        navg = 2.8e6

    if a0 is None:
        a0 = 40.0

    if patterns is None:
        patterns = ['increasing', 'decreasing', 'cyclical', 'random']

    # Function to generate different stress patterns
    def generate_stress_pattern(times, pattern_type,
                                min_stress=5.0, max_stress=25.0):
        """
        Generate different stress patterns for demonstration.
        """
        n_intervals = len(times) - 1

        if pattern_type == 'increasing':
            # Linearly increasing stress
            ds_array = np.linspace(min_stress, max_stress, n_intervals)

        elif pattern_type == 'decreasing':
            # Linearly decreasing stress
            ds_array = np.linspace(max_stress, min_stress, n_intervals)

        elif pattern_type == 'cyclical':
            # Sinusoidal pattern
            period = n_intervals / 3  # Complete 3 cycles
            ds_array = ((max_stress + min_stress)/2 +
                        (max_stress - min_stress)/2 *
                        np.sin(2 * np.pi * np.arange(n_intervals) / period))

        elif pattern_type == 'random':
            # Random stress values
            np.random.seed(42)  # For reproducibility
            ds_array = np.random.uniform(min_stress, max_stress, n_intervals)

        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")

        return ds_array

    # Create predictor with variable stress model
    predictor = CrackGrowthPredictor(model_class=VariableStressParisErdogan)

    # Create figure
    fig, axes = plt.subplots(len(patterns), 2, figsize=figsize)

    # Get colormap colors
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i) for i in range(len(patterns))]

    # Track final crack lengths for comparison
    crack_lengths = {}
    avg_stresses = {}

    # Create a list to store lines for figure legend
    legend_lines = []
    legend_labels = []

    for i, pattern in enumerate(patterns):
        # Generate stress pattern
        ds_array = generate_stress_pattern(times, pattern,
                                           min_stress, max_stress)
        avg_stresses[pattern] = np.mean(ds_array)

        # Predict crack growth
        crack_length = predictor.predict_crack_growth(
            logc=logc,
            m=m,
            ds=ds_array,
            navg=navg,
            a0=a0,
            times=times
        )

        # Store for later analysis
        crack_lengths[pattern] = crack_length

        # Plot stress pattern
        pattern_name = pattern.capitalize()
        axes[i, 0].step(times[:-1], ds_array, where='post',
                        color=colors[i], linewidth=2)
        axes[i, 0].set_xlim(left=0, right=times[-1])
        axes[i, 0].set_ylim(bottom=0, top=max_stress * 1.1)

        # Add grid and minor ticks
        axes[i, 0].grid(True, linestyle='--', alpha=0.5)
        axes[i, 0].xaxis.set_minor_locator(AutoMinorLocator())
        axes[i, 0].yaxis.set_minor_locator(AutoMinorLocator())
        axes[i, 0].tick_params(which='both', direction='in',
                               top=True, right=True)

        # Add legend
        # axes[i, 0].legend(loc='upper left', framealpha=0.9)

        # Set labels
        if i == len(patterns) - 1:
            axes[i, 0].set_xlabel('Time (years)')
        axes[i, 0].set_ylabel(r'Stress Range $\Delta S$ (MPa)')

        # Plot resulting crack growth (without label in the axis)
        line, = axes[i, 1].plot(times, crack_length,
                                color=colors[i], linewidth=2)
        axes[i, 1].set_xlim(left=0, right=times[-1])

        # Store line and label for figure legend
        legend_lines.append(line)
        legend_labels.append(pattern_name)

        # Add grid and minor ticks
        axes[i, 1].grid(True, linestyle='--', alpha=0.5)
        axes[i, 1].xaxis.set_minor_locator(AutoMinorLocator())
        axes[i, 1].yaxis.set_minor_locator(AutoMinorLocator())
        axes[i, 1].tick_params(which='both', direction='in',
                               top=True, right=True)

        # Set labels
        if i == len(patterns) - 1:
            axes[i, 1].set_xlabel('Time (years)')
        axes[i, 1].set_ylabel(r'Crack Length $\alpha$ (mm)')

    # Set y-axis limits for crack length plots to be the same
    max_length = max([cl[-1] for cl in crack_lengths.values()])
    for i in range(len(patterns)):
        axes[i, 1].set_ylim(bottom=a0*0.9, top=max_length*1.1)

    # Add a figure-level legend below the subplots
    fig.legend(
        legend_lines,
        legend_labels,
        loc='lower center',
        ncol=len(patterns),
        bbox_to_anchor=(0.5, 0.04),
        frameon=True,
        framealpha=0.9,
        borderaxespad=1
    )

    # Adjust spacing between subplots and make room for the figure legend
    plt.tight_layout(h_pad=2.5, w_pad=2.5, rect=[0, 0.08, 1, 1])

    # Save figure if requested
    if save_fig_name:
        # Get the root directory of the project
        dir_path = Path(__file__).resolve().parents[1]
        # Create the path to save the figure
        save_path = dir_path / 'figures' / save_fig_name
        # Raise an error if the directory does not exist
        if not save_path.parent.exists():
            raise FileNotFoundError(f"""Directory {save_path.parent}
                                    does not exist.""")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes


def plot_trajectories_with_observations(times, crack_lengths, obs_times,
                                        obs_lengths, labels=None,
                                        cmap_name='Paired', figsize=(8, 6),
                                        traj_alpha=0.8, obs_alpha=0.8,
                                        marker_size=30, save_fig_name=None):
    """
    Plot crack growth trajectories with overlaid observations.

    Parameters
    ----------
    times : list of arrays or array
        List of time arrays, one per trajectory
    crack_lengths : list of arrays or array
        List of crack length arrays, one per trajectory
    obs_times : list of arrays or array
        List of observation time arrays, one per trajectory
    obs_lengths : list of arrays or array
        List of observation crack length arrays, one per trajectory
    labels : list, optional
        Labels for the trajectories. If None, generates default labels
    cmap_name : str, optional
        Name of colormap to use. Default is 'Paired'.
    figsize : tuple, optional
        Figure size. Default is (8, 6).
    traj_alpha : float, optional
        Transparency of the trajectory lines. Default is 0.8.
    obs_alpha : float, optional
        Transparency of the observation markers. Default is 0.8.
    marker_size : float, optional
        Size of observation markers. Default is 30.
    save_fig_name : str, optional
        If provided, the figure will be saved with this name.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    """
    # Convert to list format if arrays are provided
    if isinstance(times, np.ndarray) and times.ndim > 1:
        times_list = [times[i, :] for i in range(times.shape[0])]
    elif isinstance(times, list):
        times_list = times
    else:
        times_list = [times]

    if isinstance(crack_lengths, np.ndarray) and crack_lengths.ndim > 1:
        crack_lengths_list = [crack_lengths[i, :]
                              for i in range(crack_lengths.shape[0])]
    elif isinstance(crack_lengths, list):
        crack_lengths_list = crack_lengths
    else:
        crack_lengths_list = [crack_lengths]

    if isinstance(obs_times, np.ndarray) and obs_times.ndim > 1:
        obs_times_list = [obs_times[i, :] for i in range(obs_times.shape[0])]
    elif isinstance(obs_times, list):
        obs_times_list = obs_times
    else:
        obs_times_list = [obs_times]

    if isinstance(obs_lengths, np.ndarray) and obs_lengths.ndim > 1:
        obs_lengths_list = [obs_lengths[i, :]
                            for i in range(obs_lengths.shape[0])]
    elif isinstance(obs_lengths, list):
        obs_lengths_list = obs_lengths
    else:
        obs_lengths_list = [obs_lengths]

    # Create default labels if none provided
    if labels is None:
        labels = [f"Trajectory {i+1}" for i in range(len(times_list))]
    elif len(labels) < len(times_list):
        labels = labels + [f"Trajectory {i+1}"
                           for i in range(len(labels), len(times_list))]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Get colormap
    cmap = plt.get_cmap(cmap_name)

    # Plot each trajectory and its observations
    for i in range(len(times_list)):
        # Get color for this trajectory
        color = cmap(i % cmap.N)

        # Clean trajectory data by removing trailing zeros if any
        t = times_list[i]
        cl = crack_lengths_list[i]

        non_zero_indices = np.where(cl > 0)[0]
        if len(non_zero_indices) > 0:
            last_idx = non_zero_indices[-1] + 1
            t_clean = t[:last_idx]
            cl_clean = cl[:last_idx]
        else:
            t_clean, cl_clean = t, cl

        # Plot the full trajectory
        traj_line, = ax.plot(t_clean, cl_clean, color=color, linewidth=2,
                             alpha=traj_alpha, label=labels[i])

        # Get corresponding observations
        if i < len(obs_times_list) and i < len(obs_lengths_list):
            obs_t = obs_times_list[i]
            obs_cl = obs_lengths_list[i]

            # Plot observations
            ax.scatter(obs_t, obs_cl, color=color, s=marker_size,
                       alpha=obs_alpha, marker='o', edgecolors='white',
                       linewidths=1.0, zorder=5)

    # Set labels and limits
    ax.set_xlabel(r'Time [years]')
    ax.set_ylabel(r'Crack length $\alpha$ [mm]')
    ax.set_xlim(left=0)

    # Add minor ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in', top=True, right=True)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)

    # Add legend
    legend_elements = []
    for i, label in enumerate(labels):
        color = cmap(i % cmap.N)
        # Line representing trajectory
        legend_elements.append(
            Line2D([0], [0], color=color, lw=2, label=label))

    # Add a generic marker for observations
    legend_elements.append(Line2D([0], [0], marker='o', color='gray',
                                  label='Observations', markerfacecolor='gray',
                                  markersize=8, markeredgecolor='white',
                                  markeredgewidth=0.5, linestyle='none'))

    ax.legend(handles=legend_elements, frameon=True, framealpha=0.9)

    # Tight layout
    plt.tight_layout()

    # Save figure if requested
    if save_fig_name:
        # Get the root directory of the project
        dir_path = Path(__file__).resolve().parents[1]
        # Create the path to save the figure
        save_path = dir_path / 'figures' / save_fig_name
        # Raise an error if the directory does not exist
        if not save_path.parent.exists():
            raise FileNotFoundError(f"Directory {save_path.parent} \
                                     does not exist.")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_posterior_trace(
        samples, var_names=None, plot_var_names=None,
        backend="matplotlib", save_fig_name=None,
        compact=False):
    """Ok
    Plots the trace and posterior distributions using ArviZ.

    Parameters
    ----------
    samples : dict
        Posterior samples, with shape (chains, draws) per variable.
    var_names : list of str, optional
        List of variables to plot. If None, all variables are plotted.
    plot_var_names : dict or list, optional
        Dictionary mapping variable names to display names,
        or a list of display names in the same order as var_names.
        If None, uses variable names as is.
    backend : str
        Backend used for plotting ("matplotlib" or "bokeh").
    save_fig_name : str, optional
        If provided, saves the figure to a file with the given name.
        Output directory is set internally.
        Default is None.
    compact : bool, optional
        Set to True for hierarchical models. Default is False.
    """

    idata = az.from_dict({'posterior': samples})
    # compact may be passed as a string from notebooks
    compact_bool = compact if isinstance(compact, bool) else str(compact) == 'True'
    az.plot_trace_dist(idata, var_names=var_names,
                       backend=backend, compact=compact_bool)

    # Access and modify the axes via the current figure
    fig = plt.gcf()
    axes_flat = fig.get_axes()

    if axes_flat:
        # If var_names is not provided, get all variable names from samples
        actual_var_names = var_names if var_names is not None \
            else list(samples.keys())

        # Process plot_var_names based on its type
        var_display_names = {}
        if plot_var_names is None:
            var_display_names = {var: var for var in actual_var_names}
        elif isinstance(plot_var_names, dict):
            var_display_names = {var: plot_var_names.get(var, var)
                                 for var in actual_var_names}
        elif isinstance(plot_var_names, list) and \
                len(plot_var_names) >= len(actual_var_names):
            var_display_names = {var: plot_var_names[i]
                                 for i, var in enumerate(actual_var_names)}
        else:
            var_display_names = {var: var for var in actual_var_names}
            print("Warning: plot_var_names format not recognized."
                  " Using original variable names.")

        # plot_trace_dist lays out axes as [dist_0, trace_0, dist_1, trace_1, ...]
        for i, var_name in enumerate(actual_var_names):
            if 2 * i + 1 >= len(axes_flat):
                break
            display_name = var_display_names.get(var_name, var_name)
            dist_ax = axes_flat[2 * i]
            trace_ax = axes_flat[2 * i + 1]

            dist_ax.set_title("")
            dist_ax.set_xlabel(display_name)
            dist_ax.set_ylabel("Density")
            dist_ax.xaxis.set_minor_locator(AutoMinorLocator())
            dist_ax.yaxis.set_minor_locator(AutoMinorLocator())
            dist_ax.tick_params(which='both', direction='in', top=True, right=True)

            trace_ax.set_title("")
            trace_ax.set_xlabel("MCMC Iteration")
            trace_ax.set_ylabel(display_name)
            trace_ax.xaxis.set_minor_locator(AutoMinorLocator())
            trace_ax.yaxis.set_minor_locator(AutoMinorLocator())
            trace_ax.tick_params(which='both', direction='in', top=True, right=True)

    plt.tight_layout()

    if save_fig_name is not None:
        main_dir = Path(__file__).resolve().parents[1]
        output_dir = main_dir / "figures"
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / save_fig_name
        suffix = output_path.suffix.lower()
        if suffix in [".png", ".jpg", ".jpeg"]:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
        else:
            plt.savefig(output_path, bbox_inches="tight")

    plt.show()


def _estimate_prior_range(prior_dist):
    """
    Estimate a reasonable plotting range for a prior distribution.

    Parameters
    ----------
    prior_dist : numpyro.distributions object
        The prior distribution

    Returns
    -------
    tuple : (min_val, max_val)
        Estimated reasonable range for plotting
    """
    import numpyro.distributions as dist

    # Handle specific distribution types
    if isinstance(prior_dist, dist.Normal):
        loc = prior_dist.loc
        scale = prior_dist.scale
        return float(loc - 4 * scale), float(loc + 4 * scale)

    elif isinstance(prior_dist, dist.HalfNormal):
        scale = prior_dist.scale
        return 0.0, float(4 * scale)  # HalfNormal support is [0, inf)

    elif isinstance(prior_dist, dist.Gamma):
        concentration = prior_dist.concentration
        rate = prior_dist.rate
        mean = concentration / rate
        std = np.sqrt(concentration) / rate
        return 0.0, float(mean + 4 * std)  # Gamma support is [0, inf)

    elif isinstance(prior_dist, dist.Weibull):
        concentration = prior_dist.concentration
        scale = prior_dist.scale
        # Weibull mean ≈ scale * Γ(1 + 1/concentration)
        mean_approx = scale * 1.0  # Rough approximation
        return 0.0, float(mean_approx * 3)  # Weibull support is [0, inf)

    elif isinstance(prior_dist, dist.Uniform):
        low = prior_dist.low
        high = prior_dist.high
        return float(low), float(high)

    else:
        # Fallback: try to sample from the distribution to estimate range
        try:
            key = jax.random.PRNGKey(42)
            samples = prior_dist.sample(key, (1000,))
            return float(np.percentile(samples, 0.1)), float(
                np.percentile(samples, 99.9))
        except Exception:
            # Last resort: return a default range
            return -10.0, 10.0


def plot_prior_posterior_comparison(
    posterior_samples,
    prior_dists,
    true_values=None,
    var_names=None,
    plot_var_names=None,
    figsize=(12, 8),
    n_cols=3,
    point_estimate='mean',
    save_fig_name=None,
    use_first_chain_only=False,
    prior_range_extension=2.0
):
    """
    Plot comparison of prior and posterior distributions
    with optional true values.

    Parameters
    ----------
    posterior_samples : dict
        Dictionary containing posterior samples for each parameter
    prior_dists : dict
        Dictionary containing numpyro distribution objects for priors
    true_values : dict, optional
        Dictionary containing true parameter values for comparison
    var_names : list, optional
        List of variable names to plot. If None, uses all
        variables in posterior_samples
    plot_var_names : dict or list, optional
        Display names for variables. If dict, maps var_names to display names.
        If list, should have same length as var_names
    figsize : tuple, default (12, 8)
        Figure size
    n_cols : int, default 3
        Number of columns in subplot grid
    point_estimate : str, default 'mean'
        Type of point estimate to show ('mean', 'median', 'mode')
    save_fig_name : str, optional
        If provided, saves the figure with this filename
    use_first_chain_only : bool, default False
        If True, uses only the first chain for posterior samples
    prior_range_extension : float, default 2.0
        Factor to extend the plotting range beyond the posterior range

    Returns
    -------
    fig, axes : matplotlib figure and axes objects
    """
    # Determine which variables to plot
    if var_names is None:
        var_names = list(posterior_samples.keys())

    # Create dictionary for display names
    var_display_names = {}
    if plot_var_names is None:
        var_display_names = {var: var for var in var_names}
    elif isinstance(plot_var_names, dict):
        var_display_names = {var: plot_var_names.get(var, var)
                             for var in var_names}
    elif isinstance(plot_var_names, list) and \
            len(plot_var_names) >= len(var_names):
        var_display_names = {var: plot_var_names[i]
                             for i, var in enumerate(var_names)}
    else:
        print("Warning: plot_var_names format not recognized. \
              Using original variable names.")
        var_display_names = {var: var for var in var_names}

    # Determine grid layout
    n_vars = len(var_names)
    n_rows = int(np.ceil(n_vars / n_cols))

    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Calculate plotting ranges for each variable
    plot_ranges = {}
    for var in var_names:
        samples = posterior_samples[var]
        if use_first_chain_only and samples.ndim > 1:
            samples = samples[0]  # Use only first chain

        # Get posterior range
        post_min, post_max = np.percentile(samples.flatten(), [0.5, 99.5])
        post_range = post_max - post_min

        # Get prior distribution for this variable
        prior_dist = prior_dists[var]

        # Handle different distribution types for prior range calculation
        if hasattr(prior_dist, 'support'):
            # Use distribution support if available
            support = prior_dist.support
            if hasattr(support, 'lower_bound') and hasattr(support,
                                                           'upper_bound'):
                prior_min = float(support.lower_bound) \
                    if support.lower_bound is not None else -np.inf
                prior_max = float(support.upper_bound) \
                    if support.upper_bound is not None else np.inf
            else:
                # Fallback: estimate reasonable range from distribution
                prior_min, prior_max = _estimate_prior_range(prior_dist)
        else:
            # Fallback: estimate reasonable range from distribution
            prior_min, prior_max = _estimate_prior_range(prior_dist)

        # Combine posterior and prior ranges with extension
        if np.isfinite(prior_min) and np.isfinite(prior_max):
            # Both bounds are finite
            range_min = min(post_min - prior_range_extension * post_range,
                            prior_min)
            range_max = max(post_max + prior_range_extension * post_range,
                            prior_max)
        elif np.isfinite(prior_min):
            # Only lower bound is finite (e.g., HalfNormal)
            range_min = min(post_min - prior_range_extension * post_range,
                            prior_min)
            range_max = post_max + prior_range_extension * post_range
        elif np.isfinite(prior_max):
            # Only upper bound is finite
            range_min = post_min - prior_range_extension * post_range
            range_max = max(post_max + prior_range_extension * post_range,
                            prior_max)
        else:
            # No finite bounds (e.g., Normal distribution)
            range_min = post_min - prior_range_extension * post_range
            range_max = post_max + prior_range_extension * post_range

        plot_ranges[var] = (range_min, range_max)

    # Plot each variable
    for i, var in enumerate(var_names):
        ax = axes.flat[i] if len(var_names) > 1 else axes

        # Get samples and plotting range
        samples = posterior_samples[var]
        if use_first_chain_only and samples.ndim > 1:
            samples = samples[0]

        range_min, range_max = plot_ranges[var]
        x_range = np.linspace(range_min, range_max, 1000)

        # Plot prior distribution
        prior_dist = prior_dists[var]
        try:
            # Evaluate log probability and convert to probability density
            log_prob = prior_dist.log_prob(x_range)
            # Handle any -inf values (outside support)
            log_prob = np.where(np.isfinite(log_prob), log_prob, -np.inf)
            prior_density = np.exp(log_prob)

            # Only plot where density is meaningful
            # (not zero due to support constraints)
            valid_mask = prior_density > 1e-10
            if np.any(valid_mask):
                ax.plot(x_range[valid_mask], prior_density[valid_mask],
                        color='coral', linestyle='dashed', linewidth=1.5,
                        label='Prior')
        except Exception as e:
            print(f"Warning: Could not plot prior for {var}: {e}")

        # Plot posterior KDE
        sns.kdeplot(data=samples, ax=ax, label='Posterior',
                    color='royalblue', fill=True,
                    alpha=0.2,
                    linewidth=1.5)

        # Calculate and plot posterior mode if requested
        if point_estimate == 'mode':
            kde = stats.gaussian_kde(samples)
            mode_idx = np.argmax(kde(x_range))
            mode_value = x_range[mode_idx]
            ax.axvline(mode_value, color='dodgerblue', linestyle='-.',
                       label='Posterior mode', linewidth=1.5)
        elif point_estimate == 'median':
            median_value = np.median(samples)
            ax.axvline(median_value, color='dodgerblue', linestyle='-.',
                       label='Posterior median', linewidth=1.5)
        elif point_estimate == 'mean':
            mean_value = np.mean(samples)
            ax.axvline(mean_value, color='dodgerblue', linestyle='-.',
                       label='Posterior mean', linewidth=1.5)

        # Plot true value if provided
        if true_values is not None and var in true_values:
            ax.axvline(true_values[var], color='maroon', linestyle='--',
                       label='True value', linewidth=2)
        # Set x-axis limits for positive-only distributions
        prior_dist = prior_dists[var]
        dist_name = type(prior_dist).__name__
        if dist_name in ['HalfNormal', 'Gamma', 'Exponential',
                         'LogNormal', 'Weibull']:
            ax.set_xlim(left=0)

        # Set plot labels and customize appearance
        ax.set_xlabel(var_display_names[var])
        ax.set_ylabel('Density')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='both', direction='in', top=True, right=True)
        # ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(frameon=True, framealpha=0.9)

    # Hide any unused axes
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)

    # Adjust layout
    plt.tight_layout()

    # Save figure if requested
    if save_fig_name is not None:
        main_dir = Path(__file__).resolve().parents[1]
        output_dir = main_dir / "figures"
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / save_fig_name
        suffix = output_path.suffix.lower()
        if suffix in [".png", ".jpg", ".jpeg"]:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
        else:
            plt.savefig(output_path, bbox_inches="tight")

    return fig, axes


def plot_random_effect_posteriors(posterior_samples, targets, param_name="ds",
                                  plot_var_name=None, figsize=None, n_cols=3,
                                  point_estimate="mode", kde_kwargs=None,
                                  save_fig_name=None,
                                  use_first_chain_only=False, priors=None):
    """
    Plot posterior distributions of random effects (component-specific
    parameters) as KDEs with target values overlaid as vertical lines.

    This function is designed for hierarchical Bayesian models where random
    effects vary by component (e.g., stress ranges ds[i] in multi-task
    learning).

    Parameters
    ----------
    posterior_samples : dict
        Dictionary containing posterior samples from MCMC inference.
        Should contain param_name as a key with shape (n_samples,
        n_components).
    targets : dict or None
        Dictionary containing target/true values for each component.
        Keys should be in the format f"{param_name}[{i}]" (e.g., "ds[0]",
        "ds[1]"). If None, no target lines are plotted.
    param_name : str, default="ds"
        Name of the random effect parameter in posterior_samples.
    plot_var_name : str, optional
        Display name for the parameter (e.g.,
        r"$\\Delta S \\ \\mathrm{[MPa]}$"). If None, uses param_name.
    figsize : tuple, optional
        Figure size (width, height). If None, automatically determined.
    n_cols : int, default=3
        Number of columns in subplot grid.
    point_estimate : str, default="mode"
        Type of point estimate to show ("mode", "mean", or "median").
    kde_kwargs : dict, optional
        Additional keyword arguments for seaborn.kdeplot.
    save_fig_name : str, optional
        Filename to save the figure. If None, figure is not saved.
    use_first_chain_only : bool, default=False
        If True, use only the first MCMC chain for plotting.
    priors : dict, optional
        Dictionary containing numpyro distribution objects for priors.
        Keys should be in the format f"{param_name}[{i}]" (e.g., "ds[0]",
        "ds[1]"). If provided, prior distributions will be plotted alongside
        posteriors. If None, no priors are plotted.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : numpy.ndarray
        Array of subplot axes.

    Examples
    --------
    >>> targets = {"ds[0]": 16.0, "ds[1]": 22.0, "ds[2]": 19.0}
    >>> fig, axes = plot_random_effect_posteriors(
    ...     posterior_samples=mtl_results['samples'],
    ...     targets=targets,
    ...     param_name="ds",
    ...     plot_var_name=r"$\\Delta S \\ \\mathrm{[MPa]}$"
    ... )

    >>> # Plot with priors and targets
    >>> import numpyro.distributions as dist
    >>> priors = {
    ...     "ds[0]": dist.Normal(15.0, 3.0),
    ...     "ds[1]": dist.Normal(20.0, 4.0),
    ...     "ds[2]": dist.Normal(18.0, 3.5)
    ... }
    >>> fig, axes = plot_random_effect_posteriors(
    ...     posterior_samples=mtl_results['samples'],
    ...     targets=targets,
    ...     priors=priors,
    ...     param_name="ds",
    ...     plot_var_name=r"$\\Delta S \\ \\mathrm{[MPa]}$"
    ... )
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from pathlib import Path

    # Extract parameter samples from posterior
    if param_name not in posterior_samples:
        raise ValueError(f"Parameter '{param_name}' not found in "
                         "posterior_samples")

    param_samples = posterior_samples[param_name]

    # Handle chain dimension if present
    if use_first_chain_only and param_samples.ndim == 3:
        param_samples = param_samples[0]  # Use first chain only
    elif param_samples.ndim == 3:
        # Flatten across chains: (n_chains, n_samples, n_components) ->
        # (n_chains*n_samples, n_components)
        n_chains, n_samples, n_components = param_samples.shape
        param_samples = param_samples.reshape(n_chains * n_samples,
                                              n_components)

    # Determine number of components
    if param_samples.ndim != 2:
        raise ValueError(f"Expected parameter samples to be 2D after "
                         f"processing, got shape {param_samples.shape}")

    n_samples, n_components = param_samples.shape

    # Set up plotting parameters
    if plot_var_name is None:
        plot_var_name = param_name

    if kde_kwargs is None:
        kde_kwargs = {}
    # Match styling from plot_prior_posterior_comparison
    kde_kwargs.setdefault('fill', True)
    kde_kwargs.setdefault('alpha', 0.2)
    kde_kwargs.setdefault('linewidth', 1.5)
    kde_kwargs.setdefault('color', 'royalblue')

    # Calculate subplot layout
    n_rows = int(np.ceil(n_components / n_cols))

    # Set figure size
    if figsize is None:
        width = min(4.0 * n_cols, 16.0)  # Cap at reasonable width
        height = min(3.0 * n_rows, 12.0)  # Cap at reasonable height
        figsize = (width, height)

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Handle single subplot case
    if n_components == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Flatten axes for easier indexing
    axes_flat = axes.flatten()

    # Plot each component's posterior
    for i in range(n_components):
        ax = axes_flat[i]

        # Get samples for this component
        component_samples = param_samples[:, i]

        # Create KDE plot
        sns.kdeplot(component_samples, ax=ax, **kde_kwargs)

        # Add prior distribution if available
        if priors is not None:
            prior_key = f"{param_name}[{i}]"
            if prior_key in priors:
                prior_dist = priors[prior_key]

                # Determine plotting range based on posterior samples
                post_min, post_max = component_samples.min(), \
                    component_samples.max()
                post_range = post_max - post_min
                range_extension = 0.5  # 20% extension beyond posterior range

                # Create extended range for prior plotting
                x_min = post_min - range_extension * post_range
                x_max = post_max + range_extension * post_range
                x_range = np.linspace(x_min, x_max, 1000)

                try:
                    # Evaluate prior log probability and convert to density
                    log_prob = prior_dist.log_prob(x_range)
                    log_prob = np.where(np.isfinite(log_prob), log_prob,
                                        -np.inf)
                    prior_density = np.exp(log_prob)

                    # Only plot where density is meaningful
                    valid_mask = prior_density > 1e-10
                    if np.any(valid_mask):
                        ax.plot(x_range[valid_mask], prior_density[valid_mask],
                                color='coral', linestyle='dashed',
                                linewidth=1.5, label='Prior')
                except Exception as e:
                    print(f"Warning: Could not plot prior for\
                           {prior_key}: {e}")

        # Add point estimate
        if point_estimate == "mode":
            # Estimate mode using KDE
            kde = stats.gaussian_kde(component_samples)
            x_range = np.linspace(component_samples.min(),
                                  component_samples.max(), 1000)
            kde_values = kde(x_range)
            mode_value = x_range[np.argmax(kde_values)]
            ax.axvline(mode_value, color='dodgerblue', linestyle='-.',
                       linewidth=1.5, label=f'Posterior mode: \
                        {mode_value:.2f} MPa')
        elif point_estimate == "mean":
            mean_value = np.mean(component_samples)
            ax.axvline(mean_value, color='dodgerblue', linestyle='-.',
                       linewidth=1.5, label=f'Mean: {mean_value:.2f}')
        elif point_estimate == "median":
            median_value = np.median(component_samples)
            ax.axvline(median_value, color='dodgerblue', linestyle='-.',
                       linewidth=1.5, label=f'Median: {median_value:.2f} \
                        MPa')

        # Add target value if available
        if targets is not None:
            target_key = f"{param_name}[{i}]"
            if target_key in targets:
                target_value = targets[target_key]
                ax.axvline(target_value, color='maroon', linestyle='--',
                           linewidth=2,
                           label=f'Target: {target_value:.2f} MPa')

        # Set labels and title
        ax.set_xlabel(plot_var_name)
        ax.set_ylabel('Density')

        # Match tick formatting from plot_prior_posterior_comparison
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='both', direction='in', top=True, right=True)

        # Match legend styling from plot_prior_posterior_comparison
        ax.legend(frameon=True, framealpha=0.9)

    # Hide unused subplots
    for i in range(n_components, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Adjust layout
    plt.tight_layout()

    # Save figure if filename provided
    if save_fig_name is not None:
        # Get the root directory of the project
        dir_path = Path(__file__).resolve().parents[1]
        # Create the path to save the figure
        save_path = dir_path / 'figures' / save_fig_name
        # Raise an error if the directory does not exist
        if not save_path.parent.exists():
            raise FileNotFoundError(f"Directory {save_path.parent} \
                                     does not exist.")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes


def plot_posterior_predictive_stl(posterior_predictions, true_times,
                                  true_crack_lengths, observed_times,
                                  observed_crack_lengths, max_samples=50,
                                  figsize=(12, 5), save_fig_name=None):
    """
    Plot posterior predictive samples for crack growth with observed data
    points in a 2-column subplot format showing both predictions and
    observations.

    Parameters
    ----------
    posterior_predictions : dict
        Dictionary containing posterior predictive samples with keys:
        - 'predicted_crack_lengths': Array of shape (n_samples, n_times)
        - 'obs': Optional array of shape (n_samples, n_times)
        with observation noise
    true_times : array
        Array of time points for the true trajectory
    true_crack_lengths : array
        Array of crack lengths for the true trajectory
    observed_times : array
        Array of time points for the observed data
    observed_crack_lengths : array
        Array of observed crack lengths (with measurement noise)
    max_samples : int, optional
        Maximum number of posterior samples to plot (for visual clarity)
    figsize : tuple, optional
        Figure size as (width, height) in inches.
        Default (12, 5) for 2-column layout
    save_fig_name : str, optional
        If provided, the figure will be saved with this name

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : numpy.ndarray
        Array of matplotlib axes objects (length 2)
    stats : dict
        Dictionary of computed statistics for both predictions and observations
    """
    import numpy as np

    # Create figure with 2 subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Initialize statistics dictionary
    stats = {}

    # Define subplot titles and data configurations
    plot_configs = [
        {
            'title': 'Posterior Predictions',
            'data_key': 'predicted_crack_lengths',
            'colors': {
                'samples': 'lavender',
                'ci': 'royalblue',
                'mean': 'royalblue',
                'mean_label': 'Posterior Mean'
            }
        },
        {
            'title': 'Posterior Observations',
            'data_key': 'obs',
            'colors': {
                'samples': 'thistle',
                'ci': 'darkslateblue',
                'mean': 'mediumslateblue',
                'mean_label': 'Posterior Mean'
            }
        }
    ]

    # Plot each subplot
    for subplot_idx, config in enumerate(plot_configs):
        ax = axes[subplot_idx]
        data_key = config['data_key']
        colors = config['colors']

        # Check if data exists for this plot type
        if data_key in posterior_predictions:
            samples = posterior_predictions[data_key]
            samples_mean = np.mean(samples, axis=0)
            lower_ci = np.percentile(samples, 2.5, axis=0)
            upper_ci = np.percentile(samples, 97.5, axis=0)

            # Plot individual sample trajectories
            for i in range(min(max_samples, samples.shape[0])):
                ax.plot(true_times, samples[i], color=colors['samples'],
                        alpha=0.3, zorder=1)

            # Add credible interval
            ax.fill_between(true_times, lower_ci, upper_ci,
                            color=colors['ci'], alpha=0.15,
                            label=r'95\% Credible Interval', zorder=2)

            # Plot posterior mean
            ax.plot(true_times, samples_mean, color=colors['mean'],
                    linewidth=1.5, linestyle="dashdot",
                    label=colors['mean_label'], zorder=4)

            # Calculate RMSE for this subplot
            # rmse = np.sqrt(np.mean((samples_mean -
            #  observed_crack_lengths)**2))

            # Store statistics
            # stats[f'{data_key}_rmse'] = rmse
            # stats[f'{data_key}_mean'] = samples_mean
            # stats[f'{data_key}_lower_ci'] = lower_ci
            # stats[f'{data_key}_upper_ci'] = upper_ci
            # stats[f'{data_key}_ci_width'] = upper_ci - lower_ci

        else:
            # If data doesn't exist, show message
            ax.text(0.5, 0.5, f'No {data_key} data available',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12, color='gray')

        # Plot the true trajectory on both subplots
        ax.plot(true_times, true_crack_lengths, 'darkorange', linewidth=1.5,
                label='True trajectory', zorder=3)

        # Plot the observations on both subplots
        ax.scatter(observed_times, observed_crack_lengths, color='coral',
                   s=40, label='Observations', zorder=5, edgecolors='white',
                   linewidths=1)

        # Customize each subplot
        ax.set_xlabel('Time (years)', fontsize=12)
        ax.set_ylabel('Crack length (mm)', fontsize=12)
        # Set grid, limits, and legend
        ax.grid(True, linestyle='--', alpha=0.2)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=min(true_crack_lengths)*0.95)
        ax.legend(fontsize=9, frameon=True, framealpha=0.2)

        # Add minor ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    # Adjust layout
    plt.tight_layout()

    # Save figure if filename provided
    if save_fig_name is not None:
        # Get the root directory of the project
        dir_path = Path(__file__).resolve().parents[1]
        # Create the path to save the figure
        save_path = dir_path / 'figures' / save_fig_name
        # Raise an error if the directory does not exist
        if not save_path.parent.exists():
            raise FileNotFoundError(f"Directory {save_path.parent} "
                                    "does not exist.")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes, stats


def plot_posterior_predictive_mtl(posterior_predictions_list,
                                  true_times_list, true_crack_lengths_list,
                                  observed_times_list,
                                  observed_crack_lengths_list,
                                  max_samples=50, figsize=(12, 8),
                                  save_fig_name=None):
    """
    Plot posterior predictive samples for Multi-Task Learning (MTL) model
    as subplots with rows representing components and columns representing
    predictions vs observations.

    Parameters
    ----------
    posterior_predictions_list : list of dict
        List of dictionaries containing posterior predictive samples
        for each component. Each dict should have keys:
        - 'predicted_crack_lengths': Array of shape (n_samples, n_times)
        - 'obs': Optional array of shape (n_samples, n_times)
        with observation noise
    true_times_list : list of array
        List of time arrays for true trajectories, one per component
    true_crack_lengths_list : list of array
        List of true crack length arrays, one per component
    observed_times_list : list of array
        List of observed time arrays, one per component
    observed_crack_lengths_list : list of array
        List of observed crack length arrays, one per component
    max_samples : int, default 50
        Maximum number of posterior samples to plot for clarity
    figsize : tuple, default (12, 8)
        Figure size as (width, height) in inches
    save_fig_name : str, optional
        If provided, saves the figure with this filename

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : numpy.ndarray
        Array of matplotlib axes objects
    stats : dict
        Dictionary containing statistics for each component
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    # Get number of components
    n_components = len(posterior_predictions_list)

    # Create subplot layout: (n_components, 2)
    # Columns: 0 = predictions, 1 = observations
    fig, axes = plt.subplots(n_components, 2, figsize=figsize, squeeze=False)

    # Initialize statistics dictionary
    component_stats = {}

    # Process each component
    for comp_idx in range(n_components):
        # Get data for this component
        posterior_predictive = posterior_predictions_list[comp_idx]
        true_times = true_times_list[comp_idx]
        true_crack_lengths = true_crack_lengths_list[comp_idx]
        observed_times = observed_times_list[comp_idx]
        observed_crack_lengths = observed_crack_lengths_list[comp_idx]

        # Plot predictions (left column)
        ax_pred = axes[comp_idx, 0]
        pred_samples = posterior_predictive['predicted_crack_lengths']
        pred_mean = np.mean(pred_samples, axis=0)
        pred_lower = np.percentile(pred_samples, 2.5, axis=0)
        pred_upper = np.percentile(pred_samples, 97.5, axis=0)

        # Plot individual prediction trajectories
        n_plot = min(max_samples, pred_samples.shape[0])
        for i in range(n_plot):
            ax_pred.plot(true_times, pred_samples[i], color='lavender',
                         alpha=0.4, zorder=1)

        # Plot credible interval
        ax_pred.fill_between(true_times, pred_lower, pred_upper,
                             color='royalblue', alpha=0.15,
                             label=r'95\% Credible Interval', zorder=2)

        # Plot posterior mean
        ax_pred.plot(true_times, pred_mean, color='royalblue',
                     linewidth=1.5, linestyle="dashdot",
                     label='Posterior Mean', zorder=4)

        # Plot true trajectory
        ax_pred.plot(true_times, true_crack_lengths, 'darkorange',
                     linewidth=1.5, label='True trajectory', zorder=3)

        # Plot observations
        ax_pred.scatter(observed_times, observed_crack_lengths,
                        color='coral', s=40, label='Observations',
                        zorder=5, edgecolors='white', linewidths=1)

        # Customize predictions subplot
        ax_pred.set_xlabel('Time (years)', fontsize=10)
        ax_pred.set_ylabel('Crack length (mm)', fontsize=10)
        # ax_pred.set_title(f'Component {comp_idx + 1} - Predictions',
        #                   fontsize=11)
        ax_pred.grid(True, linestyle='--', alpha=0.2)
        ax_pred.set_xlim(left=0)
        ax_pred.set_ylim(bottom=min(true_crack_lengths)*0.95)
        ax_pred.legend(fontsize=8, frameon=True, framealpha=0.2)
        ax_pred.xaxis.set_minor_locator(AutoMinorLocator())
        ax_pred.yaxis.set_minor_locator(AutoMinorLocator())

        # Plot observations (right column)
        ax_obs = axes[comp_idx, 1]
        if "obs" in posterior_predictive:
            obs_samples = posterior_predictive['obs']
            obs_mean = np.mean(obs_samples, axis=0)
            obs_lower = np.percentile(obs_samples, 2.5, axis=0)
            obs_upper = np.percentile(obs_samples, 97.5, axis=0)

            # Plot individual observation trajectories
            for i in range(n_plot):
                ax_obs.plot(true_times, obs_samples[i], color='thistle',
                            alpha=0.3, zorder=1)

            # Plot credible interval for observations
            ax_obs.fill_between(true_times, obs_lower, obs_upper,
                                color='darkslateblue', alpha=0.15,
                                label=r'95\% Credible Interval', zorder=2)

            # Plot mean of observations
            ax_obs.plot(true_times, obs_mean, color='mediumslateblue',
                        linewidth=1.5, linestyle="dashdot",
                        label='Posterior Mean', zorder=4)
        else:
            # Fallback to predictions if observations not available
            print(f"Warning: No observation samples for component "
                  f"{comp_idx + 1}, using predictions")
            for i in range(n_plot):
                ax_obs.plot(true_times, pred_samples[i], color='thistle',
                            alpha=0.3, zorder=1)

            ax_obs.fill_between(true_times, pred_lower, pred_upper,
                                color='darkslateblue', alpha=0.15,
                                label=r'95% Credible Interval', zorder=2)

            ax_obs.plot(true_times, pred_mean, color='mediumslateblue',
                        linewidth=1.5, linestyle="dashdot",
                        label='Posterior Mean', zorder=4)

        # Plot true trajectory
        ax_obs.plot(true_times, true_crack_lengths, 'darkorange',
                    linewidth=1.5, label='True trajectory', zorder=3)

        # Plot observations
        ax_obs.scatter(observed_times, observed_crack_lengths,
                       color='coral', s=40, label='Observations',
                       zorder=5, edgecolors='white', linewidths=1)

        # Customize observations subplot
        ax_obs.set_xlabel('Time (years)', fontsize=10)
        ax_obs.set_ylabel('Crack length (mm)', fontsize=10)
        # ax_obs.set_title(f'Component {comp_idx + 1} - Observations',
        #                  fontsize=11)
        ax_obs.grid(True, linestyle='--', alpha=0.2)
        ax_obs.set_xlim(left=0)
        ax_obs.set_ylim(bottom=min(true_crack_lengths)*0.95)
        ax_obs.legend(fontsize=8, frameon=True, framealpha=0.2)
        ax_obs.xaxis.set_minor_locator(AutoMinorLocator())
        ax_obs.yaxis.set_minor_locator(AutoMinorLocator())

        # # Calculate statistics for this component
        # pred_rmse = np.sqrt(np.mean((pred_mean - observed_crack_lengths)**2))
        # component_stats[f"component_{comp_idx + 1}"] = {
        #     "predictions_rmse": pred_rmse,
        #     "predictions_mean": pred_mean,
        #     "predictions_lower_ci": pred_lower,
        #     "predictions_upper_ci": pred_upper,
        #     "predictions_ci_width": pred_upper - pred_lower
        # }

        # if "obs" in posterior_predictive:
        #     obs_rmse = np.sqrt(np.mean((obs_mean -
        # observed_crack_lengths)**2))
        #     component_stats[f"component_{comp_idx + 1}"].update({
        #         "observations_rmse": obs_rmse,
        #         "observations_mean": obs_mean,
        #         "observations_lower_ci": obs_lower,
        #         "observations_upper_ci": obs_upper,
        #         "observations_ci_width": obs_upper - obs_lower
        #     })

    # Adjust layout
    plt.tight_layout()

    # Save figure if requested
    if save_fig_name is not None:
        from pathlib import Path
        main_dir = Path(__file__).resolve().parents[1]
        output_dir = main_dir / "figures"
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / save_fig_name
        suffix = output_path.suffix.lower()
        if suffix in [".png", ".jpg", ".jpeg"]:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
        else:
            plt.savefig(output_path, bbox_inches="tight")

    return fig, axes, component_stats


def plot_performance_metrics(summary_df, save_fig_name=None,
                             component_labels=None, parameter_labels=None,
                             figsize=(12, 8),
                             show_values=True, colors=None):
    """
    Create bar charts showing RMSE and MAPE performance metrics.

    Parameters:
    -----------
    summary_df : pd.DataFrame
        Summary table from create_comparison_summary() with component-parameter
        columns
    save_fig_name : str, optional
        If provided, saves the figure with this filename.
        If None, does not save the figure.
    component_labels : dict, optional
        Dictionary mapping component names to custom display labels.
        Keys should be component names from the 'Component' column,
        values should be the desired display labels for the legend.
        If None, uses original component names as labels.
    parameter_labels : dict, optional
        Dictionary mapping parameter names to custom display labels for x-axis.
        Keys should be parameter names (e.g., 'ds', 'logc', 'm', 'noise_std'),
        values should be the desired display labels (e.g., LaTeX-formatted).
        If None, uses original parameter names as labels.
    figsize : tuple, optional
        Figure size as (width, height). Default is (12, 8)
    show_values : bool, optional
        Whether to show metric values on top of bars. Default is True
    colors : list or dict, optional
        Colors for different components. If None, uses default color palette

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    axes : tuple
        Tuple of subplot axes (ax1, ax2)

    Examples:
    ---------
    >>> # Basic usage with data from create_comparison_summary()
    >>> from src.output_utils import create_comparison_summary
    >>> from src.visualisation import plot_performance_metrics
    >>>
    >>> # Assuming you have component_results from Bayesian inference
    >>> summary_df = create_comparison_summary(component_results)
    >>> fig, axes = plot_performance_metrics(summary_df)
    >>>
    >>> # Save to file with custom settings
    >>> fig, axes = plot_performance_metrics(
    ...     summary_df,
    ...     save_fig_name='performance_comparison.png',
    ...     figsize=(16, 10),
    ...     show_values=True
    ... )
    >>>
    >>> # Use custom component labels and colors
    >>> component_labels = {
    ...     'Component 1': 'STL Classic',
    ...     'Component 2': 'STL Spectral',
    ...     'Component 3': 'MTL Model'
    ... }
    >>> custom_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    >>> fig, axes = plot_performance_metrics(
    ...     summary_df,
    ...     component_labels=component_labels,
    ...     colors=custom_colors,
    ...     save_fig_name='custom_labeled_metrics.pdf'
    ... )
    >>>
    >>> # Use custom parameter labels with LaTeX formatting
    >>> parameter_labels = {
    ...     'ds': '$\\Delta S$',
    ...     'logc': '$\\ln C$',
    ...     'm': '$m$',
    ...     'noise_std': '$\\sigma_{\\text{noise}}$'
    ... }
    >>> fig, axes = plot_performance_metrics(
    ...     summary_df,
    ...     parameter_labels=parameter_labels,
    ...     save_fig_name='latex_param_metrics.pdf'
    ... )
    >>>
    >>> # Use both custom component and parameter labels
    >>> fig, axes = plot_performance_metrics(
    ...     summary_df,
    ...     component_labels=component_labels,
    ...     parameter_labels=parameter_labels,
    ...     save_fig_name='fully_customized_metrics.pdf'
    ... )
    >>>
    >>> # Use only custom labels without custom colors
    >>> fig, axes = plot_performance_metrics(
    ...     summary_df,
    ...     component_labels=component_labels,
    ...     save_fig_name='custom_colors_metrics.pdf'
    ... )
    >>>
    >>> # Load saved comparison data and visualize
    >>> import pandas as pd
    >>> df = pd.read_csv('outputs/stl_comparison_classic.csv')
    >>> fig, axes = plot_performance_metrics(df, save_fig_name='metrics.png')

    Notes:
    ------
    The function expects a DataFrame with the structure produced by
    create_comparison_summary(), which includes:
    - 'Component' column with component names
    - Parameter-specific columns with '_rmse' and '_mape' suffixes
    - Common parameters: 'ds', 'logc', 'm', 'noise_std'

    The function automatically detects available parameters from column names
    and creates grouped bar charts comparing RMSE and MAPE across components.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # Validate input
    if not isinstance(summary_df, pd.DataFrame):
        raise ValueError("summary_df must be a pandas DataFrame")

    # Check for Component column
    if 'Component' not in summary_df.columns:
        raise ValueError("Missing required 'Component' column")

    # Extract parameter names from column names
    # (look for _rmse and _mape suffixes)
    rmse_columns = [col for col in summary_df.columns
                    if col.endswith('_rmse')]
    mape_columns = [col for col in summary_df.columns
                    if col.endswith('_mape')]

    if not rmse_columns or not mape_columns:
        raise ValueError("No RMSE or MAPE columns found. Expected columns "
                         "with '_rmse' and '_mape' suffixes.")

    # Extract parameter names (remove the _rmse/_mape suffix)
    parameters = sorted(list(set([col[:-5] for col in rmse_columns])))
    components = summary_df['Component'].tolist()

    # Set up component display labels
    if component_labels is None:
        # Use original component names as labels
        display_labels = {comp: comp for comp in components}
    elif isinstance(component_labels, dict):
        # Use provided mapping, fall back to original name if not found
        display_labels = {comp: component_labels.get(comp, comp)
                          for comp in components}
    else:
        raise ValueError("component_labels must be a dictionary mapping "
                         "component names to display labels")

    # Set up colors
    if colors is None:
        # Use a nice color palette
        color_palette = plt.cm.Set3(np.linspace(0, 1, len(components)))
        colors = {comp: color_palette[i] for i, comp in enumerate(components)}
    elif isinstance(colors, list):
        if len(colors) < len(components):
            error_msg = (f"Need at least {len(components)} colors for "
                         f"{len(components)} components")
            raise ValueError(error_msg)
        colors = {comp: colors[i] for i, comp in enumerate(components)}

    # Set up parameter display labels
    if parameter_labels is None:
        # Use original parameter names as labels
        param_display_labels = parameters
    elif isinstance(parameter_labels, dict):
        # Use provided mapping, fall back to original name if not found
        param_display_labels = [parameter_labels.get(param, param)
                                for param in parameters]
    else:
        raise ValueError("parameter_labels must be a dictionary mapping "
                         "parameter names to display labels")

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Set up bar positions
    x = np.arange(len(parameters))
    # Total width of 0.8, divided by number of components
    width = 0.8 / len(components)

    # Plot RMSE
    for i, component in enumerate(components):
        rmse_values = []
        for param in parameters:
            rmse_col = f'{param}_rmse'
            if rmse_col in summary_df.columns:
                # Get the value for this component (row i)
                rmse_values.append(summary_df.iloc[i][rmse_col])
            else:
                rmse_values.append(0)  # or np.nan

        bars1 = ax1.bar(x + i * width - width * (len(components) - 1) / 2,
                        rmse_values, width,
                        label=display_labels[component],
                        color=colors[component],
                        alpha=0.8)

        # Add value labels on bars
        if show_values:
            for bar, value in zip(bars1, rmse_values):
                if value > 0:  # Only show non-zero values
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                             f'{value:.3f}',
                             ha='center', va='bottom', fontsize=9)

    # Plot MAPE
    for i, component in enumerate(components):
        mape_values = []
        for param in parameters:
            mape_col = f'{param}_mape'
            if mape_col in summary_df.columns:
                # Get the value for this component (row i)
                mape_values.append(summary_df.iloc[i][mape_col])
            else:
                mape_values.append(0)  # or np.nan

        bars2 = ax2.bar(x + i * width - width * (len(components) - 1) / 2,
                        mape_values, width,
                        label=display_labels[component],
                        color=colors[component],
                        alpha=0.8)

        # Add value labels on bars
        if show_values:
            for bar, value in zip(bars2, mape_values):
                if value > 0:  # Only show non-zero values
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                             f'{value:.2f}%',
                             ha='center', va='bottom', fontsize=9)

    # Customize RMSE subplot
    ax1.set_xlabel('Parameters', fontsize=12)
    ax1.set_ylabel('RMSE', fontsize=12)
    # ax1.set_title('Root Mean Square Error', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(param_display_labels, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Customize MAPE subplot
    ax2.set_xlabel('Parameters', fontsize=12)
    ax2.set_ylabel(r'MAPE (\%)', fontsize=12)
    # ax2.set_title('Mean Abs. Percentage Error', fontsize=14,
    #               fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(param_display_labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Adjust layout
    plt.tight_layout()

    # Save figure if requested
    if save_fig_name is not None:
        main_dir = Path(__file__).resolve().parents[1]
        output_dir = main_dir / "figures"
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / save_fig_name
        suffix = output_path.suffix.lower()
        if suffix in [".png", ".jpg", ".jpeg"]:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
        else:
            plt.savefig(output_path, bbox_inches="tight")

    # Show the plot
    plt.show()

    return fig, (ax1, ax2)


def plot_posterior_pairplot(posterior_samples, var_names=None,
                            plot_var_names=None, true_values=None,
                            figsize=None, use_first_chain_only=False,
                            diag_kind='kde', corner_kwargs=None,
                            save_fig_name=None):
    """
    Create a pairplot of posterior parameter samples showing correlations
    and marginal distributions.

    Parameters
    ----------
    posterior_samples : dict
        Dictionary containing posterior samples from MCMC inference.
        Each parameter should have shape (n_chains, n_samples) or (n_samples,).
    var_names : list of str, optional
        List of parameter names to include in the pairplot.
        If None, uses all parameters in posterior_samples.
    plot_var_names : dict or list, optional
        Display names for parameters. If dict, maps var_names to display names.
        If list, should have same length as var_names.
        If None, uses original parameter names.
    true_values : dict, optional
        Dictionary containing true parameter values to overlay as
        vertical/horizontal lines. Keys should match var_names.
    figsize : tuple, optional
        Figure size (width, height). If None, automatically determined
        based on number of parameters.
    use_first_chain_only : bool, default False
        If True, uses only the first MCMC chain for plotting.
    diag_kind : str, default 'kde'
        Type of plot for diagonal elements ('kde', 'hist', or 'auto').
    corner_kwargs : dict, optional
        Additional keyword arguments for seaborn PairGrid configuration.
    save_fig_name : str, optional
        Filename to save the figure. If None, figure is not saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    grid : seaborn.PairGrid
        The seaborn PairGrid object.

    Examples
    --------
    >>> # Basic pairplot
    >>> fig, grid = plot_posterior_pairplot(
    ...     posterior_samples=samples,
    ...     var_names=['logc', 'm', 'ds', 'noise_std']
    ... )

    >>> # With custom parameter names and true values
    >>> plot_var_names = {
    ...     'logc': r'$\\ln C$',
    ...     'm': r'$m$',
    ...     'ds': r'$\\Delta S$ [MPa]',
    ...     'noise_std': r'$\\sigma_{\\text{noise}}$'
    ... }
    >>> true_vals = {'logc': -30.2, 'm': 3.3, 'ds': 18.5, 'noise_std': 0.5}
    >>> fig, grid = plot_posterior_pairplot(
    ...     posterior_samples=samples,
    ...     var_names=['logc', 'm', 'ds', 'noise_std'],
    ...     plot_var_names=plot_var_names,
    ...     true_values=true_vals,
    ...     save_fig_name='parameter_pairplot.pdf'
    ... )
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from pathlib import Path

    # Determine which variables to plot
    if var_names is None:
        var_names = list(posterior_samples.keys())

    # Handle display names
    var_display_names = {}
    if plot_var_names is None:
        var_display_names = {var: var for var in var_names}
    elif isinstance(plot_var_names, dict):
        var_display_names = {var: plot_var_names.get(var, var)
                             for var in var_names}
    elif isinstance(plot_var_names, list) and \
            len(plot_var_names) >= len(var_names):
        var_display_names = {var: plot_var_names[i]
                             for i, var in enumerate(var_names)}
    else:
        var_display_names = {var: var for var in var_names}

    # Prepare data for plotting
    plot_data = {}
    for var in var_names:
        if var not in posterior_samples:
            raise ValueError(f"Variable '{var}' not found in "
                             f"posterior_samples")

        samples = posterior_samples[var]

        # Handle chain dimension if present
        if use_first_chain_only and samples.ndim > 1:
            samples = samples[0]  # Take first chain
        elif samples.ndim > 1:
            # Flatten all chains
            samples = samples.reshape(-1)

        plot_data[var_display_names[var]] = samples

    # Create DataFrame
    df = pd.DataFrame(plot_data)

    # Set figure size if not provided
    n_vars = len(var_names)
    if figsize is None:
        base_size = 2.5
        figsize = (n_vars * base_size, n_vars * base_size)

    # Set up corner plot kwargs
    if corner_kwargs is None:
        corner_kwargs = {}

    corner_kwargs.setdefault('height', figsize[0]/n_vars)
    corner_kwargs.setdefault('aspect', 1)
    corner_kwargs.setdefault('diag_sharey', False)

    # Create PairGrid
    grid = sns.PairGrid(df, **corner_kwargs)

    # Configure diagonal plots (marginals)
    if diag_kind == 'kde':
        grid.map_diag(sns.kdeplot, fill=True, color='royalblue',
                      alpha=0.7, linewidth=1.5)
    elif diag_kind == 'hist':
        grid.map_diag(plt.hist, bins=30, color='royalblue',
                      alpha=0.7, edgecolor='white')
    else:  # auto
        grid.map_diag(sns.histplot, kde=True, color='royalblue',
                      alpha=0.7, edgecolor='white', stat='density')

    # Configure off-diagonal plots (scatter + contours)

    def scatter_with_contour(x, y, **kwargs):
        ax = plt.gca()
        # Scatter plot with transparency
        ax.scatter(x, y, alpha=0.4, s=8, edgecolors='white',
                   linewidth=0.1, color='royalblue')
        # Add contour lines
        try:
            sns.kdeplot(x=x, y=y, levels=5, colors='darkblue',
                        alpha=0.6, linewidths=1)
        except Exception:
            pass  # Skip contours if KDE fails

    grid.map_lower(scatter_with_contour)
    grid.map_upper(sns.kdeplot, fill=True, levels=8, alpha=0.3, cmap="Blues")

    # Add correlation coefficients to upper triangle
    def add_correlation(x, y, **kwargs):
        ax = plt.gca()
        corr_coef = np.corrcoef(x, y)[0, 1]
        ax.text(0.5, 0.5, f'$r = {corr_coef:.3f}$',
                transform=ax.transAxes, fontsize=12, ha='center',
                va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.8, edgecolor='lightgray'))

    # Remove the upper triangle KDE and add correlations instead
    for i in range(n_vars):
        for j in range(n_vars):
            if i < j:  # Upper triangle
                ax = grid.axes[i, j]
                ax.clear()
                var_x = list(df.columns)[j]
                var_y = list(df.columns)[i]
                add_correlation(df[var_x], df[var_y])
                ax.set_xlim(df[var_x].min(), df[var_x].max())
                ax.set_ylim(df[var_y].min(), df[var_y].max())

    # Add true values if provided
    if true_values is not None:
        for i, var in enumerate(var_names):
            if var in true_values:
                true_val = true_values[var]

                # Add vertical line to diagonal plot
                diag_ax = grid.axes[i, i]
                diag_ax.axvline(true_val, color='red', linestyle='--',
                                linewidth=2, alpha=0.8, label='True Value')

                # Add lines to off-diagonal plots
                for j in range(n_vars):
                    if i != j:
                        # Vertical line (when var is on x-axis)
                        if j > i:  # Upper triangle (correlation text)
                            pass  # Skip upper triangle
                        else:  # Lower triangle
                            ax = grid.axes[j, i]
                            ax.axvline(true_val, color='red',
                                       linestyle='--', linewidth=1.5,
                                       alpha=0.6)

                        # Horizontal line (when var is on y-axis)
                        if i > j:  # Lower triangle
                            ax = grid.axes[i, j]
                            ax.axhline(true_val, color='red',
                                       linestyle='--', linewidth=1.5,
                                       alpha=0.6)

    # Customize axes
    for i in range(n_vars):
        for j in range(n_vars):
            ax = grid.axes[i, j]
            ax.tick_params(which='both', direction='in', top=True, right=True)

            # Add minor ticks for diagonal plots
            if i == j:
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())

    # Add legend for true values if they exist
    if true_values is not None:
        grid.axes[0, 0].legend(loc='upper left', frameon=True, framealpha=0.9)

    # Get figure object
    fig = grid.fig

    # Adjust layout
    plt.tight_layout()

    # Save figure if requested
    if save_fig_name is not None:
        # Get the root directory of the project
        dir_path = Path(__file__).resolve().parents[1]
        # Create the path to save the figure
        save_path = dir_path / 'figures' / save_fig_name
        # Raise an error if the directory does not exist
        if not save_path.parent.exists():
            raise FileNotFoundError(f"Directory {save_path.parent} "
                                    f"does not exist")

        # Determine file format and save with appropriate settings
        suffix = save_path.suffix.lower()
        if suffix in [".png", ".jpg", ".jpeg"]:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        else:
            plt.savefig(save_path, bbox_inches="tight")

    plt.show()

    return fig, grid
