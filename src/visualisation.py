import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
import numpy as np
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
        save_path = dir_path / 'outputs' / save_fig_name
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
        save_path = dir_path / 'outputs' / save_fig_name
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
    colors = [cmap(i % cmap.N) for i in range(len(patterns))]

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
        save_path = dir_path / 'outputs' / save_fig_name
        # Raise an error if the directory does not exist
        if not save_path.parent.exists():
            raise FileNotFoundError(f"""Directory {save_path.parent}
                                    does not exist.""")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes


def plot_selected_trajectories(times, crack_lengths, labels=None,
                               cmap_name='Paired', figsize=(8, 6),
                               alpha=0.8, save_fig_name=None):
    """
    Plot selected crack growth trajectories.

    Parameters
    ----------
    times : list of arrays or array
        List of time arrays, one per trajectory, or a single 2D array
    crack_lengths : list of arrays or array
        List of crack length arrays, one per trajectory, or a single 2D array
    labels : list, optional
        Labels for the trajectories. If None, generates default labels
    cmap_name : str, optional
        Name of colormap to use. Default is 'Paired'.
    figsize : tuple, optional
        Figure size. Default is (8, 6).
    alpha : float, optional
        Transparency of the trajectory lines. Default is 0.8.
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

    # Plot each trajectory
    for i, (t, cl, label) in enumerate(
            zip(times_list, crack_lengths_list, labels)):
        # Clean data by removing trailing zeros if any
        non_zero_indices = np.where(cl > 0)[0]
        if len(non_zero_indices) > 0:
            last_idx = non_zero_indices[-1] + 1
            t_clean = t[:last_idx]
            cl_clean = cl[:last_idx]
        else:
            t_clean, cl_clean = t, cl

        ax.plot(t_clean, cl_clean, color=cmap(i % cmap.N), linewidth=2,
                alpha=alpha, label=label)

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
    ax.legend(frameon=True, framealpha=0.9)

    # Tight layout
    plt.tight_layout()

    # Save figure if requested
    if save_fig_name:
        # Get the root directory of the project
        dir_path = Path(__file__).resolve().parents[1]
        # Create the path to save the figure
        save_path = dir_path / 'outputs' / save_fig_name
        # Raise an error if the directory does not exist
        if not save_path.parent.exists():
            raise FileNotFoundError(f"Directory {save_path.parent} \
                                     does not exist.")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


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
                       linewidths=0.5)

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
        save_path = dir_path / 'outputs' / save_fig_name
        # Raise an error if the directory does not exist
        if not save_path.parent.exists():
            raise FileNotFoundError(f"Directory {save_path.parent} \
                                     does not exist.")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_posterior_trace(
        samples, var_names=None, plot_var_names=None,
        backend="matplotlib", save_fig_name=None):
    """
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
    """

    idata = az.from_dict(posterior=samples)
    axes = az.plot_trace(idata, var_names=var_names,
                         backend=backend, compact=False)

    # Access and modify the axes if they exist
    if axes is not None and len(axes) > 0:
        num_vars = axes.shape[0]

        # Create a mapping of variable names to display names
        var_display_names = {}

        # If var_names is not provided, get all variable names from samples
        actual_var_names = var_names if var_names is not None \
            else list(samples.keys())

        # Process plot_var_names based on its type
        if plot_var_names is None:
            # Use the original variable names
            var_display_names = {var: var for var in actual_var_names}
        elif isinstance(plot_var_names, dict):
            # Use the provided mappings, fall back to original names if missing
            var_display_names = {var: plot_var_names.get(var, var)
                                 for var in actual_var_names}
        elif isinstance(plot_var_names, list) and \
                len(plot_var_names) >= len(actual_var_names):
            # Use list items as display names in the same order as var_names
            var_display_names = {var: plot_var_names[i]
                                 for i, var in enumerate(actual_var_names)}
        else:
            # Fallback to original names if plot_var_names has invalid format
            var_display_names = {var: var for var in actual_var_names}
            print("Warning: plot_var_names format not recognized.\
                   Using original variable names.")

        for i in range(num_vars):
            # Remove default titles
            if i < axes.shape[0] and 0 < axes.shape[1]:
                axes[i, 0].set_title("")
            if i < axes.shape[0] and 1 < axes.shape[1]:
                axes[i, 1].set_title("")

            # Set the labels for the KDE posteriors and trace plots
            if i < axes.shape[0] and 0 < axes.shape[1]:
                var_name = actual_var_names[i] if i < len(actual_var_names) \
                    else f"Parameter {i+1}"
                display_name = var_display_names.get(var_name, var_name)

                axes[i, 0].xaxis.set_minor_locator(AutoMinorLocator())
                axes[i, 0].yaxis.set_minor_locator(AutoMinorLocator())
                axes[i, 0].tick_params(which='both', direction='in',
                                       top=True, right=True)
                axes[i, 0].tick_params(which='both', direction='in',
                                       top=True, right=True)
                axes[i, 0].set_ylabel("Density")
                axes[i, 0].set_xlabel(display_name)

            if i < axes.shape[0] and 1 < axes.shape[1]:
                var_name = actual_var_names[i] if i < len(actual_var_names) \
                    else f"Parameter {i+1}"
                display_name = var_display_names.get(var_name, var_name)

                axes[i, 1].xaxis.set_minor_locator(AutoMinorLocator())
                axes[i, 1].yaxis.set_minor_locator(AutoMinorLocator())
                axes[i, 1].tick_params(which='both', direction='in',
                                       top=True, right=True)
                axes[i, 1].set_xlabel("MCMC Iteration")
                axes[i, 1].set_ylabel(display_name)

    plt.tight_layout()

    if save_fig_name is not None:
        main_dir = Path(__file__).resolve().parents[1]
        output_dir = main_dir / "outputs"
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / save_fig_name
        suffix = output_path.suffix.lower()
        if suffix in [".png", ".jpg", ".jpeg"]:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
        else:
            plt.savefig(output_path, bbox_inches="tight")

    plt.show()


def plot_prior_posterior_comparison(
        posterior_samples, prior_dists, true_values=None,
        var_names=None, plot_var_names=None, figsize=(12, 8),
        n_cols=2, point_estimate='mode', save_fig_name=None,
        use_first_chain_only=False, prior_range_extension=3.0):
    """
    Plot prior vs posterior distributions with true values and point estimates.

    Parameters
    ----------
    posterior_samples : dict
        Dictionary of posterior samples, with variable names as keys
    prior_dists : dict
        Dictionary of prior distributions, with variable names as keys
    true_values : dict, optional
        Dictionary of true parameter values, with variable names as keys
    var_names : list, optional
        List of variables to plot. If None, uses all keys in posterior_samples
    plot_var_names : dict or list, optional
        Dictionary mapping variable names to display names,
        or a list of display names in the same order as var_names
    figsize : tuple, optional
        Figure size, default is (12, 8)
    n_cols : int, optional
        Number of columns in the subplot grid
    point_estimate : str, optional
        Point estimate to show on plot ('mode', 'median', 'mean')
    save_fig_name : str, optional
        If provided, saves the figure to a file with the given name
    use_first_chain_only : bool, optional
        If True, use only the first MCMC chain to reduce computation cost
        and avoid potential issues with KDE estimation
    prior_range_extension : float, optional
        Factor to extend the range of the x-axis beyond the posterior's range
        to better visualize prior distributions (default: 3.0)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object
    axes : numpy.ndarray
        Array of axes objects for the subplots
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

    # Plot each variable
    for i, var in enumerate(var_names):
        if i < len(axes):
            ax = axes[i]

            # Get posterior samples for this variable
            post_samples = posterior_samples[var]

            # Check if we need to flatten or select only first chain
            if hasattr(post_samples, 'ndim') and post_samples.ndim > 1:
                if use_first_chain_only:
                    post_samples = post_samples[0]  # Use only first chain
                else:
                    post_samples = post_samples.flatten()  # Flatten all chains

            # Get prior distribution for this variable
            prior_dist = prior_dists[var]

            # Determine range for plotting
            x_min = np.percentile(post_samples, 0.1)
            x_max = np.percentile(post_samples, 99.9)

            # Get posterior range and extend it
            post_range = x_max - x_min
            x_min = x_min - prior_range_extension * post_range / 2
            x_max = x_max + prior_range_extension * post_range / 2

            # Handle special distributions with bounded support
            dist_name = prior_dist.__class__.__name__

            # For HalfNormal and other positive-only distributions
            if dist_name in ['HalfNormal', 'Gamma', 'Exponential',
                             'LogNormal', 'Weibull']:
                x_min = max(0, x_min)  # Ensure we don't go below zero

            # Very large range for heavy-tailed distributions
            if dist_name in ['Cauchy', 'StudentT']:
                x_min = x_min - prior_range_extension * post_range
                x_max = x_max + prior_range_extension * post_range

            # For bounded distributions, respect the bounds
            if hasattr(prior_dist, 'support'):
                try:
                    if hasattr(prior_dist.support, 'lower_bound'):
                        if prior_dist.support.lower_bound > -float('inf'):
                            x_min = max(x_min, prior_dist.support.lower_bound)
                    if hasattr(prior_dist.support, 'upper_bound'):
                        if prior_dist.support.upper_bound < float('inf'):
                            x_max = min(x_max, prior_dist.support.upper_bound)
                except (AttributeError, ValueError):
                    pass

            # Create x range for plotting based on the adjusted range
            x = np.linspace(x_min, x_max, 1000)

            # Plot prior distribution
            try:
                # For numpyro distributions with log_prob method
                prior_pdf = np.exp(prior_dist.log_prob(x))
                ax.plot(x, prior_pdf, label='Prior', linestyle='dashed',
                        color='coral', linewidth=1.5)
            except Exception as e:
                print(f"Warning: Could not plot prior for {var}: {e}")

            # Plot posterior KDE
            sns.kdeplot(data=post_samples, ax=ax, label='Posterior',
                        color='royalblue', fill=True,
                        alpha=0.2,
                        linewidth=1.5)

            # Calculate and plot posterior mode if requested
            if point_estimate == 'mode':
                kde = stats.gaussian_kde(post_samples)
                mode_idx = np.argmax(kde(x))
                mode_value = x[mode_idx]
                ax.axvline(mode_value, color='dodgerblue', linestyle='-.',
                           label='Posterior mode', linewidth=1.5)
            elif point_estimate == 'median':
                median_value = np.median(post_samples)
                ax.axvline(median_value, color='dodgerblue', linestyle='-.',
                           label='Posterior median', linewidth=1.5)
            elif point_estimate == 'mean':
                mean_value = np.mean(post_samples)
                ax.axvline(mean_value, color='dodgerblue', linestyle='-.',
                           label='Posterior mean', linewidth=1.5)

            # Plot true value if provided
            if true_values is not None and var in true_values:
                ax.axvline(true_values[var], color='maroon', linestyle='--',
                           label='True value', linewidth=2)
            # Set x-axis limits for positive-only distributions
            if dist_name in ['HalfNormal', 'Gamma', 'Exponential',
                             'LogNormal', 'Weibull']:
                ax.set_xlim(left=0)

            # Set plot labels and customize appearance
            ax.set_xlabel(var_display_names[var])
            ax.set_ylabel('Density')
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.tick_params(which='both', direction='in', top=True, right=True)
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend(frameon=True, framealpha=0.9)

    # Hide any unused axes
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)

    # Adjust layout
    plt.tight_layout()

    # Save figure if requested
    if save_fig_name is not None:
        main_dir = Path(__file__).resolve().parents[1]
        output_dir = main_dir / "outputs"
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / save_fig_name
        suffix = output_path.suffix.lower()
        if suffix in [".png", ".jpg", ".jpeg"]:
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
        else:
            plt.savefig(output_path, bbox_inches="tight")

    return fig, axes


def plot_posterior_predictive_stl(posterior_predictions, true_times,
                                  true_crack_lengths, observed_times,
                                  observed_crack_lengths, max_samples=50,
                                  figsize=(6.4, 4.8), save_fig_name=None):
    """
    Plot posterior predictive samples for crack growth
    with observed data points.

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
        Figure size as (width, height) in inches
    save_fig_name : str, optional
        If provided, the figure will be saved with this name

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    stats : dict
        Dictionary of computed statistics for further analysis
    """
    import numpy as np

    # Extract predicted crack lengths and
    # observations from posterior predictions
    predicted_crack_lengths = posterior_predictions['predicted_crack_lengths']

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Plot posterior predicted crack lengths (gray lines)
    for i in range(min(max_samples, predicted_crack_lengths.shape[0])):
        ax.plot(observed_times, predicted_crack_lengths[i], color='gray',
                alpha=0.2, zorder=1)

    # Plot posterior predictive mean
    mean_prediction = np.mean(predicted_crack_lengths, axis=0)
    ax.plot(observed_times, mean_prediction, color='darkblue', linewidth=2,
            label='Posterior Mean', zorder=4)

    # Plot the true trajectory
    ax.plot(true_times, true_crack_lengths, 'k-', linewidth=2,
            label='True trajectory', zorder=3)

    # Plot the observations
    ax.scatter(observed_times, observed_crack_lengths, color='red', s=60,
               label='Observations', zorder=5, edgecolors='white',
               linewidths=1)

    # Add 95% credible interval
    lower_ci = np.percentile(predicted_crack_lengths, 2.5, axis=0)
    upper_ci = np.percentile(predicted_crack_lengths, 97.5, axis=0)
    ax.fill_between(observed_times, lower_ci, upper_ci, color='blue',
                    alpha=0.15, label='95% Credible Interval',
                    zorder=2)

    # Set labels and title
    ax.set_xlabel('Time (years)', fontsize=12)
    ax.set_ylabel('Crack length (mm)', fontsize=12)
    ax.set_title('Posterior Predictive Check with ParisErdogan Model',
                 fontsize=14)

    # Set grid, limits, and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=min(true_crack_lengths)*0.95)
    ax.legend(fontsize=10)

    # Show error metrics
    rmse = np.sqrt(np.mean((mean_prediction - observed_crack_lengths)**2))
    ax.text(0.05, 0.95, f"RMSE: {rmse:.2f} mm", transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

    # Add minor ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in', top=True, right=True)

    # Save figure if filename provided
    if save_fig_name is not None:
        # Get the root directory of the project
        dir_path = Path(__file__).resolve().parents[1]
        # Create the path to save the figure
        save_path = dir_path / 'outputs' / save_fig_name
        # Raise an error if the directory does not exist
        if not save_path.parent.exists():
            raise FileNotFoundError(f"Directory {save_path.parent} \
                                    does not exist.")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.tight_layout()

    # Return computed statistics for additional analysis
    stats = {
        "rmse": rmse,
        "mean_prediction": mean_prediction,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
        "ci_width": upper_ci - lower_ci
    }

    return fig, ax, stats
