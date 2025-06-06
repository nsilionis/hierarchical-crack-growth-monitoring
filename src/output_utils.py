"""
Utilities for analyzing model outputs, calculating metrics,
and formatting results.

This module contains functions for:
- Calculating comparison metrics between true and inferred parameters
- Formatting result tables for display or publication
- Computing statistical measures for model assessment
- Summarizing and aggregating model results
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime


def calculate_parameter_error_metrics(
        true_values: Dict[str, float],
        inferred_values: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """
    Calculate error metrics between true and inferred parameter values.

    Parameters
    ----------
    true_values : Dict[str, float]
        Dictionary of true parameter values
    inferred_values : Dict[str, float]
        Dictionary of inferred parameter values

    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary of error metrics for each parameter:
        - 'absolute_error': Absolute error
        - 'relative_error': Relative error (percentage)
        - 'mape': Mean Absolute Percentage Error
        - 'rmse': Root Mean Square Error
    """
    metrics = {}

    # Ensure we only compare parameters present in both dictionaries
    common_params = set(true_values.keys()) & set(inferred_values.keys())

    for param in common_params:
        true_val = true_values[param]
        inf_val = inferred_values[param]

        # Skip if true value is zero to avoid division by zero
        if true_val == 0:
            continue

        # Calculate metrics
        abs_error = abs(inf_val - true_val)
        rel_error = 100 * abs_error / abs(true_val)  # Percentage
        mape = rel_error  # For a single value, MAPE equals relative error
        rmse = abs_error  # For a single value, RMSE equals absolute error

        metrics[param] = {
            'absolute_error': abs_error,
            'relative_error': rel_error,
            'mape': mape,
            'rmse': rmse,
        }

    return metrics


def calculate_component_metrics(
        component_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Calculate various metrics comparing true vs
    inferred parameters across components.

    Parameters
    ----------
    component_results : List[Dict[str, Any]]
        List of dictionaries containing results for each component.
        Each dict should have:
        - 'index': Component index
        - 'true_params': Dictionary with true parameter values
        - 'inferred_params': Dictionary with inferred parameter
        values and uncertainties

    Returns
    -------
    pd.DataFrame
        DataFrame containing comparison metrics for each component
    """
    # Initialize lists to store results
    components = []

    # Parameter tracking dictionaries
    param_data: Dict[str, Dict[str, List[Union[float, bool]]]] = {
        'logc': {'true': [], 'inferred': [], 'std': [], 'error': [],
                 'rel_error': []},
        'ds': {'true': [], 'inferred': [], 'std': [], 'error': [],
               'rel_error': [], 'present': False},
        'm': {'true': [], 'inferred': [], 'std': [], 'error': [],
              'rel_error': []},
        'noise_std': {'true': [], 'inferred': [], 'std': [], 'error': [],
                      'rel_error': []}
    }

    # Process each component
    for result in component_results:
        idx = result['index']
        components.append(idx)

        # Process each parameter (logc, ds if available, m, noise_std)
        for param_name in ['logc', 'ds', 'm', 'noise_std']:
            # Check if this parameter exists in both true and inferred params
            if param_name in result['true_params'] and \
                    param_name in result['inferred_params']:
                # Mark parameter as present if it's ds
                # (to indicate we should include it in output)
                if param_name == 'ds':
                    param_data[param_name]['present'] = True

                # Get values
                true_val = result['true_params'][param_name]
                inf_val = result['inferred_params'][param_name]
                std_val = result['inferred_params'].get(
                    f'{param_name}_sd', np.nan)

                # Calculate metrics
                error = true_val - inf_val
                # Use absolute value of true_val for logc which can be negative
                rel_error = 100 * error / (
                    abs(true_val) if param_name == 'logc' else true_val)

                # Store data
                param_data[param_name]['true'].append(true_val)
                param_data[param_name]['inferred'].append(inf_val)
                param_data[param_name]['std'].append(std_val)
                param_data[param_name]['error'].append(error)
                param_data[param_name]['rel_error'].append(rel_error)
            else:
                # If parameter is missing, add NaN placeholders
                if param_name != 'ds':  # Only add NaN for required params
                    param_data[param_name]['true'].append(np.nan)
                    param_data[param_name]['inferred'].append(np.nan)
                    param_data[param_name]['std'].append(np.nan)
                    param_data[param_name]['error'].append(np.nan)
                    param_data[param_name]['rel_error'].append(np.nan)

    # Create base data dictionary
    data: Dict[str, List] = {'Component': components}

    # Define parameter order ensuring ds comes after logc if present
    param_order: List[str] = ['logc']
    if param_data['ds']['present']:
        param_order.append('ds')
    param_order.extend(['m', 'noise_std'])

    # Add columns for each parameter in the defined order
    for param_name in param_order:
        # Skip ds if not present
        if param_name == 'ds' and not param_data[param_name]['present']:
            continue

        # Add columns for this parameter
        data[f'{param_name} (True)'] = param_data[param_name]['true']
        data[f'{param_name} (Inferred)'] = param_data[param_name]['inferred']
        data[f'{param_name} (Std)'] = param_data[param_name]['std']
        data[f'{param_name} (Error)'] = param_data[param_name]['error']
        data[f'{param_name} (% Error)'] = param_data[param_name]['rel_error']

    return pd.DataFrame(data)


def format_comparison_table(df: pd.DataFrame, precision: int = 4
                            ) -> pd.DataFrame:
    """
    Format the comparison table for better presentation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from calculate_component_metrics
    precision : int, optional
        Decimal precision for floating point values

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame
    """
    # Create a copy to avoid modifying the original
    formatted_df = df.copy()

    # Determine which parameters exist in this DataFrame
    param_prefixes = ['logC']
    if 'ds (True)' in df.columns:
        param_prefixes.append('ds')
    param_prefixes.extend(['m', 'Noise'])

    # Format each group of columns
    for prefix in param_prefixes:
        # Format basic values to specified precision
        if f"{prefix} (True)" in formatted_df.columns:
            formatted_df[f"{prefix} (True)"] = formatted_df[
                f"{prefix} (True)"].map(
                lambda x: f"{x:.{precision}f}" if pd.notnull(x) else "-")

        if f"{prefix} (Inferred)" in formatted_df.columns:
            formatted_df[f"{prefix} (Inferred)"] = formatted_df[
                f"{prefix} (Inferred)"].map(
                lambda x: f"{x:.{precision}f}" if pd.notnull(x) else "-")

        if f"{prefix} (Std)" in formatted_df.columns:
            formatted_df[f"{prefix} (Std)"] = formatted_df[
                f"{prefix} (Std)"].map(
                lambda x: f"{x:.{precision}f}" if pd.notnull(x) else "-")

        # Format error with sign indicator
        if f"{prefix} (Error)" in formatted_df.columns:
            formatted_df[
                f"{prefix} (Error)"] = formatted_df[f"{prefix} (Error)"].map(
                lambda x: f"+{x:.{precision}f}" if pd.notnull(x) and x > 0 else
                (f"{x:.{precision}f}" if pd.notnull(x) else "-"))

        # Format percentage with sign and percentage symbol
        if f"{prefix} (% Error)" in formatted_df.columns:
            formatted_df[
                f"{prefix} (% Error)"] = formatted_df[f"{prefix} (% Error)"
                                                      ].map(
                lambda x: f"+{x:.{precision}f}%" if pd.notnull(x) and x > 0
                else (f"{x:.{precision}f}%" if pd.notnull(x) else "-"))

    # Format component column as integer if it contains numeric values
    if all(isinstance(x, (int, float)) for x in formatted_df['Component']):
        formatted_df['Component'] = formatted_df['Component'].map(
            lambda x: f"{int(x)+1}")

    return formatted_df


def summarize_posterior(
        posterior_samples: Dict[str, np.ndarray],
        var_names: Optional[List[str]] = None,
        quantiles: List[float] = [0.025, 0.25, 0.5, 0.75, 0.975],
        true_values: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Generate a summary of posterior distributions.

    Parameters
    ----------
    posterior_samples : Dict[str, np.ndarray]
        Dictionary of posterior samples
    var_names : List[str], optional
        List of variables to include in summary, if None, uses all variables
    quantiles : List[float], optional
        Quantiles to compute
    true_values : Dict[str, float], optional
        Dictionary of true parameter values for comparison

    Returns
    -------
    pd.DataFrame
        Summary statistics of posterior distributions
    """
    # If var_names not provided, use all keys in posterior_samples
    if var_names is None:
        var_names = list(posterior_samples.keys())

    # Initialize results
    results = []

    for var in var_names:
        if var not in posterior_samples:
            continue

        # Get samples for this variable
        samples = posterior_samples[var]

        # If samples has multiple dimensions (e.g., MCMC chains), flatten it
        if samples.ndim > 1:
            samples = samples.flatten()

        # Calculate statistics
        mean_val = np.mean(samples)
        std_val = np.std(samples)
        q_vals = np.quantile(samples, quantiles)

        # Find maximum density (mode) using KDE
        try:
            from scipy import stats
            kde = stats.gaussian_kde(samples)
            x = np.linspace(min(samples), max(samples), 1000)
            mode_idx = np.argmax(kde(x))
            mode_val = x[mode_idx]
        except Exception:
            mode_val = np.nan

        # Calculate error if true value provided
        if true_values is not None and var in true_values:
            true_val = true_values[var]
            abs_error = abs(mean_val - true_val)
            rel_error = 100 * abs_error / abs(true_val) \
                if true_val != 0 else np.nan
        else:
            true_val = np.nan
            abs_error = np.nan
            rel_error = np.nan

        # Create row
        row = {
            'Variable': var,
            'Mean': mean_val,
            'Std': std_val,
            'Mode': mode_val,
        }

        # Add quantiles
        for i, q in enumerate(quantiles):
            row[f'{q*100:.1f}%'] = q_vals[i]

        # Add error metrics if true value provided
        if true_values is not None and var in true_values:
            row['True Value'] = true_val
            row['Abs Error'] = abs_error
            row['Rel Error (%)'] = rel_error

        results.append(row)

    return pd.DataFrame(results)


def save_results_table(df, filename_prefix='results', output_dir=None):
    """
    Save the results table to a CSV file with timestamp.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to save
    filename_prefix : str, optional
        Prefix for the filename
    output_dir : str or Path, optional
        Directory to save the file. If None, uses project's results directory.

    Returns
    -------
    str
        Path to the saved file
    """
    if output_dir is None:
        # Get the project root directory (two levels up from this file)
        output_dir = Path(__file__).resolve().parents[1] / 'results'
    else:
        output_dir = Path(output_dir)

    # Create the directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{filename_prefix}_{timestamp}.csv"
    file_path = output_dir / filename

    # Save to file
    df.to_csv(file_path, index=False)

    return str(file_path)


def calculate_prediction_metrics(
        observed: np.ndarray,
        predicted: np.ndarray) -> Dict[str, float]:
    """
    Calculate various error metrics between observed and predicted values.

    Parameters
    ----------
    observed : array-like
        Observed (true) values
    predicted : array-like
        Predicted values

    Returns
    -------
    dict
        Dictionary containing various error metrics:
        - 'rmse': Root Mean Squared Error
        - 'mae': Mean Absolute Error
        - 'r_squared': R-squared coefficient of determination
        - 'mape': Mean Absolute Percentage Error
    """
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score
    )

    # Convert to numpy arrays
    observed = np.array(observed)
    predicted = np.array(predicted)

    # Calculate basic metrics
    rmse = np.sqrt(mean_squared_error(observed, predicted))
    mae = mean_absolute_error(observed, predicted)
    r_squared = r2_score(observed, predicted)

    # Calculate Mean Absolute Percentage Error (MAPE)
    # Avoid division by zero
    mask = observed != 0
    if np.any(mask):
        mape = np.mean(np.abs(
            (observed[mask] - predicted[mask]) / observed[mask]
        )) * 100
    else:
        mape = np.nan

    return {
        'rmse': rmse,
        'mae': mae,
        'r_squared': r_squared,
        'mape': mape
    }
