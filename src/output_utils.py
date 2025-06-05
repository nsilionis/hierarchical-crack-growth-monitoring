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
from typing import Dict, List, Optional, Any


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
        component_results: List[Dict[str, Any]],
        true_noise_std: Optional[float] = None) -> pd.DataFrame:
    """
    Calculate metrics for multiple components and format as a DataFrame.

    Parameters
    ----------
    component_results : List[Dict[str, Any]]
        List of dictionaries containing inference results for each component
        Each dictionary should have:
        - 'index': Component index
        - 'true_params': Dict of true parameter values
        - 'inferred_params': Dict of inferred parameter values
    true_noise_std : float, optional
        True noise standard deviation value, if not included in true_params

    Returns
    -------
    pd.DataFrame
        DataFrame with comparison metrics for all components
    """
    rows = []

    for i, res in enumerate(component_results):
        # Extract parameters
        true_params = res['true_params']
        inferred_params = res['inferred_params']

        # Add noise_std to true_params if provided separately
        if true_noise_std is not None and 'noise_std' not in true_params:
            true_params['noise_std'] = true_noise_std

        # Create row with component index
        row = {'Component': i+1}

        # Add true and inferred values for each parameter
        for param in ['logc', 'm', 'noise_std']:
            if param in true_params:
                row[f'True {param}'] = true_params[param]

            if param in inferred_params:
                row[f'Inferred {param}'] = inferred_params[param]

                # Add standard deviation if available
                if f'{param}_sd' in inferred_params:
                    row[f'{param} SD'] = inferred_params[f'{param}_sd']

        # Calculate error metrics
        for param in ['logc', 'm', 'noise_std']:
            if param in true_params and param in inferred_params:
                # Calculate MAPE
                mape = 100 * abs(inferred_params[param] - true_params[param]) \
                    / abs(true_params[param])
                row[f'{param} MAPE (%)'] = mape

                # Calculate RMSE
                rmse = np.sqrt(
                    (inferred_params[param] - true_params[param]
                     )**2)
                row[f'{param} RMSE'] = rmse

        rows.append(row)

    # Create DataFrame from rows
    df = pd.DataFrame(rows)

    # Calculate average metrics row
    if len(rows) > 0:
        avg_row = {'Component': 'Average'}

        # Find columns with MAPE and RMSE
        mape_cols = [col for col in df.columns if 'MAPE' in col]
        rmse_cols = [col for col in df.columns if 'RMSE' in col]

        # Calculate averages
        for col in mape_cols + rmse_cols:
            avg_row[col] = df[col].mean()

        # Append average row
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    return df


def format_comparison_table(
        df: pd.DataFrame,
        precision: int = 4,
        include_avg_row: bool = True,
        style: Optional[str] = None) -> pd.DataFrame:
    """
    Format a comparison table for display or export.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with comparison metrics
    precision : int, optional
        Number of decimal places for displaying values
    include_avg_row : bool, optional
        Whether to include average metrics row
    style : str, optional
        Styling format ('latex', 'html', None)

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame ready for display or export
    """
    # Make a copy to avoid modifying the original
    formatted_df = df.copy()

    # Remove average row if not requested
    if not include_avg_row and 'Average' in formatted_df['Component'].values:
        formatted_df = formatted_df[formatted_df['Component'] != 'Average']

    # Set display precision
    pd.set_option('display.precision', precision)

    # Apply styling if requested
    if style == 'latex':
        return formatted_df.style.format(precision=precision).to_latex()
    elif style == 'html':
        return formatted_df.style.format(precision=precision).to_html()
    else:
        return formatted_df


def save_results_table(
        df: pd.DataFrame,
        filename: str,
        output_dir: Optional[str] = None,
        format: str = 'csv') -> str:
    """
    Save a results table to a file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    filename : str
        Name of the output file
    output_dir : str, optional
        Output directory, if None, uses project's 'outputs' directory
    format : str, optional
        Output format ('csv', 'excel', 'latex', 'html', 'markdown')

    Returns
    -------
    str
        Path to the saved file
    """
    # Set default output directory if not provided
    if output_dir is None:
        root_dir = Path(__file__).resolve().parents[1]
        output_dir = root_dir / "outputs"
    else:
        output_dir = Path(output_dir)

    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add file extension if not included
    if '.' not in filename:
        if format == 'csv':
            filename += '.csv'
        elif format == 'excel':
            filename += '.xlsx'
        elif format == 'latex':
            filename += '.tex'
        elif format == 'html':
            filename += '.html'
        elif format == 'markdown':
            filename += '.md'

    # Create full path
    filepath = output_dir / filename

    # Save based on format
    if format == 'csv':
        df.to_csv(filepath, index=False)
    elif format == 'excel':
        df.to_excel(filepath, index=False)
    elif format == 'latex':
        with open(filepath, 'w') as f:
            f.write(df.to_latex(index=False))
    elif format == 'html':
        with open(filepath, 'w') as f:
            f.write(df.to_html(index=False))
    elif format == 'markdown':
        with open(filepath, 'w') as f:
            f.write(df.to_markdown(index=False))
    else:
        raise ValueError(f"Unsupported output format: {format}")

    return str(filepath)


def calculate_prediction_metrics(
        true_values: np.ndarray,
        predicted_values: np.ndarray) -> Dict[str, float]:
    """
    Calculate metrics for comparing predictions with true values.

    Parameters
    ----------
    true_values : np.ndarray
        Array of true values
    predicted_values : np.ndarray
        Array of predicted values

    Returns
    -------
    Dict[str, float]
        Dictionary of metrics
    """
    # Ensure arrays have the same shape
    if true_values.shape != predicted_values.shape:
        raise ValueError("true_values and predicted_values \
                         must have the same shape")

    # Calculate metrics
    mse = np.mean((predicted_values - true_values)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predicted_values - true_values))

    # Calculate MAPE avoiding division by zero
    nonzero_mask = true_values != 0
    if np.any(nonzero_mask):
        mape = 100 * np.mean(np.abs((true_values[nonzero_mask] -
                                    predicted_values[nonzero_mask]) /
                                    true_values[nonzero_mask]))
    else:
        mape = np.nan

    # Calculate R-squared
    ss_total = np.sum((true_values - np.mean(true_values))**2)
    ss_residual = np.sum((true_values - predicted_values)**2)
    r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else np.nan

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r_squared': r_squared
    }


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
