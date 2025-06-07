"""
Utilities for analyzing Bayesian inference outputs and calculating
posterior evaluation metrics.

This module contains functions for:
- Calculating error metrics between true values and posterior samples
- Computing posterior summary statistics with HDI
- Creating comprehensive comparison tables for multiple components
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from numpyro.diagnostics import hpdi
from pathlib import Path
from datetime import datetime


def calculate_posterior_errors(
    true_value: float,
    posterior_samples: np.ndarray
) -> Dict[str, float]:
    """
    Calculate RMSE and MAPE between a true parameter
    value and posterior samples.

    Parameters
    ----------
    true_value : float
        True parameter value
    posterior_samples : np.ndarray
        Array of posterior samples for the parameter

    Returns
    -------
    Dict[str, float]
        Dictionary containing 'rmse' and 'mape'
    """
    # Flatten samples in case they come from multiple chains
    samples = np.asarray(posterior_samples).flatten()

    # Calculate RMSE
    squared_errors = (samples - true_value) ** 2
    rmse = float(np.sqrt(np.mean(squared_errors)))

    # Calculate MAPE
    if true_value != 0:
        abs_percentage_errors = 100 * np.abs(samples - true_value) \
            / abs(true_value)
        mape = float(np.mean(abs_percentage_errors))
    else:
        mape = np.inf

    return {
        'rmse': rmse,
        'mape': mape
    }


def calculate_posterior_summary(
    posterior_samples: np.ndarray,
    prob: float = 0.95
) -> Dict[str, float]:
    """
    Calculate summary statistics for posterior samples.
    This includes mean, standard deviation, and highest posterior
    density interval (HDI).

    Parameters
    ----------
    posterior_samples : np.ndarray
        Array of posterior samples
    prob : float, optional
        Probability mass for HDI calculation (default 0.95)

    Returns
    -------
    Dict[str, float]
        Dictionary containing mean, std, hdi_lower, hdi_upper
    """
    # Flatten samples in case they come from multiple chains
    samples = np.asarray(posterior_samples).flatten()

    # Calculate basic statistics
    mean_val = float(np.mean(samples))
    std_val = float(np.std(samples))

    # Calculate HDI using NumPyro
    hdi_bounds = hpdi(samples, prob=prob)
    hdi_lower = float(hdi_bounds[0])
    hdi_upper = float(hdi_bounds[1])

    return {
        'mean': mean_val,
        'std': std_val,
        'hdi_lower': hdi_lower,
        'hdi_upper': hdi_upper
    }


def create_comparison_summary(
    component_results: List[Dict[str, Any]],
    prob: float = 0.95
) -> pd.DataFrame:
    """
    Create a comprehensive comparison DataFrame from component results.

    This function processes multiple component inference results and creates
    a summary table with target values, posterior statistics,
    and error metrics.

    Parameters
    ----------
    component_results : List[Dict[str, Any]]
        List of dictionaries containing results for each component.
        Each dictionary should have 'true_params' and 'inferred_params' keys.
    prob : float, optional
        Probability mass for HDI calculation (default 0.95)

    Returns
    -------
    pd.DataFrame
        Comprehensive summary table with columns for each parameter showing:
        target value, posterior mean, std, HDI bounds, RMSE, and MAPE
    """
    if not component_results:
        return pd.DataFrame()

    # Auto-detect available parameters
    first_result = component_results[0]
    true_params = set(first_result.get('true_params', {}).keys())
    inferred_params = set(first_result.get('inferred_params', {}).keys())
    available_params = sorted(list(true_params & inferred_params))

    # Initialize data for DataFrame
    data = {'Component': []}

    # Create columns for each parameter
    for param in available_params:
        data[f'{param}_target'] = []
        data[f'{param}_mean'] = []
        data[f'{param}_std'] = []
        data[f'{param}_hdi_lower'] = []
        data[f'{param}_hdi_upper'] = []
        data[f'{param}_rmse'] = []
        data[f'{param}_mape'] = []

    # Process each component
    for i, result in enumerate(component_results):
        data['Component'].append(f"Component {i+1}")

        true_params = result.get('true_params', {})
        inferred_params = result.get('inferred_params', {})

        for param in available_params:
            if param in true_params and param in inferred_params:
                true_val = true_params[param]
                posterior_samples = inferred_params[param]

                # Calculate summary statistics
                summary_stats = calculate_posterior_summary(
                    posterior_samples, prob=prob
                    )

                # Calculate error metrics
                error_metrics = calculate_posterior_errors(
                    true_val, posterior_samples
                    )

                # Store all metrics
                data[f'{param}_target'].append(true_val)
                data[f'{param}_mean'].append(summary_stats['mean'])
                data[f'{param}_std'].append(summary_stats['std'])
                data[f'{param}_hdi_lower'].append(summary_stats['hdi_lower'])
                data[f'{param}_hdi_upper'].append(summary_stats['hdi_upper'])
                data[f'{param}_rmse'].append(error_metrics['rmse'])
                data[f'{param}_mape'].append(error_metrics['mape'])
            else:
                # Handle missing parameters with NaN
                for suffix in ['_target', '_mean', '_std', '_hdi_lower',
                               '_hdi_upper', '_rmse', '_mape']:
                    data[f'{param}{suffix}'].append(np.nan)

    return pd.DataFrame(data)


def save_comparison_summary(
    df: pd.DataFrame,
    filename: str = None,
    format: str = 'csv'
) -> str:
    """
    Save comparison summary DataFrame to multiple formats optimized
    for presentations and documents.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    filename : str, optional
        Custom filename. If None, generates timestamped filename
    format : str, optional
        Output format. Options: 'csv', 'excel', 'html' (default 'csv')

    Returns
    -------
    str
        Path to saved file
    """
    # Validate format
    valid_formats = ['csv', 'excel', 'html']
    if format not in valid_formats:
        raise ValueError(f"Format must be one of {valid_formats}")

    # Set default output directory
    main_dir = Path(__file__).resolve().parents[1]
    output_dir = main_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = f"bayesian_comparison_summary_{timestamp}"
    else:
        # Remove existing extension from custom filename
        base_name = Path(filename).stem

    # Set appropriate extension based on format
    extensions = {
        'csv': '.csv',
        'excel': '.xlsx',
        'html': '.html'
    }

    filename_with_ext = base_name + extensions[format]
    file_path = output_dir / filename_with_ext

    # Save DataFrame based on format
    if format == 'csv':
        df.to_csv(file_path, index=False)

    elif format == 'excel':
        # Save with formatting for better presentation
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Bayesian_Comparison', index=False)
            # Get the workbook and worksheet for formatting
            worksheet = writer.sheets['Bayesian_Comparison']
            # Auto-adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column = [cell for cell in column]
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except (AttributeError, TypeError):
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[
                    column[0].column_letter].width = adjusted_width

    elif format == 'html':
        # Save as HTML with styling for better presentation
        html_str = df.to_html(
            index=False,
            float_format='{:.4f}'.format,
            table_id='bayesian-comparison',
            classes='table table-striped table-hover',
            escape=False
        )
        # Add some basic CSS styling
        styled_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Bayesian Inference Comparison Summary</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .table {{ border-collapse: collapse; width: 100%; }}
        .table th, .table td {{ padding: 8px 12px; text-align: center;
        border: 1px solid #ddd; }}
        .table th {{ background-color: #f2f2f2; font-weight: bold; }}
        .table-striped tbody tr:nth-child(odd) {{ background-color: #f9f9f9; }}
        .table-hover tbody tr:hover {{ background-color: #e9e9e9; }}
        h1 {{ color: #333; text-align: center; }}
    </style>
</head>
<body>
    <h1>Bayesian Inference Comparison Summary</h1>
    {html_str}
</body>
</html>
"""
        with open(file_path, 'w') as f:
            f.write(styled_html)

    return str(file_path)
