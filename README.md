# Hierarchical Bayesian Models for Stochastic Crack Growth Monitoring

This project implements hierarchical Bayesian models for analyzing and monitoring stochastic crack growth in materials using Paris' law and probabilistic programming.

## Project Structure

- `src/` - Source code for model definitions and utilities.
- `notebooks/` - Jupyter notebooks for experiments and analysis.
- `data/` - Input datasets (not included in repo).
- `figures/` - Generated plots and figures.

## Installation

```sh
pip install -r requirements.txt
```

## Usage

Open and run the main notebook:

```sh
jupyter notebook notebooks/HBM_for_SCG.ipynb
```

## Requirements

- Python 3.8+
- jax
- numpyro
- matplotlib
- seaborn
- scipy
- tqdm
- arviz

## References

- [Paris' Law](https://en.wikipedia.org/wiki/Paris%27_law)
- [NumPyro Documentation](https://num.pyro.ai/en/stable/)
