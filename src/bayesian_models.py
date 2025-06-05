import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az
from typing import Dict, Optional, Any
from src.crack_growth_models import ParisErdogan


class STLBayesianModel:
    """
    Single-task learner Bayesian model for Paris law parameters.

    This class encapsulates a Bayesian model for inferring Paris law parameters
    from crack growth data. It assumes the data comes from a single component.
    """

    def __init__(self,
                 priors: Optional[Dict[str, dist.Distribution]] = None,
                 crack_growth_data: Optional[Dict[str, Any]] = None):
        """
        Initialize the Bayesian model with priors and data.

        Parameters
        ----------
        priors : dict, optional
            Dictionary of prior distributions for parameters
        crack_growth_data : dict, optional
            Dictionary containing the crack growth data
        """
        # Default priors if none provided
        if priors is None:
            self.priors = {
                "logc": dist.Normal(-30.0, 2.0),
                "m": dist.Normal(3.0, 0.5),
                "ds": dist.Gamma(5.0, 0.3),
                "noise_std": dist.HalfNormal(2.0)
            }
        else:
            self.priors = priors

        # Store data
        self.crack_growth_data = crack_growth_data

        # Initialize storage for posterior samples
        self.mcmc = None
        self.posterior_samples = None

    def model(self, component_idx=0, Y=1.12, navg=None):
        """
        Bayesian model for Paris law parameters.

        Parameters
        ----------
        component_idx : int, optional
            Index of the component to model
        Y : float, optional
            Geometry factor for SIF calculation
        navg : float, optional
            Average cycles per year. If None, uses a default value

        Returns
        -------
        None
        """
        # Prior distributions
        logc = numpyro.sample("logc", self.priors["logc"])
        m = numpyro.sample("m", self.priors["m"])
        ds = numpyro.sample("ds", self.priors["ds"])
        noise_std = numpyro.sample("noise_std", self.priors["noise_std"])

        # Default navg if not provided
        if navg is None:
            navg = 2.8e6

        # Extract data for this component
        times = self.crack_growth_data["times"][component_idx]
        data = self.crack_growth_data["noisy_crack_lengths"][component_idx]

        # Initial crack length (first observation)
        init_crack = data[0]

        # Create Paris-Erdogan model instance
        paris = ParisErdogan(
            logc=logc,
            m=m,
            ds=ds,
            navg=navg,
            a0=init_crack,
            Y=Y,
            t=times
        )

        # Initialize array for predicted crack lengths
        crack_lengths = jnp.zeros(len(times))
        crack_lengths = crack_lengths.at[0].set(init_crack)

        # Generate crack growth trajectory using the Paris-Erdogan model
        for i in range(1, len(times)):
            crack_lengths = crack_lengths.at[i].set(
                paris.state_eq(crack_lengths[i-1], times[i-1])
            )

        # Likelihood for observations
        numpyro.sample(
            "obs",
            dist.Normal(crack_lengths, noise_std),
            obs=data
        )

        # Store predicted crack lengths for posterior predictive checks
        numpyro.deterministic("predicted_crack_lengths", crack_lengths[1:])

    def run_inference(self,
                      component_idx: int = 0,
                      Y: float = 1.12,
                      navg: Optional[float] = None,
                      num_warmup: int = 1000,
                      num_samples: int = 1000,
                      num_chains: int = 4,
                      progress_bar: bool = True) -> Dict:
        """
        Run MCMC inference for the model.

        Parameters
        ----------
        component_idx : int, optional
            Index of the component to model
        Y : float, optional
            Geometry factor for SIF calculation
        navg : float, optional
            Average cycles per year. If None, uses a default value
        num_warmup : int, optional
            Number of warmup steps for MCMC
        num_samples : int, optional
            Number of samples to draw after warmup
        num_chains : int, optional
            Number of MCMC chains to run
        progress_bar : bool, optional
            Whether to show a progress bar during sampling

        Returns
        -------
        dict
            Dictionary containing MCMC results and summary
        """
        # Set up MCMC
        nuts_kernel = NUTS(self.model,
                           target_accept_prob=0.9)
        self.mcmc = MCMC(
            nuts_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=progress_bar
        )

        # Run inference
        rng_key = jr.PRNGKey(0)
        self.mcmc.run(
            rng_key,
            component_idx=component_idx,
            Y=Y,
            navg=navg
        )

        # Extract and store results
        self.posterior_samples = self.mcmc.get_samples(group_by_chain=True)

        # Create summary statistics
        summary = az.summary(az.from_numpyro(self.mcmc), round_to=3)

        # Return results and summary
        return {
            "mcmc": self.mcmc,
            "samples": self.posterior_samples,
            "summary": summary
        }

    def summarise_posterior(self, print_summary: bool = True
                            ) -> az.InferenceData:
        """
        Create a summary of the posterior distribution.

        Parameters
        ----------
        print_summary : bool, optional
            Whether to print the summary to stdout

        Returns
        -------
        arviz.InferenceData
            ArviZ InferenceData object containing the posterior data
        """
        if self.mcmc is None:
            raise ValueError("No MCMC results available. \
                             Please run inference first.")

        # Convert to ArviZ format
        inference_data = az.from_numpyro(self.mcmc)

        if print_summary:
            print(az.summary(inference_data, round_to=3))

        return inference_data

    def generate_predictions(self,
                             num_samples: int = 1000,
                             component_idx: int = 0,
                             Y: float = 1.12,
                             navg: Optional[float] = None,
                             random_seed: int = 42) -> Dict[str, jnp.ndarray]:
        """
        Generate posterior predictions for crack growth.

        Parameters
        ----------
        num_samples : int, optional
            Number of posterior samples to use for prediction
        component_idx : int, optional
            Index of the component to predict
        Y : float, optional
            Geometry factor for SIF calculation
        navg : float, optional
            Average cycles per year. If None, uses default value 2.8e6
        random_seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        dict
            Dictionary of predictions
        """
        if self.posterior_samples is None:
            raise ValueError("No posterior samples available. \
                             Please run inference first.")

        # Set up the RNG key
        rng_key = jr.PRNGKey(random_seed)

        # Extract data for this component to get
        # the initial crack length and times
        times = self.crack_growth_data["times"][component_idx]
        data = self.crack_growth_data["noisy_crack_lengths"][component_idx]
        init_crack = data[0]

        # Create a custom prediction function that
        # manually implements the Paris model
        # to avoid the broadcasting issues
        def predict_fn(samples):
            # Extract parameters from samples
            logc = samples["logc"]
            m = samples["m"]
            ds = samples["ds"]
            noise_std = samples["noise_std"]

            if navg is None:
                navg_value = 2.8e6
            else:
                navg_value = navg

            # Initialize array for each sample's prediction
            n_times = len(times)
            crack_lengths = jnp.zeros(n_times)
            crack_lengths = crack_lengths.at[0].set(init_crack)

            # Create Paris-Erdogan model
            paris = ParisErdogan(
                logc=logc,
                m=m,
                ds=ds,
                navg=navg_value,
                a0=init_crack,
                Y=Y,
                t=times
            )

            # Generate crack growth trajectory
            for i in range(1, n_times):
                crack_lengths = crack_lengths.at[i].set(
                    paris.state_eq(crack_lengths[i-1], times[i-1])
                )

            # Store full crack length array including initial point
            return {
                "predicted_crack_lengths": crack_lengths,
                "obs": crack_lengths + noise_std * jr.normal(
                    rng_key, crack_lengths.shape)
            }

        # Flatten chains for prediction
        flat_samples = {}
        for k, v in self.posterior_samples.items():
            flat_samples[k] = v.reshape(-1)

        # Select random subset of samples if requested
        n_available = len(flat_samples["logc"])
        if num_samples < n_available:
            indices = jr.choice(rng_key, n_available, (num_samples,),
                                replace=False)
            for k in flat_samples.keys():
                flat_samples[k] = flat_samples[k][indices]

        # Run prediction for each posterior sample
        all_predictions = []
        for i in range(min(num_samples, n_available)):
            sample = {k: v[i] for k, v in flat_samples.items()}
            all_predictions.append(predict_fn(sample))

        # Combine results
        predictions = {
            "predicted_crack_lengths": jnp.stack([p["predicted_crack_lengths"]
                                                  for p in all_predictions]),
            "obs": jnp.stack([p["obs"] for p in all_predictions])
        }

        return predictions
