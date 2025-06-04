import jax.numpy as jnp
import jax.random as jr
# import jax.lax as lax
import numpy as np
# import tqdm
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
# from numpyro.diagnostics import hpdi
import arviz as az
from typing import Dict, List, Optional
from src.crack_growth_models import ParisErdogan
from src.predictive_models import ObservationModel, IdentityObservation


class STLBayesianModel:
    """
    This class implements a single-task Bayesian model using NumPyro.
    The goal is to learn the parameters of a crack growth model
    using crack growth data from a specific cracked component.
    Hence, the model is referred to as a single-task model.

    This model assumes constant load conditions throughout
    the crack growth process. The model treats material parameters and
    loading conditions as random variables which can be inferred
    from the observed crack growth data.
    """

    # Required parameter names for the model to function properly
    REQUIRED_PRIORS = ["logc", "m", "ds", "noise_std"]

    def __init__(self, priors: Dict[str, dist.Distribution],
                 crack_growth_data: Dict[str, List[np.ndarray]],
                 observation_model: Optional[ObservationModel] = None):
        """
        Initialize the Bayesian model with priors and
        crack growth data.

        Parameters
        ----------
        priors : Dict[str, dist.Distribution]
            Dictionary containing the prior distributions for model parameters.
            Must include the following keys:
            - "logc": Prior for natural log of Paris law parameter C
            - "m": Prior for Paris law exponent m
            - "ds": Prior for stress range
            - "noise_std": Prior for observation noise standard deviation

            May optionally include:
            - "navg": Prior for average cycles per time unit

        crack_growth_data : Dict[str, List[np.ndarray]]
            Dictionary containing the crack growth data for each component.
            Typically outputs from
            CrackObservationGenerator.create_observations().

        observation_model : ObservationModel, optional
            Model for transforming simulated crack lengths to observations.
            If None, uses IdentityObservation which returns values unchanged.

        Raises
        ------
        ValueError
            If any required prior is missing from the priors dictionary
        TypeError
            If any value in the priors dictionary is not a NumPyro distribution
        """
        # Validate required priors are present
        missing_priors = [prior for prior in self.REQUIRED_PRIORS
                          if prior not in priors]
        if missing_priors:
            raise ValueError(f"Missing required priors: \
                             {', '.join(missing_priors)}")

        # Validate all priors are NumPyro distributions
        invalid_priors = [k for k, v in priors.items()
                          if not isinstance(v, dist.Distribution)]
        if invalid_priors:
            raise TypeError(f"The following priors are not NumPyro \
                            distributions: {', '.join(invalid_priors)}")

        self.priors = priors
        self.crack_growth_data = crack_growth_data
        self.mcmc = None
        self.posterior_samples = None

        # Set observation model (use identity if none provided)
        self.observation_model = observation_model or IdentityObservation()

        # Validate that crack_growth_data contains the necessary data
        required_data = ["times", "crack_lengths", "initial_crack_length"]
        missing_data = [field for field in required_data
                        if field not in crack_growth_data]
        if missing_data:
            raise ValueError(f"Missing required data fields \
                             in crack_growth_data: {', '.join(missing_data)}")

    def _crack_growth_step(self, state_tuple, t_idx):
        """
        Single step of crack growth for use with jax.lax.scan

        Parameters
        ----------
        state_tuple : tuple
            Tuple containing (crack_length, time)
        t_idx : int
            Index in the time array (used to get next time value)

        Returns
        -------
        tuple
            Updated (crack_length, time)
        """
        crack_state, t = state_tuple
        next_t = self.times[t_idx + 1]  # Get the next time point
        next_crack = self.growth_model.state_eq(crack_state, t)
        return (next_crack, next_t)

    def model(self, component_idx: int = 0, Y: float = 1.12,
              navg: Optional[float] = None) -> None:
        """
        Define the probabilistic model for crack growth
        using data from crack_growth_data.

        This function implements Paris law with NumPyro primitives, setting up:
        1. Prior distributions for all model parameters
        2. Forward simulation of crack growth based on Paris law
        3. Likelihood model for observations given the simulated crack growth

        Parameters
        ----------
        component_idx : int, optional
            Index of the component to model in the crack_growth_data.
            Default is 0.
        Y : float, optional
            Geometry factor for stress intensity factor calculation.
            Default is 1.12.
        navg : float, optional
            Average number of cycles per time unit.
            If None, it will be sampled from the prior, which must be provided.

        Notes
        -----
        The model samples the following parameters from priors:
        - logc: Natural logarithm of Paris law parameter C
        - m: Paris law exponent parameter
        - ds: Stress range
        - noise_std: Standard deviation of the observation noise
        - navg: Average number of cycles (if not provided as parameter)
        """
        # Get the data for the specified component
        self.times = self.crack_growth_data["times"][component_idx]
        observations = self.crack_growth_data["crack_lengths"][component_idx]
        a0 = self.crack_growth_data["initial_crack_length"][component_idx]

        # Sample model parameters from priors
        logc = numpyro.sample("logc", self.priors["logc"])
        m = numpyro.sample("m", self.priors["m"])
        ds = numpyro.sample("ds", self.priors["ds"])
        noise_std = numpyro.sample("noise_std", self.priors["noise_std"])

        # If navg is not provided, sample it from the prior
        if navg is None:
            if "navg" not in self.priors:
                if "avg_cycles" in self.crack_growth_data:
                    # Use the average cycles from the data if available
                    navg = self.crack_growth_data["avg_cycles"]
                    [component_idx][0]
                else:
                    raise ValueError("navg parameter not provided, "
                                     "no prior specified, "
                                     "and no avg_cycles in data")
            else:
                navg = numpyro.sample("navg", self.priors["navg"])

        # Create the crack growth model instance with parameters
        model_params = {
            'logc': logc,
            'm': m,
            'ds': ds,
            'navg': navg,
            'a0': a0,
            'Y': Y,
            't': self.times
        }
        self.growth_model = ParisErdogan(**model_params)

        # Create array to store crack lengths
        n_times = len(self.times)

        # Initialize the first crack length with a0
        crack_lengths = jnp.zeros(n_times)
        crack_lengths = crack_lengths.at[0].set(a0)

        # Use simple JAX operations for each time step
        for i in range(1, n_times):
            prev_crack = crack_lengths[i-1]
            dt = self.times[i] - self.times[i-1]

            # Calculate SIF
            dk = Y * ds * jnp.sqrt(jnp.pi * prev_crack)

            # Apply Paris law to get crack growth increment
            da_dn = jnp.exp(logc) * dk**m

            # Calculate new crack length
            new_crack = prev_crack + navg * dt * da_dn

            # Store the new crack length
            crack_lengths = crack_lengths.at[i].set(new_crack)

        # Apply the observation model (if any)
        observed_crack_lengths = self.observation_model.observe(crack_lengths)

        # Observe the crack lengths with normal noise
        numpyro.sample("obs",
                       dist.Normal(observed_crack_lengths, noise_std),
                       obs=observations)

        # Return deterministic quantities for later inspection
        numpyro.deterministic("predicted_crack_lengths", crack_lengths[1:])

    def run_inference(self, component_idx: int = 0, Y: float = 1.12,
                      navg: Optional[float] = None, num_warmup: int = 2000,
                      num_samples: int = 4000, num_chains: int = 4,
                      random_seed: int = 42, progress_bar: bool = True
                      ) -> Dict:
        """
        Run MCMC inference using the NUTS sampler.

        This method performs Bayesian inference for the crack growth model
        parameters using the No-U-Turn Sampler (NUTS).

        Parameters
        ----------
        component_idx : int, optional
            Index of the component to model in the crack_growth_data.
            Default is 0.
        Y : float, optional
            Geometry factor for stress intensity factor calculation.
            Default is 1.12.
        navg : float, optional
            Average number of cycles per time unit.
            If None, it will be sampled from the prior.
        num_warmup : int, optional
            Number of warmup steps for MCMC. Default is 1000.
        num_samples : int, optional
            Number of samples to draw after warmup. Default is 1000.
        num_chains : int, optional
            Number of MCMC chains to run. Default is 4.
        random_seed : int, optional
            Random seed for reproducibility. Default is 42.
        progress_bar : bool, optional
            Whether to show progress bar during sampling. Default is True.

        Returns
        -------
        dict
            Dictionary containing the inference results, including:
            - mcmc: The MCMC object
            - samples: Posterior samples
            - summary: Summary statistics of the posterior
        """
        # Create kernel for NUTS sampler with initialization strategy
        kernel = NUTS(self.model,
                      init_strategy=init_to_median,
                      target_accept_prob=0.9)

        # Setup MCMC with the kernel
        self.mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=progress_bar,
        )

        # Run MCMC inference
        rng_key = jr.PRNGKey(random_seed)
        rng_key_, rng_key = jr.split(rng_key)
        self.mcmc.run(
            rng_key_,
            component_idx=component_idx,
            Y=Y,
            navg=navg,
        )

        # Get samples from the posterior
        self.posterior_samples = self.mcmc.get_samples(group_by_chain=True)

        # Store results for later use by other methods
        self._results = {
            "mcmc": self.mcmc,
            "samples": self.posterior_samples
        }

        # Generate summary statistics
        summary = self.summarise_posterior(print_summary=False)
        self._results["summary"] = summary

        return {
            "mcmc": self.mcmc,
            "samples": self.posterior_samples,
            "summary": summary
        }

    def summarise_posterior(self, print_summary=True):
        """
        Summarize the posterior distribution from MCMC sampling.

        Parameters
        ----------
        print_summary : bool, optional
            Whether to print the summary statistics, by default True

        Returns
        -------
        dict
            Summary statistics for each parameter
        """
        if not hasattr(self, '_results'):
            raise ValueError("No inference results available. \
                              Run 'run_inference' first.")

        # Convert NumPyro samples to ArviZ InferenceData
        inference_data = az.from_numpyro(self.mcmc)

        # Generate summary using ArviZ
        summary = az.summary(inference_data, round_to=4)

        # Convert to dictionary for consistent return format
        summary_dict = summary.to_dict()

        # Print summary if requested
        if print_summary:
            print(summary)

        return summary_dict

    def generate_predictions(self, num_samples: int = 100,
                             component_idx: int = 0,
                             Y: float = 1.12,
                             navg: Optional[float] = None,
                             random_seed: int = None) -> Dict:
        """
        Generate posterior predictive samples.

        Parameters
        ----------
        num_samples : int, optional
            Number of posterior predictive samples to generate. Default is 100.
        component_idx : int, optional
            Index of the component to generate predictions for. Default is 0.
        Y : float, optional
            Geometry factor for stress intensity factor calculation.
            Default is 1.12.
        navg : float, optional
            Average number of cycles per time unit.
            If None, posterior samples of navg will be used if available.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        dict
            Dictionary with posterior predictive samples
        """
        if self.posterior_samples is None:
            raise ValueError("No posterior samples available.\
                             Run inference first.")

        # Set up random number generator
        rng_key = jr.PRNGKey(random_seed if random_seed is not None else 0)

        # Create predictive object with named arguments
        predictive = Predictive(self.model,
                                posterior_samples=self.posterior_samples,
                                num_samples=num_samples)

        # Generate predictions with specified parameters
        predictions = predictive(
            rng_key,
            component_idx=component_idx,  # Use specified component
            Y=Y,
            navg=navg  # Use specified navg or posterior samples if None
        )

        return predictions
