import jax.numpy as jnp
import jax.random as jr
import jax.lax as lax
import numpy as np
import tqdm
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median
from numpyro.diagnostics import hpdi
import arviz as az
from typing import Dict, List, Optional, Union, Any, Tuple
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

    def _crack_growth_step(self, crack_state, t_idx):
        """
        Single step of crack growth for use with jax.lax.scan

        Parameters
        ----------
        crack_state : float
            Current crack length
        t_idx : int
            Index in the time array

        Returns
        -------
        float
            Updated crack length
        """
        t = self.times[t_idx]
        return self.growth_model.state_eq(crack_state, t)

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
                    raise ValueError("navg parameter not provided, \
                                     no prior specified, \
                                     and no avg_cycles in data")
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

        # Simulate crack growth using JAX's functional approach
        # First time point is the initial crack length
        crack_lengths = jnp.zeros_like(self.times)
        crack_lengths = crack_lengths.at[0].set(a0)

        # Use jax.lax.scan for the time evolution
        # (more JAX-friendly than Python loop)
        # We start from index 1 since we already know index 0
        time_indices = jnp.arange(0, len(self.times)-1)
        _, crack_lengths_rest = lax.scan(
            lambda prev_crack, t_idx:
            (self._crack_growth_step(prev_crack, t_idx),
             self._crack_growth_step(prev_crack, t_idx)),
            a0,  # Initial state is a0
            time_indices  # Indices to iterate over
        )

        # Combine initial crack length with computed values
        crack_lengths = jnp.concatenate([jnp.array([a0]), crack_lengths_rest])

        # Apply the observation model to the simulated crack lengths
        # For an identity observation model, this just returns the input
        observed_crack_lengths = self.observation_model.observe(crack_lengths)

        # Observe the crack lengths with normal noise
        numpyro.sample("obs",
                       dist.Normal(observed_crack_lengths, noise_std),
                       obs=observations)

        # Return deterministic quantities for later inspection
        numpyro.deterministic("predicted_crack_lengths", crack_lengths)
