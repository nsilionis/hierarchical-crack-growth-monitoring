import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import arviz as az
from typing import Dict, Optional, Any
import warnings
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

    def generate_predictions(self, num_samples: int = 1000,
                             component_idx: int = 0,
                             Y: float = 1.12, navg: Optional[float] = None,
                             random_seed: int = 42) -> Dict[str, jnp.ndarray]:
        """
        Generate posterior predictions for crack growth.

        .. deprecated::
            This method is deprecated and will be removed in a future version.
            Use `generate_posterior_predictive` instead for proper posterior
            predictive sampling with NumPyro's Predictive class.

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
        warnings.warn(
            "generate_predictions is deprecated and will be \
                removed in a future version. "
            "Use generate_posterior_predictive instead for \
                proper posterior predictive sampling.",
            DeprecationWarning,
            stacklevel=2
        )

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

    def generate_posterior_predictive(self,
                                      num_samples: int = 1000,
                                      component_idx: int = 0,
                                      Y: float = 1.12,
                                      navg: Optional[float] = None,
                                      random_seed: int = 42
                                      ) -> Dict[str, jnp.ndarray]:
        """
        Generate proper posterior predictive samples
        using Numpyro's Predictive class.

        This samples from the full posterior predictive distribution,
        properly accounting for all sources of uncertainty
        in the generative process.

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
            Dictionary of predictions including samples
            from the posterior predictive
        """
        if self.posterior_samples is None:
            raise ValueError("No posterior samples available.\
                              Please run inference first.")

        # Set up the RNG key
        rng_key = jr.PRNGKey(random_seed)

        # Flatten chains for prediction
        flat_samples = {}
        for k, v in self.posterior_samples.items():
            # Exclude 'predicted_crack_lengths' since
            # it's a deterministic variable
            if k != 'predicted_crack_lengths':
                flat_samples[k] = v.reshape(-1)

        # Select random subset of samples if requested
        n_available = len(flat_samples["logc"])
        if num_samples < n_available:
            idx_key, pred_key = jr.split(rng_key)
            indices = jr.choice(idx_key, n_available,
                                (num_samples,), replace=False)
            for k in flat_samples.keys():
                flat_samples[k] = flat_samples[k][indices]
        else:
            pred_key = rng_key

        # Define a predictive model that returns what we want to observe
        def predictive_model(component_idx=0, Y=1.12, navg=None):
            # Prior distributions - will be overridden by the posterior samples
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

            # Store deterministic crack growth for plotting
            numpyro.deterministic("predicted_crack_lengths", crack_lengths)

            # Sample from the posterior predictive distribution
            # This properly incorporates observation noise
            numpyro.sample("obs", dist.Normal(crack_lengths, noise_std))

        # Use Numpyro's Predictive to generate posterior predictive samples
        predictive = Predictive(predictive_model,
                                posterior_samples=flat_samples,
                                num_samples=num_samples,
                                return_sites=["predicted_crack_lengths", "obs"]
                                )

        # Generate samples
        predictive_samples = predictive(
            pred_key,
            component_idx=component_idx,
            Y=Y,
            navg=navg
        )

        # Format the results
        predictions = {
            "predicted_crack_lengths":
            predictive_samples["predicted_crack_lengths"],
            "obs":
            predictive_samples["obs"]
        }

        return predictions


class MTLBayesianModel:
    """
    Multi-task learning Bayesian model for Paris law parameters.

    This class implements a hierarchical Bayesian model where:
    - Material parameters (C, m) are fixed effects shared across components
    - Stress ranges are random effects varying between components
    - Component data is pooled for joint inference
    """

    def __init__(self,
                 priors: Optional[Dict[str, dist.Distribution]] = None,
                 hyperpriors: Optional[Dict[str, dist.Distribution]] = None,
                 crack_growth_data: Optional[Dict[str, Any]] = None):
        """
        Initialize the MTL Bayesian model with priors and data.

        Parameters
        ----------
        priors : dict, optional
            Dictionary of prior distributions for fixed effect parameters
        hyperpriors : dict, optional
            Dictionary of hyperprior distributions for random effect parameters
        crack_growth_data : dict, optional
            Dictionary containing the crack growth data for multiple components
        """
        # Default priors for fixed effects if none provided
        if priors is None:
            self.priors = {
                "logc": dist.Normal(-30.0, 2.0),
                "m": dist.Normal(3.0, 0.5),
                "noise_std": dist.HalfNormal(2.0)
            }
        else:
            self.priors = priors

        if hyperpriors is None:
            self.hyperpriors = hyperpriors
            self.priors["ds"] = dist.Weibull(14.9, 1.6)
        else:
            self.hyperpriors = hyperpriors
            self._validate_hyperpriors()

        # Store data
        self.crack_growth_data = crack_growth_data

        # Get number of components from data
        if crack_growth_data is not None:
            self.n_components = len(crack_growth_data["times"])
        else:
            self.n_components = 0

        # Initialize storage for posterior samples
        self.mcmc = None
        self.posterior_samples = None

    def _validate_hyperpriors(self):
        """
        Validate that hyperpriors contain required keys
        for Weibull distribution.
        """
        required_keys = {'weibull_concentration', 'weibull_scale'}
        provided_keys = set(self.hyperpriors.keys())

        if not required_keys.issubset(provided_keys):
            missing = required_keys - provided_keys
            raise ValueError(f"Hyperpriors missing required keys: {missing}")

    def model(self, component_idx=0, Y=1.12, navg=None):
        """
        Hierarchical Bayesian model for Paris law
        parameters across multiple components.

        Parameters
        ----------
        component_idx : int, optional
            Index of the component to model (used for interface compatibility)
        Y : float, optional
            Geometry factor for SIF calculation
        navg : float, optional
            Average cycles per year. If None, uses a default value

        Returns
        -------
        None
        """
        # Fixed effects - shared across all components
        logc = numpyro.sample("logc", self.priors["logc"])
        m = numpyro.sample("m", self.priors["m"])
        noise_std = numpyro.sample("noise_std", self.priors["noise_std"])

        # Hyperpriors for stress range distribution
        if self.hyperpriors is None:
            # Simple i.i.d. random effects with fixed prior
            with numpyro.plate("components", self.n_components):
                ds = numpyro.sample("ds", self.priors["ds"])
        else:
            weibull_concentration = numpyro.sample("weibull_concentration",
                                                   self.hyperpriors[
                                                       "weibull_concentration"]
                                                   )
            weibull_scale = numpyro.sample("weibull_scale",
                                           self.hyperpriors["weibull_scale"])

            # Random effects - component-specific stress ranges
            with numpyro.plate("components", self.n_components):
                ds = numpyro.sample("ds",
                                    dist.Weibull(weibull_concentration,
                                                 weibull_scale))

        # Default navg if not provided
        if navg is None:
            navg = 2.8e6

        # Initialize list to store predictions for all components
        predicted_crack_lengths_all = []

        # Process each component
        for comp_idx in range(self.n_components):
            # Extract data for this component
            times = self.crack_growth_data["times"][comp_idx]
            data = self.crack_growth_data["noisy_crack_lengths"][comp_idx]

            # Initial crack length (first observation)
            init_crack = data[0]

            # Create Paris-Erdogan model instance with
            # component-specific stress
            paris = ParisErdogan(
                logc=logc,
                m=m,
                ds=ds[comp_idx],  # Component-specific stress range
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

            # Likelihood for observations of this component
            with numpyro.plate(f"obs_component_{comp_idx}", len(data)):
                numpyro.sample(
                    f"obs_{comp_idx}",
                    dist.Normal(crack_lengths, noise_std),
                    obs=data
                )

            # Store predicted crack lengths
            predicted_crack_lengths_all.append(crack_lengths[1:])

        # Store ALL predicted crack lengths for all components
        # Handle variable-length time series by padding if necessary
        if len(predicted_crack_lengths_all) > 0:
            # Check if all components have the same number of time steps
            lengths = [len(pred) for pred in predicted_crack_lengths_all]
            max_length = max(lengths)

            if min(lengths) == max_length:
                # All components have same length - use direct stacking
                numpyro.deterministic("predicted_crack_lengths",
                                      jnp.stack(predicted_crack_lengths_all))
            else:
                # Components have different lengths - pad with NaN
                padded_predictions = []
                for pred in predicted_crack_lengths_all:
                    if len(pred) < max_length:
                        # Pad with NaN values for missing time steps
                        padding = jnp.full(max_length - len(pred), jnp.nan)
                        padded_pred = jnp.concatenate([pred, padding])
                    else:
                        padded_pred = pred
                    padded_predictions.append(padded_pred)

                numpyro.deterministic("predicted_crack_lengths",
                                      jnp.stack(padded_predictions))

    def run_inference(self,
                      component_idx: int = 0,
                      Y: float = 1.12,
                      navg: Optional[float] = None,
                      num_warmup: int = 1000,
                      num_samples: int = 1000,
                      num_chains: int = 4,
                      progress_bar: bool = True) -> Dict:
        """
        Run MCMC inference for the hierarchical model.

        Parameters
        ----------
        component_idx : int, optional
            Index of the component (kept for interface compatibility,
            but MTL processes all components)
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

        return az.summary(inference_data, round_to=3)

    def check_rhat(self,
                   threshold: float = 1.01,
                   print_results: bool = True,
                   return_dict: bool = False) -> Optional[Dict[str, Any]]:
        """
        Check R-hat convergence diagnostics for MCMC chains.

        The R-hat statistic (also known as the potential scale reduction
        factor) measures the between- and within-chain variances for each
        model parameter. Values close to 1.0 indicate good convergence,
        while values substantially greater than 1.0 suggest that the chains
        have not yet converged.

        Parameters
        ----------
        threshold : float, optional
            R-hat threshold for convergence assessment (default: 1.01).
            Common thresholds are:
            - 1.01 (strict): Recommended for final results
            - 1.05 (moderate): Acceptable for exploratory analysis
            - 1.10 (lenient): May indicate convergence issues
        print_results : bool, optional
            Whether to print convergence diagnostics to stdout (default: True)
        return_dict : bool, optional
            Whether to return a dictionary with convergence statistics
            (default: False)

        Returns
        -------
        dict or None
            If return_dict=True, returns a dictionary containing:
            - 'rhat_values': pandas Series of R-hat values for each parameter
            - 'converged_params': number of parameters below threshold
            - 'valid_params': number of parameters with valid R-hat values
            - 'nan_params': number of parameters with NaN R-hat values
            - 'total_params': total number of parameters
            - 'max_rhat': maximum R-hat value (excluding NaN)
            - 'all_converged': boolean indicating if all valid parameters
              converged
            - 'threshold': the threshold used for assessment
            - 'convergence_rate': convergence rate for valid parameters only

        Raises
        ------
        ValueError
            If no MCMC results are available (inference not run yet)
            If threshold is not a positive number

        Notes
        -----
        R-hat values are computed using the rank-normalized split-R-hat as
        implemented in ArviZ, which is more robust than the traditional R-hat
        statistic for heavy-tailed distributions.

        References
        ----------
        Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P. C.
        (2021). Rank-normalization, folding, and localization: An improved
        R-hat for assessing convergence of MCMC. Bayesian analysis, 16(2),
        667-718.
        """
        # Input validation
        if not isinstance(threshold, (int, float)) or threshold <= 0:
            raise ValueError(
                f"Threshold must be a positive number, got {threshold}")

        if self.mcmc is None:
            raise ValueError(
                "No MCMC results available. Please run inference first.")

        # Get posterior summary with R-hat values
        try:
            az_post = self.summarise_posterior(print_summary=False)
            rhat_values = az_post['r_hat']
        except KeyError as e:
            raise ValueError(
                f"R-hat values not found in posterior summary: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to compute posterior summary: {e}")

        # Compute convergence statistics (excluding NaN values)
        valid_rhat = rhat_values.dropna()
        nan_mask = rhat_values.isna()
        converged_mask = valid_rhat <= threshold
        n_converged = converged_mask.sum()
        n_valid = len(valid_rhat)
        n_nan = nan_mask.sum()
        n_total = len(rhat_values)
        max_rhat = valid_rhat.max() if len(valid_rhat) > 0 else float('nan')
        all_converged = n_converged == n_valid

        if print_results:
            print("R-hat Convergence Diagnostics:")
            print("=" * 50)
            print(f"Threshold: {threshold}")
            print("-" * 50)

            for param, rhat in rhat_values.items():
                if rhat != rhat:  # Check for NaN
                    status = "N/A"
                    color_code = "(constant/deterministic)"
                    print(f"{param:<25}: {'NaN':<6} {status} {color_code}")
                else:
                    status = "✓" if rhat <= threshold else "⚠"
                    color_code = "" if rhat <= threshold else "(!)"
                    print(f"{param:<25}: {rhat:.4f} {status} {color_code}")

            print("\nConvergence Summary:")
            if n_nan > 0:
                print(f"Parameters with NaN R-hat: {n_nan} "
                      "(constant/deterministic)")
            if n_valid > 0:
                print(f"Parameters with R-hat ≤ {threshold}: "
                      f"{n_converged}/{n_valid}")
                print(f"Convergence rate: {n_converged/n_valid*100:.1f}%")
                print(f"Max R-hat: {max_rhat:.4f}")

                if all_converged:
                    print("✅ All variable parameters have converged!")
                else:
                    n_not_converged = n_valid - n_converged
                    print(f"⚠️  {n_not_converged} parameter(s) may not have "
                          "converged.")
                    print("Consider running more MCMC samples or checking "
                          "model specification.")
            else:
                print("All parameters have NaN R-hat values "
                      "(constant/deterministic)")

        # Return dictionary if requested
        if return_dict:
            return {
                'rhat_values': rhat_values,
                'converged_params': int(n_converged),
                'valid_params': int(n_valid),
                'nan_params': int(n_nan),
                'total_params': int(n_total),
                'max_rhat': float(max_rhat),
                'all_converged': bool(all_converged),
                'threshold': threshold,
                'convergence_rate': (float(n_converged/n_valid) if n_valid > 0
                                     else float('nan'))
            }

        return None

    def generate_posterior_predictive(self,
                                      num_samples: int = 1000,
                                      component_idx: int = 0,
                                      Y: float = 1.12,
                                      navg: Optional[float] = None,
                                      random_seed: int = 42
                                      ) -> Dict[str, jnp.ndarray]:
        """
        Generate posterior predictive samples for the specified component.

        For interface compatibility with STLBayesianModel, this method
        returns predictions for a single component specified by component_idx.

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
            Dictionary of predictions for the specified component
        """
        if self.posterior_samples is None:
            raise ValueError("No posterior samples available. \
                             Please run inference first.")

        # Set up the RNG key
        rng_key = jr.PRNGKey(random_seed)

        # Flatten chains for prediction, excluding deterministic variables
        flat_samples = {}
        for k, v in self.posterior_samples.items():
            if k != 'predicted_crack_lengths':
                # For ds parameter, extract only the component of interest
                if k == 'ds':
                    flat_samples[k] = v.reshape(-1, self.n_components)[
                        :, component_idx]
                else:
                    flat_samples[k] = v.reshape(-1)

        # Select random subset of samples if requested
        n_available = len(flat_samples["logc"])
        if num_samples < n_available:
            idx_key, pred_key = jr.split(rng_key)
            indices = jr.choice(
                idx_key, n_available, (num_samples,), replace=False)
            for k in flat_samples.keys():
                flat_samples[k] = flat_samples[k][indices]
        else:
            pred_key = rng_key

        # Define a predictive model for the specified component
        def predictive_model(component_idx=0, Y=1.12, navg=None):
            # Prior distributions - will be overridden by the posterior samples
            logc = numpyro.sample("logc", self.priors["logc"])
            m = numpyro.sample("m", self.priors["m"])
            # Use a dummy prior for ds - will be
            # overridden by posterior samples
            ds = numpyro.sample("ds", dist.Gamma(5.0, 0.3))
            noise_std = numpyro.sample("noise_std", self.priors["noise_std"])

            # Default navg if not provided
            if navg is None:
                navg = 2.8e6

            # Extract data for the specified component
            times = self.crack_growth_data["times"][component_idx]
            data = self.crack_growth_data["noisy_crack_lengths"][component_idx]

            # Initial crack length (first observation)
            init_crack = data[0]

            # Create Paris-Erdogan model instance
            paris = ParisErdogan(
                logc=logc,
                m=m,
                ds=ds,  # Use component-specific stress from posterior
                navg=navg,
                a0=init_crack,
                Y=Y,
                t=times
            )

            # Initialize array for predicted crack lengths
            crack_lengths = jnp.zeros(len(times))
            crack_lengths = crack_lengths.at[0].set(init_crack)

            # Generate crack growth trajectory
            for i in range(1, len(times)):
                crack_lengths = crack_lengths.at[i].set(
                    paris.state_eq(crack_lengths[i-1], times[i-1])
                )

            # Store deterministic crack growth for plotting
            numpyro.deterministic("predicted_crack_lengths", crack_lengths)

            # Sample from the posterior predictive distribution
            numpyro.sample("obs", dist.Normal(crack_lengths, noise_std))

        # Use Numpyro's Predictive to generate posterior predictive samples
        predictive = Predictive(predictive_model,
                                posterior_samples=flat_samples,
                                num_samples=num_samples,
                                return_sites=["predicted_crack_lengths", "obs"]
                                )

        # Generate samples
        predictive_samples = predictive(
            pred_key,
            component_idx=component_idx,
            Y=Y,
            navg=navg
        )

        # Format the results
        predictions = {
            "predicted_crack_lengths":
            predictive_samples["predicted_crack_lengths"],
            "obs":
            predictive_samples["obs"]
        }

        return predictions
