import numpy as np
import jax.numpy as jnp
from abc import ABC, abstractmethod
from src.crack_growth_models import ParisErdogan, BaseCrackGrowthModel


class ObservationModel(ABC):
    """
    Abstract base class for observation models.

    Observation models transform the true system state (crack length)
    into observed measurements. This could include noise, bias,
    or other measurement effects.
    """

    @abstractmethod
    def observe(self, x, t=None):
        """
        Transform true state to observed state.

        Parameters
        ----------
        x : float or array
            True state (e.g., crack length)
        t : float or array, optional
            Time of observation

        Returns
        -------
        float or array
            Observed state
        """
        pass


class IdentityObservation(ObservationModel):
    """
    Identity observation model that returns the state unchanged.
    """

    def observe(self, x, t=None):
        """
        Return the state unchanged.

        Parameters
        ----------
        x : float or array
            True state (e.g., crack length)
        t : float or array, optional
            Time of observation (not used)

        Returns
        -------
        float or array
            Same as input state
        """
        return x


class LinearObservation(ObservationModel):
    """
    Linear observation model with scaling and offset.

    Implements y = a*x + b transformation.
    """

    def __init__(self, scale=1.0, offset=0.0):
        """
        Initialize with scaling factor and offset.

        Parameters
        ----------
        scale : float, optional
            Scaling factor for the state
        offset : float, optional
            Offset to add after scaling
        """
        self.scale = scale
        self.offset = offset

    def observe(self, x, t=None):
        """
        Apply linear transformation to state.

        Parameters
        ----------
        x : float or array
            True state (e.g., crack length)
        t : float or array, optional
            Time of observation (not used)

        Returns
        -------
        float or array
            Transformed state: scale*x + offset
        """
        return self.scale * x + self.offset


class GaussianNoiseObservation(ObservationModel):
    """
    Observation model that adds Gaussian noise to the state.
    """

    def __init__(self, std_dev, random_seed=None):
        """
        Initialize with noise standard deviation.

        Parameters
        ----------
        std_dev : float
            Standard deviation of the Gaussian noise
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.std_dev = std_dev
        self.rng = np.random.RandomState(random_seed)

    def observe(self, x, t=None):
        """
        Add Gaussian noise to the state.

        Parameters
        ----------
        x : float or array
            True state (e.g., crack length)
        t : float or array, optional
            Time of observation (not used)

        Returns
        -------
        float or array
            State with added noise
        """
        # Generate noise with appropriate shape
        if isinstance(x, (np.ndarray, jnp.ndarray)):
            noise = self.rng.normal(0, self.std_dev, size=x.shape)
        else:
            noise = self.rng.normal(0, self.std_dev)

        return x + noise


class CompositeObservation(ObservationModel):
    """
    Combines multiple observation models in sequence.
    """

    def __init__(self, observation_models):
        """
        Initialize with a list of observation models.

        Parameters
        ----------
        observation_models : list
            List of observation models to apply in sequence
        """
        self.models = observation_models

    def observe(self, x, t=None):
        """
        Apply all observation models in sequence.

        Parameters
        ----------
        x : float or array
            True state (e.g., crack length)
        t : float or array, optional
            Time of observation

        Returns
        -------
        float or array
            State after all transformations
        """
        result = x
        for model in self.models:
            result = model.observe(result, t)
        return result


class CrackGrowthPredictor:
    """
    A utility class for predicting crack growth using
    different crack growth models.
    Can be used for visualization purposes
    and as a component in Bayesian models.
    """

    def __init__(self, model_class=ParisErdogan,
                 model_params=None, observation_model=None):
        """
        Initialize the predictor with a specified crack growth model.

        Parameters
        ----------
        model_class : class, optional
            Class to use for crack growth modeling, defaults to ParisErdogan.
            Must be a subclass of BaseCrackGrowthModel.
        model_params : dict, optional
            Additional parameters to pass to the model constructor
            (e.g., Y factor for SIF calculation)
        observation_model : ObservationModel, optional
            Model for transforming true crack lengths to observations
        """
        # Validate that model_class is a subclass of BaseCrackGrowthModel
        if not issubclass(model_class, BaseCrackGrowthModel):
            raise TypeError("model_class must be a subclass \
                            of BaseCrackGrowthModel")

        self.model_class = model_class
        self.model_params = model_params or {}
        self.observation_model = observation_model or IdentityObservation()

    def predict_crack_growth(self, logc, m, ds,
                             navg, a0, times, include_observations=True):
        """
        Predict crack growth using the specified crack growth model.

        Parameters
        ----------
        logc : float or array
            Natural logarithm of Paris law parameter C
        m : float or array
            Paris law exponent
        ds : float or array
            Stress range or stress range array (for variable stress models)
            - For constant stress: single value or array of values
            (one per sample)
            - For variable stress: 1D array of stress values
            (for single sample) or 2D array where each row
            corresponds to a sample's stress history
        navg : float or array
            Average number of cycles per time unit
        a0 : float or array
            Initial crack length
        times : array
            Time points for prediction.
            Can be 1D array (same times for all samples) or
            2D array of shape (n_samples, n_times)
        include_observations : bool, optional
            Whether to apply the observation model to the predictions

        Returns
        -------
        crack_lengths : array
            Predicted crack lengths at the specified time points
            Shape (n_samples, n_times) or (n_times,) for single sample
        """
        # Convert inputs to arrays if they are scalars
        if not isinstance(logc, (np.ndarray, jnp.ndarray)):
            logc = np.array([logc])
        if not isinstance(m, (np.ndarray, jnp.ndarray)):
            m = np.array([m])
        if not isinstance(navg, (np.ndarray, jnp.ndarray)):
            navg = np.array([navg])
        if not isinstance(a0, (np.ndarray, jnp.ndarray)):
            a0 = np.array([a0])

        # Handle ds differently depending on the model type
        is_variable_stress = 'Variable' in self.model_class.__name__

        # For variable stress models, ds could be a 1D array
        # for single instance
        # or a 2D array where each row is a sample's stress history
        if is_variable_stress:
            if not isinstance(ds, (np.ndarray, jnp.ndarray)):
                # Convert scalar to 1D array
                ds = np.array([ds])

            # Determine if ds represents multiple instances
            # or variable stress for a single instance
            if ds.ndim == 1:
                # Single instance with variable stress
                # Check if length matches expected (times.shape[0]-1)
                expected_length = len(times)-1 if times.ndim == 1 \
                      else times.shape[1]-1
                if len(ds) != expected_length:
                    raise ValueError(f"For variable stress with single \
                                     instance, ds length must be \
                                      {expected_length} (times - 1)")

                # Reshape to represent a single sample
                batch_size = 1
                ds = ds.reshape(1, -1)
            else:
                # Multiple instances, each with its own stress history
                batch_size = ds.shape[0]
        else:
            # Standard constant stress model
            if not isinstance(ds, (np.ndarray, jnp.ndarray)):
                ds = np.array([ds])
            batch_size = len(ds)

        # Handle time arrays
        if len(times.shape) > 1:
            # Multiple time series for multiple samples (n_samples, n_times)
            n_samples = times.shape[0]
            n_times = times.shape[1]

            # Ensure all parameter arrays have the same batch dimension
            if len(logc) != n_samples:
                if len(logc) == 1:
                    logc = np.repeat(logc, n_samples)
                else:
                    raise ValueError("logc must have same first dimension as \
                                     times or be a scalar")
            if len(m) != n_samples:
                if len(m) == 1:
                    m = np.repeat(m, n_samples)
                else:
                    raise ValueError("m must have same first dimension as \
                                     times or be a scalar")
            if len(navg) != n_samples:
                if len(navg) == 1:
                    navg = np.repeat(navg, n_samples)
                else:
                    raise ValueError("navg must have same first dimension as \
                                     times or be a scalar")
            if len(a0) != n_samples:
                if len(a0) == 1:
                    a0 = np.repeat(a0, n_samples)
                else:
                    raise ValueError("a0 must have same first dimension as \
                                     times or be a scalar")

            # For constant stress, ensure ds matches n_samples
            if not is_variable_stress and len(ds) != n_samples:
                if len(ds) == 1:
                    ds = np.repeat(ds, n_samples)
                else:
                    raise ValueError("ds must have same first dimension as \
                                     times or be a scalar")

            # For variable stress with multiple samples, ensure ds shape
            # matches [n_samples, n_times-1]
            if is_variable_stress:
                if ds.shape[0] != n_samples:
                    if ds.shape[0] == 1:
                        # Repeat the single stress history for all samples
                        ds = np.repeat(ds, n_samples, axis=0)
                    else:
                        raise ValueError("For variable stress, ds must have \
                                         first dimension matching times or \
                                         be a single stress history")

            # Initialize output array with proper shape
            crack_lengths = np.zeros_like(times)

            # Compute crack growth for each set of parameters
            #  with its own time series
            for i in range(n_samples):
                # Create common parameter dictionary
                model_params = {
                    'logc': logc[i],
                    'm': m[i],
                    'navg': navg[i],
                    'a0': a0[i],
                    't': times[i],
                    **self.model_params
                }

                # Add model-specific parameters
                if is_variable_stress:
                    model_params['ds_array'] = ds[i]
                else:
                    model_params['ds'] = ds[i]

                # Create model with unified parameter interface
                model = self.model_class(**model_params)

                # Set initial crack length
                crack_lengths[i, 0] = a0[i]

                # Use discrete time formulation for crack growth prediction
                for j in range(1, n_times):
                    # Always pass time information -
                    # models that don't need it will ignore
                    crack_lengths[i, j] = model.state_eq(
                        crack_lengths[i, j-1], times[i, j-1]
                        )
        else:
            # Single time series for all samples
            # Determine batch size based on parameter arrays
            if is_variable_stress and ds.ndim > 1:
                # Multiple samples, each with its own stress array
                batch_size = ds.shape[0]
            else:
                # Either constant stress or a single variable stress array
                batch_size = max(len(logc), len(m), len(ds)
                                 if not is_variable_stress else 1,
                                 len(navg), len(a0))

            # Broadcast all parameter arrays to the same length
            if len(logc) < batch_size:
                logc = np.ones(batch_size) * logc[0] if len(logc) == 1 \
                    else np.array(logc)
            if len(m) < batch_size:
                m = np.ones(batch_size) * m[0] if len(m) == 1 else np.array(m)
            if len(navg) < batch_size:
                navg = np.ones(batch_size) * navg[0] if len(navg) == 1 \
                    else np.array(navg)
            if len(a0) < batch_size:
                a0 = np.ones(batch_size) * a0[0] if len(a0) == 1 \
                    else np.array(a0)

            # For constant stress, broadcast ds if needed
            if not is_variable_stress and len(ds) < batch_size:
                ds = np.ones(batch_size) * ds[0] if len(ds) == 1 \
                    else np.array(ds)

            # Initialize output array
            crack_lengths = np.zeros((batch_size, len(times)))

            # Compute crack growth for each set of parameters
            for i in range(batch_size):
                # Create common parameter dictionary
                model_params = {
                    'logc': logc[i],
                    'm': m[i],
                    'navg': navg[i],
                    'a0': a0[i],
                    't': times,
                    **self.model_params
                }

                # Add model-specific parameters
                if is_variable_stress:
                    model_params['ds_array'] = ds[i] if ds.ndim > 1 \
                        else ds.flatten()
                else:
                    model_params['ds'] = ds[i]

                # Create model with unified parameter interface
                model = self.model_class(**model_params)

                # Set initial crack length
                crack_lengths[i, 0] = a0[i]

                # Use discrete time formulation for crack growth prediction
                for j in range(1, len(times)):
                    # Always pass time information
                    crack_lengths[i, j] = model.state_eq(
                        crack_lengths[i, j-1], times[j-1]
                        )

        # Apply observation model if requested
        if include_observations and self.observation_model is not None:
            # Make a copy to avoid modifying the original true crack lengths
            observed_lengths = np.copy(crack_lengths)

            # Apply observation model to each trajectory and time point
            if observed_lengths.ndim == 1:
                # Single trajectory
                for j in range(len(observed_lengths)):
                    observed_lengths[j] = self.observation_model.observe(
                        observed_lengths[j], times[j]
                        if len(times.shape) > 1 else times[j])
            else:
                # Multiple trajectories
                for i in range(observed_lengths.shape[0]):
                    for j in range(observed_lengths.shape[1]):
                        observed_lengths[i, j] = \
                            self.observation_model.observe(
                                observed_lengths[i, j], times[i, j]
                                if len(times.shape) > 1 else times[j]
                            )

            crack_lengths = observed_lengths

        # If input was a single set of parameters, return a single trajectory
        if (len(times.shape) == 1 and batch_size == 1) or \
                (len(times.shape) > 1 and n_samples == 1):
            return crack_lengths[0]
        else:
            return crack_lengths

    def predict_failure_time(self, logc, m, ds, navg, a0, a_cr,
                             max_time=3.0, time_step=0.01):
        """
        Predict the time to failure (crack reaching critical length).

        Parameters
        ----------
        logc, m, ds, navg, a0 : float or array
            Paris law and initial condition parameters
        a_cr : float
            Critical crack length for failure
        max_time : float, optional
            Maximum time to simulate
        time_step : float, optional
            Time step for simulation

        Returns
        -------
        failure_times : float or array
            Time to failure for each parameter set
        """
        times = np.arange(0, max_time, time_step)
        crack_lengths = self.predict_crack_growth(logc, m, ds, navg, a0, times)

        # Find the first time when crack length exceeds critical length
        if crack_lengths.ndim == 1:
            # Single trajectory
            failure_index = np.argmax(crack_lengths >= a_cr)
            if failure_index == 0 and crack_lengths[0] < a_cr:
                # No failure within the time window
                return max_time
            return times[failure_index]
        else:
            # Multiple trajectories
            failure_times = np.zeros(len(crack_lengths))
            for i in range(len(crack_lengths)):
                failure_index = np.argmax(crack_lengths[i] >= a_cr)
                if failure_index == 0 and crack_lengths[i, 0] < a_cr:
                    # No failure within the time window
                    failure_times[i] = max_time
                else:
                    failure_times[i] = times[failure_index]
            return failure_times

    def predict_variable_stress_observations(self, stress_periods, logc, m,
                                             navg, a0,
                                             include_observations=True,
                                             time_discretization=50):
        """
        Generate crack growth predictions for variable stress with observations
        at stress period boundaries.

        This method is specifically designed for variable stress scenarios
        where observations are taken at the end of each stress period, which
        is a common experimental setup.

        Parameters
        ----------
        stress_periods : list of tuples
            List of (start_time, end_time, stress_level) tuples defining
            stress periods
        logc : float or array
            Natural logarithm of Paris law parameter C
        m : float or array
            Paris law exponent
        navg : float or array
            Average number of cycles per time unit
        a0 : float or array
            Initial crack length
        include_observations : bool, optional
            Whether to apply the observation model to the predictions
        time_discretization : int, optional
            Number of time points for internal integration (default: 50)
            Higher values give more accurate results

        Returns
        -------
        dict
            Dictionary containing:
            - 'observation_times': Times at stress period boundaries
            - 'crack_lengths': Crack lengths at observation times
            - 'full_times': Full time array used for integration
            - 'full_crack_lengths': Full crack length trajectory
            - 'stress_periods': Original stress periods for reference

        Examples
        --------
        >>> predictor = CrackGrowthPredictor(
        ...     model_class=VariableStressParisErdogan)
        >>> stress_periods = [(0.0, 0.5, 30.0), (0.5, 1.0, 25.0)]
        >>> result = predictor.predict_variable_stress_observations(
        ...     stress_periods=stress_periods,
        ...     logc=np.log(5e-14), m=3.12, navg=2.8e6, a0=39.0
        ... )
        >>> print(result['observation_times'])  # [0.0, 0.5, 1.0]
        >>> print(result['crack_lengths'])      # [39.0, 55.2, 72.1]
        """
        from src.crack_growth_models import VariableStressParisErdogan

        # Validate that we're using a variable stress model
        if not ('Variable' in self.model_class.__name__ or
                self.model_class == VariableStressParisErdogan):
            raise ValueError("This method requires a variable stress model. "
                             "Use VariableStressParisErdogan or similar.")

        # Extract observation times from stress periods
        observation_times = [stress_periods[0][0]]  # Start with first time
        for _, end_time, _ in stress_periods:
            if end_time not in observation_times:
                observation_times.append(end_time)
        observation_times = np.array(sorted(observation_times))

        # Create fine time discretization for accurate integration
        full_times = np.linspace(observation_times[0], observation_times[-1],
                                 time_discretization)

        # Convert stress periods to stress array for the fine discretization
        def convert_stress_periods_to_array(stress_periods, times):
            ds_array = []
            for i in range(len(times)-1):
                t_mid = (times[i] + times[i+1])/2  # midpoint of interval
                # Find which period contains this time
                for start, end, stress in stress_periods:
                    if start <= t_mid < end:
                        ds_array.append(stress)
                        break
                else:
                    raise ValueError(f"No stress period covers time {t_mid}")
            return np.array(ds_array)

        ds_array = convert_stress_periods_to_array(stress_periods, full_times)

        # Predict full trajectory
        full_crack_lengths = self.predict_crack_growth(
            logc=logc, m=m, ds=ds_array, navg=navg, a0=a0,
            times=full_times, include_observations=include_observations
        )

        # Extract observations at stress period boundaries
        observed_indices = [np.argmin(np.abs(full_times - t))
                            for t in observation_times]
        observed_crack_lengths = full_crack_lengths[observed_indices]

        return {
            'observation_times': observation_times,
            'crack_lengths': observed_crack_lengths,
            'full_times': full_times,
            'full_crack_lengths': full_crack_lengths,
            'stress_periods': stress_periods
        }
