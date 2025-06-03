import numpy as np
import jax.numpy as jnp
from abc import ABC, abstractmethod
from src.crack_growth_models import ParisErdogan


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
    A utility class for predicting crack growth using the Paris-Erdogan law.
    Can be used for visualization purposes and
    as a component in Bayesian models.
    """

    def __init__(self, Y=1.12, observation_model=None):
        """
        Initialize the predictor.

        Parameters
        ----------
        Y : float, optional
            The geometry factor Y for the stress intensity factor calculation.
            Default is 1.12 for a standard geometry.
        observation_model : ObservationModel, optional
            Model for transforming true crack lengths to observations
        """
        self.Y = Y
        self.observation_model = observation_model or IdentityObservation()

    def predict_crack_growth(self, logc, m, ds, navg,
                             a0, times, include_observations=True):
        """
        Predict crack growth using the Paris-Erdogan model.

        Parameters
        ----------
        logc : float or array
            Natural logarithm of Paris law parameter C
        m : float or array
            Paris law exponent
        ds : float or array
            Stress range
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
        if not isinstance(ds, (np.ndarray, jnp.ndarray)):
            ds = np.array([ds])
        if not isinstance(navg, (np.ndarray, jnp.ndarray)):
            navg = np.array([navg])
        if not isinstance(a0, (np.ndarray, jnp.ndarray)):
            a0 = np.array([a0])

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
                    raise ValueError("logc must have same first"
                                     "dimension as times or be a scalar")
            if len(m) != n_samples:
                if len(m) == 1:
                    m = np.repeat(m, n_samples)
                else:
                    raise ValueError("m must have same first dimension as"
                                     "times or be a scalar")
            if len(ds) != n_samples:
                if len(ds) == 1:
                    ds = np.repeat(ds, n_samples)
                else:
                    raise ValueError("ds must have same first dimension "
                                     "as times or be a scalar")
            if len(navg) != n_samples:
                if len(navg) == 1:
                    navg = np.repeat(navg, n_samples)
                else:
                    raise ValueError("navg must have same first dimension"
                                     " as times or be a scalar")
            if len(a0) != n_samples:
                if len(a0) == 1:
                    a0 = np.repeat(a0, n_samples)
                else:
                    raise ValueError("a0 must have same first dimension"
                                     " as times or be a scalar")

            # Initialize output array with proper shape
            crack_lengths = np.zeros_like(times)

            # Compute crack growth for each set of parameters
            # with its own time series
            for i in range(n_samples):
                paris = ParisErdogan(
                    logc=logc[i],
                    m=m[i],
                    ds=ds[i],
                    navg=navg[i],
                    a0=a0[i],
                    Y=self.Y,
                    t=times[i]
                )

                # Set initial crack length
                crack_lengths[i, 0] = a0[i]

                # Use discrete time formulation for crack growth prediction
                for j in range(1, n_times):
                    crack_lengths[i, j] = paris.state_eq(crack_lengths[i, j-1])

        else:
            # Single time series for all samples
            # Determine batch size from parameter arrays
            batch_size = max(len(logc), len(m), len(ds), len(navg), len(a0))

            # Broadcast all parameter arrays to the same length
            if len(logc) < batch_size:
                logc = np.ones(batch_size) * logc[0] \
                    if len(logc) == 1 else np.array(logc)
            if len(m) < batch_size:
                m = np.ones(batch_size) * m[0] if len(m) == 1 else np.array(m)
            if len(ds) < batch_size:
                ds = np.ones(batch_size) * ds[0] \
                    if len(ds) == 1 else np.array(ds)
            if len(navg) < batch_size:
                navg = np.ones(batch_size) * navg[0] \
                    if len(navg) == 1 else np.array(navg)
            if len(a0) < batch_size:
                a0 = np.ones(batch_size) * a0[0] \
                    if len(a0) == 1 else np.array(a0)

            # Initialize output array
            crack_lengths = np.zeros((batch_size, len(times)))

            # Compute crack growth for each set of parameters
            for i in range(batch_size):
                paris = ParisErdogan(
                    logc=logc[i],
                    m=m[i],
                    ds=ds[i],
                    navg=navg[i],
                    a0=a0[i],
                    Y=self.Y,
                    t=times
                )

                # Set initial crack length
                crack_lengths[i, 0] = a0[i]

                # Use discrete time formulation for crack growth prediction
                for j in range(1, len(times)):
                    crack_lengths[i, j] = paris.state_eq(crack_lengths[i, j-1])

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
