import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
from abc import ABC, abstractmethod


class SIFCalculator(ABC):
    """
    Abstract base class for Stress Intensity Factor (SIF) calculators.

    This defines the interface for all SIF calculation methods, allowing
    different implementations based on geometry and loading conditions.
    """

    @abstractmethod
    def calculate(self, a, ds):
        """
        Calculate the Stress Intensity Factor.

        Parameters
        ----------
        a : float or array
            Crack length
        ds : float or array
            Stress range

        Returns
        -------
        float or array
            Stress Intensity Factor
        """
        pass


class SimpleGeometrySIF(SIFCalculator):
    """
    SIF calculator for simple geometry with constant correction factor.
    """

    def __init__(self, Y=1.12):
        """
        Initialize with geometry correction factor.

        Parameters
        ----------
        Y : float
            Geometry correction factor
        """
        self.Y = Y

    def calculate(self, a, ds):
        """
        Calculate SIF using Y factor.

        Parameters
        ----------
        a : float or array
            Crack length
        ds : float or array
            Stress range

        Returns
        -------
        float or array
            Stress Intensity Factor
        """
        return self.Y * ds * jnp.sqrt(jnp.pi * a)


class BaseCrackGrowthModel(ABC):
    """
    Abstract base class for fatigue crack growth models.

    This class defines the common interface and functionality for all
    crack growth models, allowing for different material laws and
    loading conditions.
    """

    def __init__(self, logc, m, navg, a0, t, sif_calculator=None):
        """
        Initialize the base crack growth model.

        Parameters
        ----------
        logc : float
            Natural logarithm of the material parameter C
        m : float
            Material exponent
        navg : float
            Average number of cycles per time unit
        a0 : float
            Initial crack length
        t : array
            Time points for evaluation
        sif_calculator : SIFCalculator, optional
            Calculator for Stress Intensity Factor,
            defaults to SimpleGeometrySIF
        """
        self.logc = logc
        self.m = m
        self.navg = navg
        self.a0 = a0
        self.t = t

        # Set default SIF calculator if none provided
        self.sif_calculator = sif_calculator or SimpleGeometrySIF()

        # Calculate time step if possible
        if len(t) >= 2:
            self.dt = t[1] - t[0]
        else:
            self.dt = 0.01  # Default time step
            print(f"Warning: Time array has {len(t)} elements.\
                   Using default dt={self.dt}")

    def SIF(self, a, ds=None):
        """
        Calculate Stress Intensity Factor using the assigned calculator.

        Parameters
        ----------
        a : float or array
            Crack length
        ds : float or array, optional
            Stress range, uses the model's default if not provided

        Returns
        -------
        float or array
            Stress Intensity Factor
        """
        # This should be implemented by child classes which know
        # how to determine the stress range
        raise NotImplementedError(
            "Child classes must implement the SIF method"
        )

    @abstractmethod
    def state_eq(self, x, t=None):
        """
        State equation for crack growth.

        Parameters
        ----------
        x : float or array
            Current crack length
        t : float, optional
            Current time

        Returns
        -------
        float or array
            Updated crack length
        """
        pass


class ParisErdogan(BaseCrackGrowthModel):
    """
    Implementation of the Paris-Erdogan law for fatigue crack growth prediction
    with constant stress range.

    This class implements the Paris-Erdogan law:
    da/dN = C * (ΔK)^m
    where da/dN is the crack growth rate, C and m are material parameters,
    and ΔK is the stress intensity factor range.
    """

    def __init__(self, logc, m, ds, navg, a0, t, sif_calculator=None, Y=1.12):
        """
        Initialize the Paris-Erdogan model with constant stress range.

        Parameters
        ----------
        logc : float
            Natural logarithm of the Paris law coefficient C
        m : float
            Paris law exponent
        ds : float
            Stress range (constant for all time points)
        navg : float
            Average number of cycles per time unit
        a0 : float
            Initial crack length
        t : array
            Time points for evaluation
        sif_calculator : SIFCalculator, optional
            Calculator for SIF, defaults to SimpleGeometrySIF
        Y : float, optional
            Geometry correction factor, used only if sif_calculator is None
        """
        # If Y is provided but not sif_calculator, create one with the Y value
        if sif_calculator is None and Y != 1.12:
            sif_calculator = SimpleGeometrySIF(Y=Y)

        super().__init__(logc, m, navg, a0, t, sif_calculator)
        self.ds = ds

    def SIF(self, a, ds=None):
        """
        Calculate the Stress Intensity Factor (SIF) range.

        Parameters
        ----------
        a : float or array
            Crack length
        ds : float or array, optional
            Stress range, uses the model's default if not provided

        Returns
        -------
        float or array
            Stress intensity factor range
        """
        # Use provided stress range or default
        stress_range = self.ds if ds is None else ds
        return self.sif_calculator.calculate(a, stress_range)

    def ParisCont(self, dn, a):
        """
        Paris-Erdogan crack growth rate equation.

        Parameters
        ----------
        dn : float
            Current number of cycles (not used in calculation)
        a : float or array
            Current crack length

        Returns
        -------
        float or array
            Crack growth rate
        """
        dk = self.SIF(a)
        dadn = jnp.exp(self.logc) * dk**self.m
        return dadn

    def ContinuousTime(self):
        """
        Solve the Paris-Erdogan equation using a continuous time approach.

        Returns
        -------
        tuple of arrays
            (number of cycles, crack length) at each time point
        """
        # ODE solution
        n_points = len(self.t)

        # Pre-allocate arrays with correct shape and type
        nsol = jnp.zeros((1, n_points))
        a = jnp.zeros((1, n_points))

        # Solve the ODE
        t_eval = self.t * self.navg
        sol = solve_ivp(self.ParisCont, [0, t_eval.max()],
                        [self.a0], t_eval=t_eval, method='RK45', rtol=1e-6)

        # Handle the solution
        len_n = sol.t.shape[0]
        if len_n != n_points:
            nsol = jnp.concatenate((sol.t, jnp.ones(n_points - len_n)
                                    * sol.t[-1]))
            a = jnp.concatenate((sol.y.flatten(), jnp.ones(n_points - len_n)
                                 * sol.y[0, -1]))
        else:
            nsol = sol.t
            a = sol.y.flatten()

        return nsol.reshape(1, -1), a.reshape(1, -1)

    def state_eq(self, x, t=None):
        """
        State equation for discrete-time implementation.

        Parameters
        ----------
        x : float or array
            Current crack length
        t : float, optional
            Current time

        Returns
        -------
        float or array
            Updated crack length after one time step
        """
        # Vectorize this operation for efficiency when x is an array
        x = x + self.navg * self.dt * (jnp.exp(self.logc)
                                       * (self.SIF(x))**self.m)
        return x


class VariableStressParisErdogan(BaseCrackGrowthModel):
    """
    Implementation of the Paris-Erdogan law with time-varying stress ranges.

    This class extends the base Paris-Erdogan model to allow for different
    stress ranges at different time intervals.
    """

    def __init__(self, logc, m, ds_array, navg, a0, t, sif_calculator=None):
        """
        Initialize the Paris-Erdogan model with variable stress ranges.

        Parameters
        ----------
        logc : float
            Natural logarithm of the Paris law coefficient C
        m : float
            Paris law exponent
        ds_array : array
            Array of stress ranges per time interval.
            Should have length len(t)-1, where each value is the
            stress range for the interval [t[i], t[i+1]]
        navg : float
            Average number of cycles per time unit
        a0 : float
            Initial crack length
        t : array
            Time points for evaluation
        sif_calculator : SIFCalculator, optional
            Calculator for SIF, defaults to SimpleGeometrySIF
        """
        super().__init__(logc, m, navg, a0, t, sif_calculator)

        # Validate that ds_array has the correct length (one per time interval)
        expected_length = len(t) - 1
        if len(ds_array) != expected_length:
            raise ValueError(f"ds_array must have length {expected_length} "
                             f"(one per time interval), "
                             f"but got {len(ds_array)}")

        # Store the stress range array
        self.ds_array = np.array(ds_array)

    def get_stress_range(self, time):
        """
        Get the stress range for the time interval containing the given time.

        Parameters
        ----------
        time : float
            Time point

        Returns
        -------
        float
            Stress range for the interval containing the time
        """
        # Find the index of the time interval
        if time >= self.t[-1]:
            # If time is at or beyond the last time point,
            # use the last stress range
            return self.ds_array[-1]
        elif time < self.t[0]:
            # If time is before the first time point,
            # use the first stress range
            return self.ds_array[0]
        else:
            # Find the interval index (the largest i where t[i] <= time)
            idx = np.maximum(0,
                             np.searchsorted(self.t, time, side='right') - 1)

            # Return the corresponding stress range
            if idx >= len(self.ds_array):
                return self.ds_array[-1]
            return self.ds_array[idx]

    def SIF(self, a, ds=None, time=None):
        """
        Calculate the Stress Intensity Factor (SIF) range.

        Parameters
        ----------
        a : float or array
            Crack length
        ds : float or array, optional
            Stress range (overrides time-based lookup if provided)
        time : float, optional
            Current time for stress range lookup

        Returns
        -------
        float or array
            Stress intensity factor range
        """
        # Determine stress range based on provided parameters
        if ds is not None:
            # Use explicitly provided stress range
            stress_range = ds
        elif time is not None:
            # Look up stress range based on time
            stress_range = self.get_stress_range(time)
        else:
            # No way to determine stress range
            raise ValueError("Either ds or time must be \
                              provided to calculate SIF")

        return self.sif_calculator.calculate(a, stress_range)

    def state_eq(self, x, t):
        """
        State equation for discrete-time implementation
        with time-varying stress.

        Parameters
        ----------
        x : float or array
            Current crack length
        t : float
            Current time (used to determine stress range)

        Returns
        -------
        float or array
            Updated crack length after one time step
        """
        # Get stress range for the current time interval
        current_ds = self.get_stress_range(t)

        # Calculate SIF and crack growth increment
        x = x + self.navg * self.dt * (jnp.exp(self.logc) *
                                       (self.SIF(x, ds=current_ds))**self.m)
        return x
