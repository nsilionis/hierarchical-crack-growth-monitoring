import jax.numpy as jnp
from scipy.integrate import solve_ivp


class ParisErdogan:
    """
    Implementation of the Paris-Erdogan law for
    fatigue crack growth prediction.
    This class implements the Paris-Erdogan law:
    da/dN = C * (ΔK)^m
    where da/dN is the crack growth rate, C and m are material parameters,
    and ΔK is the stress intensity factor range.

    Attributes
    ----------
    logc : float
        Natural logarithm of the Paris law coefficient C
    m : float
        Paris law exponent
    ds : float or array
        Stress range
    navg : float
        Average number of cycles per time unit
    a0 : float
        Initial crack length
    t : array
        Time points for evaluation
    Y : float
        Geometry correction factor
    dt : float
        Time step size
    """

    def __init__(self, logc, m, ds, navg, a0, Y, t):
        """
        Initialize the Paris-Erdogan model.

        Parameters
        ----------
        logc : float
            Natural logarithm of the Paris law coefficient C
        m : float
            Paris law exponent
        ds : float or array
            Stress range
        navg : float
            Average number of cycles per time unit
        a0 : float
            Initial crack length
        Y : float
            Geometry correction factor
        t : array
            Time points for evaluation
        """
        self.logc = logc
        self.m = m
        self.ds = ds
        self.navg = navg
        self.a0 = a0
        self.t = t
        self.Y = Y

        # Check that t has at least 2 elements before calculating dt
        if len(t) >= 2:
            self.dt = t[1] - t[0]
        else:
            # Handle the case when t has fewer than 2 elements
            self.dt = 0.01  # Default time step value
            print(f"""Warning: Time array has {len(t)} elements.
                   Using default dt={self.dt}""")

    def SIF(self, a):
        """
        Calculate the Stress Intensity Factor (SIF) range.

        Parameters
        ----------
        a : float or array
            Crack length

        Returns
        -------
        dk : float or array
            Stress intensity factor range
        """
        dk = self.Y * self.ds * jnp.sqrt(jnp.pi * a)
        return dk

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
        dadn : float or array
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
        nsol : array
            Number of cycles at each time point
        a : array
            Crack length at each time point
        """
        # ODE solution
        n_points = len(self.t)

        # Pre-allocate arrays with correct shape and type
        # Consider using regular numpy arrays
        #  instead of jax arrays with 'object' dtype
        nsol = jnp.zeros((1, n_points))
        a = jnp.zeros((1, n_points))

        # Solve the ODE once - no need for loop when only handling one case
        t_eval = self.t * self.navg
        sol = solve_ivp(self.ParisCont, [0, t_eval.max()],
                        [self.a0], t_eval=t_eval, method='RK45', rtol=1e-6)

        # Handle the solution directly instead of using .at indexing
        len_n = sol.t.shape[0]
        if len_n != n_points:
            # This code block might not be needed if t_eval is properly set
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
            Current time (not used)

        Returns
        -------
        x : float or array
            Updated crack length after one time step
        """
        # Vectorize this operation for efficiency when x is an array
        x = x + self.navg * self.dt * \
            (jnp.exp(self.logc) * (self.SIF(x))**self.m)
        return x

    def out_eq(self, x, t=None):
        """
        Output equation for observation model.

        Parameters
        ----------
        x : float or array
            Current crack length
        t : float, optional
            Current time (not used)

        Returns
        -------
        x : float or array
            Observed crack length
        """
        h = 1
        x = h * x
        return x
