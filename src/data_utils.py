import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from src.predictive_models import GaussianNoiseObservation


class SCGDataLoader:
    """
    This class is responsible for loading a dataset that contains stochastic
    crack growth (SCG) data. The data corresponds to crack growth realisations
    simulated using the the methodology described in:
    "Spectral fatigue analysis of ship structures based on a stochastic
    crack growth state model" by Makris et al. (2023).
    """

    def __init__(self):
        """
        Initializes the SCGDataLoader with the path to the dataset.

        Attributes
        root_dir : Path
            The root directory of the project.
        data_dir : Path
            The directory where the dataset is stored.
        -------
        Raises
        ------
        FileNotFoundError
            If the root or data directory does not exist.
        """
        # Get root directory of the project
        self.root_dir = Path(__file__).resolve().parents[1]
        # Check if the root directory exists
        if not self.root_dir.exists():
            raise FileNotFoundError(f"""Root directory {self.root_dir}
                                    does not exist.""")
        # Define the path to the data directory
        self.data_dir = self.root_dir / 'data'
        # Check if the data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"""Data directory {self.data_dir}
                                     does not exist.""")

    def load_data(self) -> dict:
        """
        Loads the data from separate .npy files and packs
        it into a dictionary. Each key corresponds to a
        different type of data. The keys are:
        - 'times': time instants of the crack growth (years)
        - 'avg_cycles': average number of cycles per load
        realisation
        - 'stress_ranges': stress range realisations (MPa)
        - 'crack_lengths': crack lengths at each time instant (mm)
        - 'paris_c': The C coefficient of the Paris law
        - 'paris_m': The m coefficient of the Paris law
        - 'initial_crack_length': The initial crack length (mm)
        The values are 2D numpy arrays with varying shapes/sizes.

        Returns
        -------
        data : dict
            A dictionary containing the loaded data.
        """
        # Data will be loaded from .npy files
        # Check if file with time instants
        # exists and load it
        if not (self.data_dir / 'gp_tr_times.npy').exists():
            raise FileNotFoundError(f"""File 'gp_tr_times.npy' does not exist
                                    in directory {self.data_dir}.""")
        else:
            times = np.load(self.data_dir / 'gp_tr_times.npy',
                            allow_pickle=False)
        # Check if file with crack lengths
        # exists and load it
        if not (self.data_dir / 'gp_tr_crack_lengths.npy').exists():
            raise FileNotFoundError(f"""File 'gp_tr_crack_lengths.npy' doesn't
                                     exist in directory {self.data_dir}.""")
        else:
            crack_lengths = np.load(self.data_dir / 'gp_tr_crack_lengths.npy',
                                    allow_pickle=False)
            # Reshape the crack lengths array to match the times array
            crack_lengths = crack_lengths[:, :times.shape[1]]
        # Check if file with Paris C coefficient
        # exists and load it
        if not (self.data_dir / 'c_gp_data.npy').exists():
            raise FileNotFoundError(f"""File 'c_gp_data.npy' does not exist
                                    in directory {self.data_dir}.""")
        else:
            paris_c = np.load(self.data_dir / 'c_gp_data.npy',
                              allow_pickle=False)
            # Reshape the Paris C coefficient array
            paris_c = paris_c[:, 0]
        # Check if file with Paris m coefficient
        # exists and load it
        if not (self.data_dir / 'm_gp_data.npy').exists():
            raise FileNotFoundError(f"""File 'm_gp_data.npy' does not exist
                                    in directory {self.data_dir}.""")
        else:
            paris_m = np.load(self.data_dir / 'm_gp_data.npy',
                              allow_pickle=False)
            # Reshape the Paris m coefficient array
            paris_m = paris_m[:, 0]
        # Check if file with average cycles
        # per load realisation exists and load it
        if not (self.data_dir / 'navg_ind.npy').exists():
            raise FileNotFoundError(f"""File 'navg_ind.npy' does not exist
                                    in directory {self.data_dir}.""")
        else:
            avg_cycles = np.load(self.data_dir / 'navg_ind.npy',
                                 allow_pickle=False)
        # Check if file with initial crack length
        # exists and load it
        if not (self.data_dir / 'a0_gp.npy').exists():
            raise FileNotFoundError(f"""File 'a0_gp.npy' doesn't
                                    exist in directory {self.data_dir}.""")
        else:
            initial_crack_length = np.load(self.data_dir / 'a0_gp.npy',
                                           allow_pickle=False)
            # Reshape the initial crack length array
            initial_crack_length = initial_crack_length[:, 0]
        # Load the stress ranges
        if not (self.data_dir / 'stress_range.npy').exists():
            raise FileNotFoundError(f"""File 'stress_range.npy' does not
                                    exist in directory {self.data_dir}.""")
        else:
            stress_ranges = np.load(self.data_dir / 'stress_range.npy',
                                    allow_pickle=False)
        # Pack the data into a dictionary
        data = {
            'times': times,
            'avg_cycles': avg_cycles,
            'stress_ranges': stress_ranges,
            'crack_lengths': crack_lengths,
            'paris_c': paris_c,
            'paris_m': paris_m,
            'initial_crack_length': initial_crack_length
        }
        return data


class TrajectorySelector:
    """
    This class is responsible for selecting crack
    growth trajectories (or instance) from the loaded dataset
    that correspond to specific Paris' law parameters.

    This class provides methods to select trajectories using different
    selection schemes based on assessing similarity to target
    Paris' law parameters.

    This class returns crack growth trajectories that can be used
    for parameter identification within a hierarchical Bayesian
    or multi-task learning framework.
    """

    def __init__(self, data: Dict[str, np.ndarray]):
        """
        Initializes the TrajectorySelector with the loaded data
        from the SCGDataLoader.

        Parameters
        ----------
        data : Dict[str, np.ndarray]
            A dictionary containing the loaded data from SCGDataLoader.
            The keys should match those defined in SCGDataLoader.load_data().
        """
        self.data = data
        # Check if the data contains the required keys
        required_keys = ['times', 'avg_cycles', 'stress_ranges',
                         'crack_lengths', 'paris_c', 'paris_m',
                         'initial_crack_length']
        for key in required_keys:
            if key not in self.data:
                raise KeyError(f"""Data is missing required key: {key}""")
        # Extract Paris' law parameters from the data - other keys are not used
        # required for trajectory selection
        self.paris_c = self.data['paris_c']
        self.paris_m = self.data['paris_m']

        # Calculate parameter statistics used by the selection methods
        self._calculate_parameter_stats()
        # Set target parameters for selection
        self._set_target_parameters()

    def _calculate_parameter_stats(self) -> None:
        """
        Calculates the mean, standard deviation and CoV
        of the Paris' law parameters (C and m) from the loaded data.
        """
        # Log-transform C for better numerical stability
        self._logc = np.log(self.paris_c)
        # Calculate means and standard deviations
        self._mean_logc = np.mean(self._logc)
        self._std_logc = np.std(self._logc)
        self._cov_logc = self._std_logc / self._mean_logc \
            if self._mean_logc != 0 else 0
        self._mean_m = np.mean(self.paris_m)
        self._std_m = np.std(self.paris_m)
        self._cov_m = self._std_m / self._mean_m if self._mean_m != 0 else 0

    def _set_target_parameters(self, scale_logc: float = 1.0,
                               scale_m: float = 1.0):
        """
        Sets the target Paris' law parameters for trajectory selection.
        The parameters are selected by shifting them their means using
        their corresponding CoVs, scaled by the provided scale factors.
        The target parameters are set as follows:
        - target_c = (1 + scale_logc * cov_logc) * mean_logc
        - target_m = (1 + scale_m * cov_m) * mean_m

        Parameters
        ----------
        scale_logc : float, optional
            Scaling factor for the log(C) parameter, by default 1.0
        scale_m : float, optional
            Scaling factor for the m parameter, by default 1.0
        """
        self.target_logc = (1 + scale_logc * self._cov_logc) * self._mean_logc
        self.target_m = (1 + scale_m * self._cov_m) * self._mean_m

    def select_by_radius(self, radius: float = 0.01) -> List[int]:
        """
        Selects trajectories by checking if they fall within a radius
        from the targets, defined as a fraction of their corresponding
        standard deviations. The radius is calculated as:
        - radius_c = radius * std_logc
        - radius_m = radius * std_m

        Parameters
        ----------
        radius : float, optional
            The fraction of the standard deviation to use as the radius for
            selection, by default 0.01

        Returns
        -------
        List[int]
            A list of indices of the selected trajectories.
        """
        # Get the indices of the trajectories that fall within the radius.
        hits = []

        for i in range(self.paris_m.shape[0]):
            if (np.abs(self.paris_m[i] - self.target_m) < radius*self._std_m) \
             and (np.abs(self._logc[i] - self.target_logc)
                  < radius*self._std_logc):
                hits.append(i)
        return hits

    def extract_trajectories(self,
                             indices: List[int]) -> Dict[str, np.ndarray]:
        """
        Extract crack growth trajectories for the specified indices.

        This method cleans up the trajectories by removing any zero-padding
        that may exist in the crack length data.

        Parameters
        ----------
        indices : List[int]
            The indices of the trajectories to extract

        Returns
        -------
        Dict[str, Any]
            Dictionary containing extracted data for the selected trajectories:
            - 'times': List of time arrays, one per trajectory
            - 'crack_lengths': List of crack length arrays, one per trajectory
            - 'paris_c': Paris C parameters for selected trajectories
            - 'paris_m': Paris m parameters for selected trajectories
            - 'initial_crack_length': Initial crack lengths
            for selected trajectories
        """
        times = []
        crack_lengths = []
        paris_c = self.paris_c[indices]
        paris_m = self.paris_m[indices]
        initial_crack_length = self.data['initial_crack_length'][indices]

        # Extract and clean up trajectories
        for idx in indices:
            # Extract the time points and crack length data
            time_data = self.data['times'][idx]
            crack_data = self.data['crack_lengths'][idx]

            # Find non-zero elements (to remove padding)
            non_zero_indices = np.where(crack_data > 0)[0]

            # Only include non-zero data points
            if len(non_zero_indices) > 0:
                # +1 to include the last non-zero element
                last_idx = non_zero_indices[-1] + 1
                times.append(time_data[:last_idx])
                crack_lengths.append(crack_data[:last_idx])
            else:
                # If all zeros (shouldn't happen),
                # include only the initial point
                times.append(time_data[0:1])
                crack_lengths.append(crack_data[0:1])

        # Return all extracted data
        return {
            'times': times,
            'crack_lengths': crack_lengths,
            'paris_c': paris_c,
            'paris_m': paris_m,
            'initial_crack_length': initial_crack_length
        }


class CrackObservationGenerator:
    """
    This class is responsible for sub-sampling crack growth trajectories
    and adding noise to create synthetic crack length observations.

    It takes crack growth trajectories (typically from
    TrajectorySelector.extract_trajectories())
    and creates observations with configurable
    sampling strategies and noise models.
    These observations can be used as inputs to
    Bayesian models for parameter identification.
    """

    def __init__(self, trajectories: Dict[str, Any],
                 random_seed: Optional[int] = None):
        """
        Initialize the CrackObservationGenerator with specific
        crack growth trajectories.

        Parameters
        ----------
        trajectories : Dict[str, Any]
            Dictionary containing crack growth trajectories, typically from
            TrajectorySelector.extract_trajectories(). The dictionary should
            have the following structure:
            - 'times': List of arrays, one per trajectory,
            containing time points
            - 'crack_lengths': List of arrays, one per trajectory,
            containing crack lengths
            - 'paris_c': Array of Paris law C parameters for each trajectory
            - 'paris_m': Array of Paris law m parameters for each trajectory
            - 'initial_crack_length': Array of initial crack
            lengths for each trajectory
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.trajectories = trajectories
        self.rng = np.random.RandomState(random_seed)

    def sample_trajectories(self,
                            n_points: Union[int, List[int]] = 10,
                            strategy: str = 'uniform',
                            include_endpoints: bool = True
                            ) -> Dict[str, List[np.ndarray]]:
        """
        Sample points from each trajectory according to the specified strategy.

        Parameters
        ----------
        n_points : int or List[int], optional
            Number of points to sample from each trajectory. If a list,
            must have the same length as the number of trajectories.
        strategy : str, optional
            Sampling strategy: 'uniform' or 'random'
            - 'uniform': Evenly spaced points along the timeline
            - 'random': Randomly sampled points from the timeline
        include_endpoints : bool, optional
            Whether to always include the first
            and last points of each trajectory

        Returns
        -------
        Dict[str, List[np.ndarray]]
            Dictionary containing sampled times
            and crack lengths for each trajectory
        """
        times = self.trajectories['times']
        crack_lengths = self.trajectories['crack_lengths']

        # Convert n_points to a list if it's a single integer
        if isinstance(n_points, int):
            n_points = [n_points] * len(times)
        elif len(n_points) != len(times):
            raise ValueError("If n_points is a list, it must have the same \
                             length as the number of trajectories")

        sampled_times = []
        sampled_crack_lengths = []

        for i, (time_arr, crack_arr) in enumerate(zip(times, crack_lengths)):
            num_samples = n_points[i]

            # Ensure we don't try to sample more points than available
            if num_samples > len(time_arr):
                raise ValueError(f"Cannot sample {num_samples} points \
                                  from trajectory {i} with only \
                                    {len(time_arr)} points")

            # Apply the sampling strategy
            if strategy == 'uniform':
                # For uniform sampling, we create
                # indices that are evenly spaced
                if include_endpoints:
                    if num_samples > 2:
                        # Create evenly spaced indices including endpoints
                        indices = np.linspace(0, len(time_arr) - 1,
                                              num_samples, dtype=int
                                              )
                    else:
                        # If only 1 or 2 points requested and
                        # include_endpoints is True, we include the endpoints
                        indices = np.array(
                            [0, len(time_arr) - 1][:num_samples]
                            )
                else:
                    # Create evenly spaced indices excluding endpoints
                    indices = np.linspace(
                        0, len(time_arr) - 1, num_samples + 2, dtype=int
                        )[1:-1]

            elif strategy == 'random':
                # For random sampling, we randomly select indices
                indices = self.rng.choice(
                    len(time_arr), size=num_samples, replace=False
                    )
                indices.sort()  # Sort to maintain chronological order

                # Add endpoints if required
                if include_endpoints:
                    if 0 not in indices:
                        indices = np.append([0], indices)
                    if len(time_arr) - 1 not in indices:
                        indices = np.append(indices, [len(time_arr) - 1])
                    # If we now have too many points, randomly remove some
                    # (but not endpoints)
                    if len(indices) > num_samples:
                        middle_indices = indices[1:-1]
                        to_remove = len(indices) - num_samples
                        remove_indices = self.rng.choice(
                            len(middle_indices), size=to_remove, replace=False
                            )
                        # +1 because we skip the first index
                        indices = np.delete(indices, remove_indices + 1)

            else:
                raise ValueError(f"Unknown sampling strategy: {strategy}. \
                                  Use 'uniform' or 'random'.")

            # Sample times and crack lengths
            sampled_times.append(time_arr[indices])
            sampled_crack_lengths.append(crack_arr[indices])

        return {
            'times': sampled_times,
            'crack_lengths': sampled_crack_lengths,
            'paris_c': self.trajectories['paris_c'],
            'paris_m': self.trajectories['paris_m'],
            'initial_crack_length': self.trajectories['initial_crack_length']
        }

    def add_observation_noise(self,
                              trajectories: Dict[str, List[np.ndarray]],
                              std_dev: float = 1.0,
                              random_seed: Optional[int] = None
                              ) -> Dict[str, List[np.ndarray]]:
        """
        Add Gaussian noise to the crack length observations.

        Parameters
        ----------
        trajectories : Dict[str, List[np.ndarray]]
            Dictionary containing trajectory data,
            typically from sample_trajectories()
        std_dev : float, optional
            Standard deviation of the Gaussian noise to add
        random_seed : int, optional
            Random seed for noise generation

        Returns
        -------
        Dict[str, List[np.ndarray]]
            Dictionary with the original data plus noisy observations
        """
        # Create a copy of the input to avoid modifying the original
        result = trajectories.copy()

        # Create noise model
        noise_model = GaussianNoiseObservation(
            std_dev=std_dev, random_seed=random_seed
            )

        # Add noise to each trajectory
        noisy_crack_lengths = []
        for crack_arr in trajectories['crack_lengths']:
            noisy_arr = np.array([noise_model.observe(x) for x in crack_arr])
            noisy_crack_lengths.append(noisy_arr)

        # Add the noisy observations to the result
        result['noisy_crack_lengths'] = noisy_crack_lengths

        return result

    def create_observations(self,
                            n_points: Union[int, List[int]] = 10,
                            strategy: str = 'uniform',
                            std_dev: float = 1.0,
                            include_endpoints: bool = True,
                            random_seed: Optional[int] = None
                            ) -> Dict[str, List[np.ndarray]]:
        """
        Combined method to sample trajectories and add noise in one step.

        Parameters
        ----------
        n_points : int or List[int], optional
            Number of points to sample from each trajectory
        strategy : str, optional
            Sampling strategy: 'uniform', 'random', or 'log'
        std_dev : float, optional
            Standard deviation of the Gaussian noise to add
        include_endpoints : bool, optional
            Whether to always include the first and last points
        random_seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        Dict[str, List[np.ndarray]]
            Dictionary containing original and noisy sampled data
        """
        # Sample the trajectories
        sampled = self.sample_trajectories(
            n_points=n_points,
            strategy=strategy,
            include_endpoints=include_endpoints
        )

        # Add noise with the specified random seed
        observations = self.add_observation_noise(
            trajectories=sampled,
            std_dev=std_dev,
            random_seed=random_seed
        )

        return observations
