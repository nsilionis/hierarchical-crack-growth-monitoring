import numpy as np
from pathlib import Path


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

        -------
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
        - 'crack_lengths': crack lengths at each time instant (mm)
        - 'paris_c': The C coefficient of the Paris law
        - 'paris_m': The m coefficient of the Paris law
        - 'initial_crack_length': The initial crack length (mm)
        The values are 2D numpy arrays with varying shapes/sizes.

        -------
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
        # Pack the data into a dictionary
        data = {
            'times': times,
            'avg_cycles': avg_cycles,
            'crack_lengths': crack_lengths,
            'paris_c': paris_c,
            'paris_m': paris_m,
            'initial_crack_length': initial_crack_length
        }
        return data
