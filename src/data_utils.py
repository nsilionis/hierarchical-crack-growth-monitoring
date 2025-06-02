import numpy as np
import pandas as pd
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

    def load_data(self) -> pd.DataFrame:
        """
        Loads the data from separate .npy files and combines them into a
        single pandas DataFrame.

        -------
        Returns
        """
        # Get the list of .npy files in the data directory
        npy_files = list(self.data_dir.glob('*.npy'))
        if not npy_files:
            raise FileNotFoundError("""No .npy files found in
                                    the data directory.""")
        # Load each .npy file and concatenate them into a single DataFrame
        data_frames = []
        for file in npy_files:
            data = np.load(file, allow_pickle=True)
            df = pd.DataFrame(data)
            data_frames.append(df)
        combined_df = pd.concat(data_frames, ignore_index=True)
        return combined_df
    