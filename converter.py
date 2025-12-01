import os
import pandas as pd
import numpy as np


def csv_to_bin(file_path: str, save_path: str = None):
    """
        - convert a CSV file into an interleaved binary .bin file for the software to load

        @Parameters
        file_path : str
            - path to the input CSV file
        save_path : str, optional
            - target path for the output file

        @Returns
        str
            - path to the created .bin file

        @Raises
        FileNotFoundError
            -if the CSV file does not exist
        ValueError
            - if the CSV file does not contain the required columns
        """ 

    # make sure that the input CSV exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    # read the file
    df = pd.read_csv(file_path, low_memory=True)

    # check that the expected columns are present
    if "adc1" not in df.columns or "adc2" not in df.columns:
        raise ValueError("The csv file should have columns 'adc1' and 'adc2'.")

    # convert the columns to 16-bit integers
    signal_1 = np.asarray(df["adc1"], dtype=np.int16)
    signal_2 = np.asarray(df["adc2"], dtype=np.int16)
    # interleave the two signals column-wise -> adc1, adc2 per sample
    interleaved = np.column_stack((signal_1, signal_2))

    # if no save path is given, place the .bin next to the CSV
    if save_path is None:
        save_path = os.path.splitext(file_path)[0] + ".bin"

    # write the data to disk and return save path
    interleaved.tofile(save_path)
    return save_path
