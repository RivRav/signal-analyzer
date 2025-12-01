from PySide6.QtCore import QObject, Signal
import numpy as np
from algorithms import find_peaks, compute_baseline


class PeakWorker(QObject):
    """- a class responsible for running baseline calculation and peak detection in a background thread (both signals separately and gives results back to the GUI when done)"""
    
    # update peak/baseline info + error
    finished = Signal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    error = Signal(str)
    # update progress bar (percentage, status msg)
    progress = Signal(int, str)

    def __init__(self, signal_1: np.ndarray, signal_2: np.ndarray):
        """
        - keep references to the two signals + configure section size

        @Parameters
        signal_1 : np.ndarray
            - 1st input signal
        signal_2 : np.ndarray
            - 2nd input signal
        """

        # amount of samples processed per loop iteration
        super().__init__()
        self.signal_1 = signal_1
        self.signal_2 = signal_2
        self.section_size = 200000

    def process_signal(self, signal: np.ndarray):
        """
        - calculate the baseline and detect peaks for a signal (done section by section)

        @Parameters
        signal : np.ndarray
            - the input signal to process

        @Returns
        tuple
            tumor_peaks : np.ndarray
                - tumor peaks in the full signal
            water_peaks : np.ndarray
                - water peaks in the full signal
            full_baseline : np.ndarray
                - baseline values aligned with the full signal length
        """

        signal_length = len(signal)

        baselines = []
        all_tumor_peaks = []
        all_water_peaks = []
        # total standard deviation for threshold comparisons
        total_signal_sd = np.std(signal)

        # iterate in chunks
        for sect in range(0, signal_length, self.section_size):
            start = sect
            end = min(start + self.section_size, signal_length)
            section = signal[start:end]

            section_sd = np.std(section)

            # a flat baseline for the curr section
            section_bl = compute_baseline(section)
            baselines.append(np.full_like(section, section_bl, dtype=float))

            # extend the boundaries to avoid cutting off peaks at edges
            extend = 10000
            start_extended = max(0, start - extend)
            end_extended = min(signal_length, end + extend)
            extended_section = signal[start_extended:end_extended]

            # detect peaks within extended region
            tumor_peaks_from_ext, water_peaks_from_ext = find_peaks(
                extended_section,
                baseline=section_bl,
                section_sd=section_sd,
                signal_total_sd=total_signal_sd
            )

            # shift the local peaks back into full-signal
            tumor_peaks_from_ext = tumor_peaks_from_ext + start_extended
            water_peaks_from_ext = water_peaks_from_ext + start_extended

            # keep only peaks that are inside the og section
            tumor_peaks = tumor_peaks_from_ext[(tumor_peaks_from_ext >= start) & (tumor_peaks_from_ext < end)]
            water_peaks = water_peaks_from_ext[(water_peaks_from_ext >= start) & (water_peaks_from_ext < end)]

            # add to global lists
            all_tumor_peaks.extend(tumor_peaks)
            all_water_peaks.extend(water_peaks)

        # merge baselines from each section
        full_baseline = np.concatenate(baselines)
        return np.array(all_tumor_peaks), np.array(all_water_peaks), full_baseline


    def run(self):
        """- run the processing for both signals (section process and updating progress bar)"""

        try:
            print("Baseline computation and peak detection started.")

            # 1st signal
            self.progress.emit(0, "Processing signal 1...")
            tumor_peaks_1, water_peaks_1, baseline_1 = self.process_signal(self.signal_1)

            # 2nd signal
            self.progress.emit(50, "Processing signal 2...")
            tumor_peaks_2, water_peaks_2, baseline_2 = self.process_signal(self.signal_2)

            print("Baseline computation and peak detection ended.")

            self.progress.emit(100, "Peaks detected")

            # results to GUI
            self.finished.emit(
                tumor_peaks_1,
                tumor_peaks_2,
                water_peaks_1,
                water_peaks_2,
                baseline_1,
                baseline_2
            )

        except Exception as e:
            # forward any exception
            self.error.emit(str(e))
