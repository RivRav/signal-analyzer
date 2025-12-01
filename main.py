import sys
import os
import pandas as pd
import numpy as np
from PySide6.QtWidgets import QApplication, QFileDialog
from PySide6.QtCore import QThread
from workers import PeakWorker
from formatters import format_size, format_time
from algorithms import minmax_downsample
from converter import csv_to_bin
from main_window import MainWindow


class SignalAnalyzer(MainWindow):
    """- Main app window that connects everything"""
        
    def __init__(self):
        super().__init__()

        # internal state holders
        self.worker = None
        self.thread = None
        # x-axis index array
        self.i = None
        self.file_path = None
        self.signal_1 = None
        self.signal_2 = None
        self.time = None
        # sampling rate
        self.sampling_rate = 50000
        self.plotting_start_index = 0
        self.plotting_end_index = None
        # values after peak detection
        self.s1_tumor_peaks = np.array([])
        self.s2_tumor_peaks = np.array([])
        self.s1_water_peaks = np.array([])
        self.s2_water_peaks = np.array([])
        self.baseline_1 = None
        self.baseline_2 = None

        # UI menu actions
        self.open_action.triggered.connect(self.load_data)
        self.convert_action.triggered.connect(self.convert_file_to_bin)
        self.exit_action.triggered.connect(self.close)
        # UI toggles
        self.peaks_checkbox.stateChanged.connect(self.on_checkbox_toggle)
        self.water_checkbox.stateChanged.connect(self.on_checkbox_toggle)
        self.baseline_checkbox.stateChanged.connect(self.on_checkbox_toggle)
        self.signal_1_checkbox.stateChanged.connect(self.on_checkbox_toggle)
        self.signal_2_checkbox.stateChanged.connect(self.on_checkbox_toggle)
        # range slider
        self.range_slider.valueChanged.connect(self.on_slider_change)
        self.range_button.clicked.connect(self.reset_slider_range)

    def show_progress_busy(self, text):
        """
        - show a progress bar while doing a task

        @Parameters
        text : str
            - status text
        """

        self.status_label.setText(text)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()

    def update_progress(self, value, text=None):
        """
        - update the progress bar state

        @Parameters
        value : int
            - progress value (0-100)
        text : str, optional
            - new status if given
        """

        if self.progress_bar.minimum() == 0 and self.progress_bar.maximum() == 0:
            self.progress_bar.setRange(0, 100)

        self.progress_bar.setValue(value)
        if text is not None:
            self.status_label.setText(text)

    def hide_progress(self, final_text="Status: idle"):
        """
        - hide the progress bar when idle

        @Parameters
        final_text : str, optional
            - status text
        """

        self.progress_bar.hide()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText(final_text)

    def on_worker_progress(self, value, message):
        """
        - progress messages to UI

        @Parameters
        value : int
            - progress value
        message : str
            - worker state msg
        """

        self.update_progress(value, message)

    def load_data(self):
        """- load a file and initialize signals, do the time axis, and start the peak detection thread"""

        start_dir = os.path.expanduser("~/Documents")

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Data File",
            start_dir,
            "Data files (*.bin *.csv)"
        )

        if not file_path:
            print("No file selected.")
            return

        self.file_path = file_path
        file_name = os.path.basename(file_path)
        self.file_label.setText(f"Selected file: {file_name}")
        print("Loading file:", file_path)

        # disable UI until data and peaks are ready
        self.signal_1_checkbox.setEnabled(False)
        self.signal_2_checkbox.setEnabled(False)
        self.baseline_checkbox.setEnabled(False)
        self.peaks_checkbox.setEnabled(False)
        self.water_checkbox.setEnabled(False)

        self.reset_slider_range()

        self.show_progress_busy("Loading data...")

        file_type = os.path.splitext(file_path)[1].lower()
        file_size = os.path.getsize(file_path)
        file_size_str = format_size(file_size)

        try:
            # ability to load csv even if binary preffered
            if file_type == ".csv":

                self.i = pd.read_csv(file_path, low_memory=True)
                number_of_points = len(self.i)
                self.signal_1 = self.i["adc1"].values
                self.signal_2 = self.i["adc2"].values

            # load binary with memmap
            elif file_type == ".bin":

                number_of_samples = file_size // (2 * np.dtype(np.int16).itemsize)
                raw_data = np.memmap(file_path, dtype=np.int16, mode="r", shape=(number_of_samples, 2))
                raw_data = np.array(raw_data, copy=True)
                self.signal_1 = raw_data[:, 0]
                self.signal_2 = raw_data[:, 1]
                number_of_points = len(raw_data)
                self.i = pd.DataFrame({"adc1": self.signal_1, "adc2": self.signal_2})

            else:
                print("This file type is not supported.")
                return

            print("File loaded!")

            # get duration
            signal_time_sec = number_of_points / self.sampling_rate
            signal_time_str = format_time(signal_time_sec)

            # update UI labels
            self.file_size_label.setText(f"File size: {file_size_str}")
            self.data_points_label.setText(f"Data points: {number_of_points:,}")
            self.signal_time_label.setText(f"Signal duration: {signal_time_str}")

            # x-axis index and time array
            self.plotting_end_index = len(self.i)
            self.i = np.arange(len(self.i))
            self.time = self.i / self.sampling_rate

            # draw init plot
            self.plot_data()

            # enable signal visibility toggles
            self.signal_1_checkbox.setEnabled(True)
            self.signal_2_checkbox.setEnabled(True)

            # start worker for peak detection (when done, pass the data and clean)
            self.thread = QThread()
            self.worker = PeakWorker(self.signal_1, self.signal_2)
            self.worker.moveToThread(self.thread)
            self.worker.progress.connect(self.on_worker_progress)
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.on_peaks_detection_finished)
            self.worker.error.connect(self.on_peaks_detection_error)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)

            self.update_progress(0, "Detecting peaks...")

            self.thread.start()

        except Exception as e:
            print("Error loading file:", e)
            self.hide_progress("Error while loading")

    def on_peaks_detection_finished(self, tumor_peaks_1, tumor_peaks_2, water_peaks_1, water_peaks_2, baseline_1, baseline_2):
        """
        - called when the worker has detected peaks. Stores results and updates UI

        @Parameters
        tumor_peaks_1 : np.ndarray
            - tumor peaks detected in signal 1
        tumor_peaks_2 : np.ndarray
            - tumor peaks detected in signal 2
        water_peaks_1 : np.ndarray
            - water peaks detected in signal 1
        water_peaks_2 : np.ndarray
            - water peaks detected in signal 2
        baseline_1 : np.ndarray
            - baseline values based on signal 1
        baseline_2 : np.ndarray
            - baseline values based on signal 2
        """

        self.s1_tumor_peaks = tumor_peaks_1
        self.s2_tumor_peaks = tumor_peaks_2
        self.s1_water_peaks = water_peaks_1
        self.s2_water_peaks = water_peaks_2
        self.baseline_1 = baseline_1
        self.baseline_2 = baseline_2

        # update left panel
        self.peaks_1_count_label.setText(f"Signal 1 tumor peaks: {len(tumor_peaks_1)}")
        self.peaks_2_count_label.setText(f"Signal 2 tumor peaks: {len(tumor_peaks_2)}")

        # update plot with peaks
        self.plot_current_range()

        # enable visibility toggles
        self.peaks_checkbox.setEnabled(True)
        self.water_checkbox.setEnabled(True)
        self.baseline_checkbox.setEnabled(True)

        self.hide_progress("Peaks are ready")

    def on_peaks_detection_error(self, message):
        """
        - show error when worker fails

        @Parameters
        message : str
            - error msg
        """

        print("Error detecting peaks:", message)
        self.hide_progress("Error during peak detection")

    def convert_file_to_bin(self):
        """- UI dialog for converting CSV to .bin"""

        self.show_progress_busy("Converting to .bin...")

        start_dir = os.path.expanduser("~/Documents")
        file_path, _ = QFileDialog.getOpenFileName(self,"Select a file", start_dir,"All files (*)")

        if not file_path:
            print("No file selected.")
            self.hide_progress("Conversion cancelled")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save as .bin",
            os.path.splitext(file_path)[0] + ".bin",
            "Binary files (*.bin)"
        )
        if not save_path:
            print("Save cancelled.")
            self.hide_progress("Conversion cancelled")
            return

        try:
            saved_file = csv_to_bin(file_path, save_path)
            print(f"File converted and saved to: {saved_file}")
            self.hide_progress("Conversion finished")
        except Exception as e:
            print("Error converting file:", e)
            self.hide_progress("Conversion error")


    def get_plotting_range(self):
        """
        - return currently selected data subset

        @Returns
        tuple
            x_range : np.ndarray
                - sample indices
            y1_range : np.ndarray
                - signal 1 values
            y2_range : np.ndarray
                - signal 2 values
            start : int
                - start index
            end : int
                - end index
        """

        start = self.plotting_start_index
        end = self.plotting_end_index
        x_range = self.i[start:end]
        y1_range = self.signal_1[start:end]
        y2_range = self.signal_2[start:end]
        return x_range, y1_range, y2_range, start, end

    def plot_current_range(self):
        """- plot only selected range"""

        self.plot_signals(*self.get_plotting_range())

    def plot_data(self):
        """- plot data after loading"""

        if self.i is None:
            print("Load a data file first before plotting")
            return

        self.plot_current_range()

    def plot_signals(self, row_indexes, signal_1, signal_2, plotting_start_index, plotting_end_index):
        """
        - draw signals, baseline and peaks

        @Parameters
        row_indexes : np.ndarray
            - selected range indices
        signal_1 : np.ndarray
            - signal 1 values
        signal_2 : np.ndarray
            - signal 2 values
        plotting_start_index : int
            - start sample
        plotting_end_index : int
            - end sample 
        """

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # signal 1
        if self.signal_1_checkbox.isChecked() and signal_1 is not None:
            time_range = self.time[plotting_start_index:plotting_end_index]
            x_down, y_down = minmax_downsample(time_range, signal_1, canvas_width=self.canvas.width())
            ax.plot(x_down, y_down, color="cornflowerblue", label='Signal 1', linewidth=0.8)

            # baseline
            if self.baseline_checkbox.isChecked() and self.baseline_1 is not None:
                baseline_range = self.baseline_1[plotting_start_index:plotting_end_index]
                time_range = self.time[plotting_start_index:plotting_end_index]
                x_down_b, y_down_b = minmax_downsample(time_range, baseline_range, canvas_width=self.canvas.width())
                ax.plot(x_down_b, y_down_b, color='black', linewidth=1.5)
            
            # tumor peaks
            if self.peaks_checkbox.isChecked():
                tumor_peaks_in_range = self.s1_tumor_peaks[(self.s1_tumor_peaks >= plotting_start_index) & (self.s1_tumor_peaks < plotting_end_index)]
                if len(tumor_peaks_in_range) > 0:
                    ax.plot(self.time[tumor_peaks_in_range], self.signal_1[tumor_peaks_in_range], color='red', marker='o', linestyle='None', label="Signal 1 tumor peaks")
            
            # water peaks
            if self.water_checkbox.isChecked():
                water_peaks_in_range = self.s1_water_peaks[(self.s1_water_peaks >= plotting_start_index) & (self.s1_water_peaks < plotting_end_index)]
                if len(water_peaks_in_range) > 0:
                    ax.plot(self.time[water_peaks_in_range], self.signal_1[water_peaks_in_range], color='blue', marker='v', linestyle='None')

        # signal 2
        if self.signal_2_checkbox.isChecked() and signal_2 is not None:
            time_range = self.time[plotting_start_index:plotting_end_index]
            x_down, y_down = minmax_downsample(time_range, signal_2, canvas_width=self.canvas.width())
            ax.plot(x_down, y_down, color="orange", label='Signal 2', linewidth=0.8, alpha=0.9)

            # baseline
            if self.baseline_checkbox.isChecked() and self.baseline_2 is not None:
                baseline_range = self.baseline_2[plotting_start_index:plotting_end_index]
                time_range = self.time[plotting_start_index:plotting_end_index]
                x_down_b, y_down_b = minmax_downsample(time_range, baseline_range, canvas_width=self.canvas.width())
                ax.plot(x_down_b, y_down_b, color='black', linewidth=1.5, label='Baseline')

            # tumor peaks
            if self.peaks_checkbox.isChecked():
                tumor_peaks_in_range = self.s2_tumor_peaks[
                    (self.s2_tumor_peaks >= plotting_start_index) & (self.s2_tumor_peaks < plotting_end_index)]
                if len(tumor_peaks_in_range) > 0:
                    ax.plot(self.time[tumor_peaks_in_range], self.signal_2[tumor_peaks_in_range], color='green', marker='o', linestyle='None',
                            label="Signal 2 tumor peaks")

            # water peaks
            if self.water_checkbox.isChecked():
                water_peaks_in_range = self.s2_water_peaks[
                    (self.s2_water_peaks >= plotting_start_index) & (self.s2_water_peaks < plotting_end_index)]
                if len(water_peaks_in_range) > 0:
                    ax.plot(self.time[water_peaks_in_range], self.signal_2[water_peaks_in_range], color='blue', marker='v', linestyle='None',
                            label="Water")

        ax.set_title("Signal data")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Response")
        ax.legend()

        # redraw canvas
        self.canvas.draw()

    def on_slider_change(self, values):
        """
        - range slider change callback -> update view

        @Parameters
        values : tuple
            - a start/end in percent for slider range
        """

        start, end = values
        self.slider_label.setText(f"Range: {start}% – {end}%")

        if self.i is not None:
            total_data_points = len(self.i)
            self.plotting_start_index = int(start / 100 * total_data_points)
            self.plotting_end_index = int(end / 100 * total_data_points)
            if self.plotting_start_index < self.plotting_end_index:
                self.plot_current_range()

    def reset_slider_range(self):
        """ - reset slider and redraw plot"""

        if self.i is None:
            return

        self.range_slider.setValue((0, 100))
        self.slider_label.setText("Range: 0% - 100%")
        self.plotting_start_index = 0
        self.plotting_end_index = len(self.i)
        self.plot_current_range()

    def on_checkbox_toggle(self, _):
        """ - plot checkbox callback -> replot depending on settings"""

        if self.i is not None:
            self.plot_current_range()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # stylesheet
    with open("style.qss", "r") as file:
        app.setStyleSheet(file.read())

    window = SignalAnalyzer()
    window.show()
    sys.exit(app.exec())