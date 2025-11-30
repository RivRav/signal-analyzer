from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QCheckBox, QMenuBar, QMenu, QFrame, QProgressBar
)
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from superqt import QRangeSlider
from PySide6.QtGui import QIcon


class MainWindow(QMainWindow):
    """
    - main app window that has all of the user interface elements
    - a left-side control panel with file information, peak counts, and visualization toggles
    - a large central area for plotting + menu bar, toolbar, progress bar, and range slider
    """

    def __init__(self):
        super().__init__()
        # window settings
        self.setWindowIcon(QIcon("icon.ico"))
        self.setWindowTitle("Signal analyzer")
        self.resize(1100, 700)

        # menu bar
        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)

        self.file_menu = QMenu("File", self)
        self.menu_bar.addMenu(self.file_menu)

        self.open_action = self.file_menu.addAction("Open file")
        self.convert_action = self.file_menu.addAction("Convert a file to .bin format")
        self.exit_action = self.file_menu.addAction("Exit")

        # main (horizontal) layout
        main_layout = QHBoxLayout()
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # left panel
        left_panel = QVBoxLayout()
        main_layout.addLayout(left_panel, 1)
        left_panel.addSpacing(10)

        # file info
        file_info_title = QLabel("File info")
        file_info_title.setProperty("role", "title")
        left_panel.addWidget(file_info_title)

        self.file_label = QLabel("Selected file: None")
        left_panel.addWidget(self.file_label)

        self.file_size_label = QLabel("File size: --")
        left_panel.addWidget(self.file_size_label)

        self.data_points_label = QLabel("Data points: --")
        left_panel.addWidget(self.data_points_label)

        self.signal_time_label = QLabel("Signal duration: --")
        left_panel.addWidget(self.signal_time_label)

        # horizontal line separator (for next section)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("margin: 6px 0;")
        left_panel.addWidget(line)

        # peaks info
        peaks_title = QLabel("Peaks")
        peaks_title.setProperty("role", "title")
        left_panel.addWidget(peaks_title)

        self.peaks_1_count_label = QLabel("Signal 1 tumor peaks: --")
        left_panel.addWidget(self.peaks_1_count_label)

        self.peaks_2_count_label = QLabel("Signal 2 tumor peaks: --")
        left_panel.addWidget(self.peaks_2_count_label)

        # separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("margin: 6px 0;")
        left_panel.addWidget(line)

        # plot controls
        plot_controls_title = QLabel("Plot controls")
        plot_controls_title.setProperty("role", "title")
        left_panel.addWidget(plot_controls_title)

        self.peaks_checkbox = QCheckBox("Show peaks")
        self.peaks_checkbox.setChecked(True)
        self.peaks_checkbox.setEnabled(False)
        left_panel.addWidget(self.peaks_checkbox)

        # layer toggles
        self.water_checkbox = QCheckBox("Show water")
        self.water_checkbox.setChecked(False)
        self.water_checkbox.setEnabled(False)
        left_panel.addWidget(self.water_checkbox)

        self.baseline_checkbox = QCheckBox("Show baseline")
        self.baseline_checkbox.setChecked(False)
        self.baseline_checkbox.setEnabled(False)
        left_panel.addWidget(self.baseline_checkbox)

        self.signal_1_checkbox = QCheckBox("Signal 1")
        self.signal_1_checkbox.setChecked(True)
        self.signal_1_checkbox.setEnabled(False)
        left_panel.addWidget(self.signal_1_checkbox)

        self.signal_2_checkbox = QCheckBox("Signal 2")
        self.signal_2_checkbox.setChecked(True)
        self.signal_2_checkbox.setEnabled(False)
        left_panel.addWidget(self.signal_2_checkbox)

        # fill vertical space so status and progress bar stay at bottom
        left_panel.addStretch()

        # status label for app activity
        self.status_label = QLabel("Status: idle")
        left_panel.addWidget(self.status_label)

        # progress bar for app activity
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.hide()
        left_panel.addWidget(self.progress_bar)

        # right panel
        right_panel = QVBoxLayout()
        main_layout.addLayout(right_panel, 4)

        # canvas for signal drawing
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        # toolbar (zoom, pan, save)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_panel.addWidget(self.toolbar)
        right_panel.addWidget(self.canvas)

        # range slider for plot view (0–100% of the data)
        self.range_slider = QRangeSlider(Qt.Horizontal)
        self.range_slider.setRange(0, 100)
        self.range_slider.setValue((0, 100))
        right_panel.addWidget(self.range_slider)

        # slider label/button + reset button
        self.slider_label = QLabel("Range: 0% – 100%")
        self.range_button = QPushButton("Reset range")
        self.range_button.setFixedSize(100, 30)
        range_control_row = QHBoxLayout()
        range_control_row.addWidget(self.slider_label)
        range_control_row.addStretch()
        range_control_row.addWidget(self.range_button)
        right_panel.addLayout(range_control_row)
