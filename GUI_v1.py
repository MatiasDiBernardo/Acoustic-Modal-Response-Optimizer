# BRO Acoustics
# Author: Alongi, Di Bernardo, Lucana, Vereertbrugghen
# Description: GUI for BRO Acoustics

import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout,
    QHBoxLayout, QTabWidget, QFormLayout, QCheckBox, QStackedLayout, QMessageBox,
    QGroupBox, QScrollArea, QFrame
)
from PyQt5.QtGui import QDoubleValidator, QIcon, QFont
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pyvista as pv
from pyvistaqt import QtInteractor


class BROAcousticsApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BRO Acoustics")
        self.setGeometry(100, 100, 1400, 800)

        self.layout = QVBoxLayout(self)
        self.tabs = QTabWidget()

        self.data_entry_tab = QWidget()
        self.instruction_manual_tab = QWidget()

        self.tabs.addTab(self.data_entry_tab, "Data Entry")
        self.tabs.addTab(self.instruction_manual_tab, "Instruction Manual")

        self.layout.addWidget(self.tabs)
        self.init_data_entry_tab()

    def init_data_entry_tab(self):
        layout = QHBoxLayout()

        # Left: Inputs
        input_frame = QFrame()
        input_layout = QVBoxLayout(input_frame)
        self.dim_inputs = {}
        self.tol_inputs = {}

        form_layout = QFormLayout()
        dimensions = ['Lx', 'Ly', 'Lz']
        for dim in dimensions:
            dim_input = QLineEdit()
            dim_input.setValidator(QDoubleValidator(0.01, 10.0, 2))
            dim_input.setPlaceholderText("m")
            dim_input.setToolTip(f"Enter {dim} in meters (max 10.0, decimal with .)")

            tol_input = QLineEdit()
            tol_input.setValidator(QDoubleValidator(0.01, 10.0, 2))
            tol_input.setPlaceholderText("m")
            tol_input.setToolTip(f"Enter uncertainty for {dim} in meters")

            self.dim_inputs[dim] = dim_input
            self.tol_inputs[dim] = tol_input

            row = QHBoxLayout()
            row.addWidget(dim_input)
            row.addWidget(QLabel("±"))
            row.addWidget(tol_input)

            group = QWidget()
            group.setLayout(row)
            form_layout.addRow(f"{dim}:", group)

        input_layout.addLayout(form_layout)

        # Listening point selection
        self.listen_label = QLabel("Position the listening point")
        self.listen_canvas = QLabel("[Graphical Selector Placeholder]")
        self.listen_canvas.setFixedSize(250, 250)
        self.listen_canvas.setStyleSheet("background-color: #e0e0e0; border: 1px solid #ccc;")

        self.listen_checkbox = QCheckBox("Manually select listening point")
        input_layout.addWidget(self.listen_label)
        input_layout.addWidget(self.listen_canvas)
        input_layout.addWidget(self.listen_checkbox)

        self.generate_btn = QPushButton("Generate")
        self.generate_btn.clicked.connect(self.run_simulation)
        input_layout.addWidget(self.generate_btn)

        layout.addWidget(input_frame)

        # Right: Results area
        result_frame = QFrame()
        result_layout = QVBoxLayout(result_frame)

        self.result_tabs = QTabWidget()

        # Placeholder tab
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        tab1_layout.addWidget(self.canvas)

        self.plot_checkbox = QCheckBox("Show curve 1")
        tab1_layout.addWidget(self.plot_checkbox)

        # PyVista window
        self.plotter = QtInteractor()
        tab1_layout.addWidget(self.plotter.interactor)

        self.result_tabs.addTab(tab1, "Option 1")
        result_layout.addWidget(self.result_tabs)

        layout.addWidget(result_frame)
        self.data_entry_tab.setLayout(layout)

    def run_simulation(self):
        try:
            dims = {key: float(field.text()) for key, field in self.dim_inputs.items() if field.text()}
            tols = {key: float(field.text()) for key, field in self.tol_inputs.items() if field.text()}

            if len(dims) != 3 or len(tols) != 3:
                QMessageBox.warning(self, "Input Error", "Please fill in all dimensions and tolerances.")
                return

            if self.listen_checkbox.isChecked():
                listening_point = (1.0, 1.0, 1.0)  # Replace with actual selection
            else:
                listening_point = self.optimize_listening_point(dims)

            results = self.run_acoustic_simulation(dims, tols, listening_point)
            self.display_results(results)

        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid numbers for all inputs.")

    def optimize_listening_point(self, dims):
        # Placeholder optimization
        return (dims['Lx'] / 2, dims['Ly'] / 2, dims['Lz'] / 2)

    def run_acoustic_simulation(self, dims, tols, listening_point):
        # Placeholder function to simulate
        print(f"Running simulation with dims: {dims}, tolerances: {tols}, point: {listening_point}")
        return {
            "frequencies": [20, 100, 500, 1000, 5000],
            "levels": [70, 75, 72, 68, 65]
        }

    def display_results(self, results):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(results['frequencies'], results['levels'])
        ax.set_title("Frequency Response")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Level (dB)")
        self.canvas.draw()

        self.plotter.clear()
        mesh = pv.Cube()  # Placeholder
        self.plotter.add_mesh(mesh)
        self.plotter.reset_camera()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BROAcousticsApp()
    window.show()
    sys.exit(app.exec_())
