"""
Prueba de estructura de GUI con resultados obtenidos de ejemplo.
Esta GUI permite al usuario ingresar dimensiones de una sala, posiciones de fuente y receptor,
y generar una optimización de la respuesta modal acústica.
Los resultados se muestran en gráficos y se pueden exportar a un archivo CSV.
La GUI está construida con PyQt5 y utiliza Matplotlib para los gráficos.
La estructura incluye pestañas para la entrada de datos y un manual de instrucciones.
El usuario puede ingresar dimensiones de la sala, tolerancias, posiciones de fuente y receptor,
y ver los resultados de la optimización en gráficos interactivos.
Esta implementación es un ejemplo básico y puede ser extendida con más funcionalidades según sea necesario.


"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout,
    QHBoxLayout, QFormLayout, QTabWidget, QMessageBox, QCheckBox, QGroupBox,
    QComboBox, QTextEdit, QFileDialog
)
from PyQt5.QtGui import QDoubleValidator, QPixmap
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plots.graph_room_outline import plot_room_iterative
from plots.graph_mag_response import general_mag_response

def get_scalar(val):
    try:
        return float(val.item()) if hasattr(val, 'item') and val.size == 1 else float(val[0])
    except Exception:
        return float(val)

class BROAcousticsGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BRO Acoustics Optimizer")
        self.setGeometry(100, 100, 1400, 800)

        self.layout = QVBoxLayout(self)
        self.tabs = QTabWidget()

        self.main_tab = QWidget()
        self.instruction_tab = QWidget()
        self.tabs.addTab(self.main_tab, "Room Optimization")
        self.tabs.addTab(self.instruction_tab, "Instructions")

        self.layout.addWidget(self.tabs)
        self.init_main_tab()
        self.init_instruction_tab()

    def init_instruction_tab(self):
        layout = QVBoxLayout()
        instructions = QLabel("""
        <h3>Instrucciones de uso:</h3>
        <ul>
        <li>Ingrese las dimensiones de la sala y sus tolerancias en el Paso 1.</li>
        <li>Configure la posición del receptor y la fuente sonora en los pasos siguientes.</li>
        <li>Presione "Generate Optimization" para ver los resultados.</li>
        <li>Cambie entre pestañas para ver la geometría base, simple o compleja.</li>
        <li>Use el botón de exportación para guardar las dimensiones y méritos.</li>
        </ul>
        """)
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        self.instruction_tab.setLayout(layout)

    def init_main_tab(self):
        layout = QHBoxLayout()

        input_group = QGroupBox("Input Configuration")
        input_layout = QVBoxLayout()

        dim_group = QGroupBox("Step 1: Room Dimensions and Tolerances")
        dim_layout = QFormLayout()
        self.inputs = {}
        for dim in ['Lx', 'Ly', 'Lz']:
            dim_input = QLineEdit()
            tol_input = QLineEdit()
            dim_input.setValidator(QDoubleValidator(0.01, 100.0, 2))
            tol_input.setValidator(QDoubleValidator(0.0, 10.0, 2))
            dim_input.setPlaceholderText("m")
            tol_input.setPlaceholderText("± m")
            self.inputs[dim] = dim_input
            self.inputs[f"tol_{dim}"] = tol_input

            row = QHBoxLayout()
            row.addWidget(dim_input)
            row.addWidget(QLabel("±"))
            row.addWidget(tol_input)
            container = QWidget()
            container.setLayout(row)
            dim_layout.addRow(dim, container)
        dim_group.setLayout(dim_layout)
        input_layout.addWidget(dim_group)

        listener_group = QGroupBox("Step 2: Listener Position")
        listener_layout = QFormLayout()
        for axis in ['x', 'y', 'z']:
            field = QLineEdit()
            field.setValidator(QDoubleValidator(0.0, 100.0, 2))
            field.setPlaceholderText("m")
            self.inputs[f"receiver_{axis}"] = field
            listener_layout.addRow(f"Receiver {axis.upper()}", field)
        listener_group.setLayout(listener_layout)
        input_layout.addWidget(listener_group)

        source_group = QGroupBox("Step 3: Source Position")
        source_layout = QFormLayout()
        for axis in ['x', 'y', 'z']:
            field = QLineEdit()
            field.setValidator(QDoubleValidator(0.0, 100.0, 2))
            field.setPlaceholderText("m")
            self.inputs[f"source_{axis}"] = field
            source_layout.addRow(f"Source {axis.upper()}", field)
        source_group.setLayout(source_layout)
        input_layout.addWidget(source_group)

        control_group = QGroupBox("Step 4: Controls")
        control_layout = QVBoxLayout()
        self.run_button = QPushButton("Generate Optimization")
        self.run_button.clicked.connect(self.run_example)
        self.pause_button = QPushButton("Pause")
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        self.speed_selector = QComboBox()
        self.speed_selector.addItems(["Fast", "Normal", "Slow"])

        control_layout.addWidget(self.run_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.export_button)
        control_layout.addWidget(QLabel("Processing Speed"))
        control_layout.addWidget(self.speed_selector)
        control_group.setLayout(control_layout)
        input_layout.addWidget(control_group)

        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setFixedHeight(150)
        input_layout.addWidget(QLabel("Program Output"))
        input_layout.addWidget(self.terminal)

        input_group.setLayout(input_layout)
        layout.addWidget(input_group, 2)

        result_group = QGroupBox("Results")
        result_layout = QVBoxLayout()

        self.result_tabs = QTabWidget()
        self.curves = {}
        self.merits = {}
        self.room_geometries = {}

        for label in ["Base", "Simple", "Compleja"]:
            fig = Figure(figsize=(6, 4))
            canvas = FigureCanvas(fig)
            self.curves[label] = (fig, canvas)

            tab = QWidget()
            tab_layout = QVBoxLayout()
            tab_layout.addWidget(canvas)
            tab.setLayout(tab_layout)

            self.result_tabs.addTab(tab, f"{label} Magnitude")

        self.result_tabs.currentChanged.connect(self.update_room_plan_plot)
        result_layout.addWidget(self.result_tabs)

        room_group = QGroupBox("Room Plan")
        room_layout = QVBoxLayout()
        self.room_image = QLabel()
        self.room_image.setAlignment(Qt.AlignCenter)
        room_layout.addWidget(self.room_image)
        room_group.setLayout(room_layout)
        result_layout.addWidget(room_group)

        result_group.setLayout(result_layout)
        layout.addWidget(result_group, 3)

        self.main_tab.setLayout(layout)

    def run_example(self):
        try:
            self.terminal.clear()
            start_time = time.time()
            self.terminal.append("[INFO] Starting optimization...")

            Lx = float(self.inputs['Lx'].text() or 2.5)
            Ly = float(self.inputs['Ly'].text() or 3.0)
            Dx = float(self.inputs['tol_Lx'].text() or 0.4)
            Dy = float(self.inputs['tol_Ly'].text() or 0.6)
            source_position = tuple(float(self.inputs[f'source_{a}'].text() or d) for a, d in zip('xyz', [1.9, 1.0, 1.3]))
            receiver_position = tuple(float(self.inputs[f'receiver_{a}'].text() or d) for a, d in zip('xyz', [1.25, 1.9, 1.2]))
            freqs = np.arange(20, 200, 1)

            mag0 = np.load("example/mag0.npy")
            merit0 = np.load("example/merit_0.npy")
            self.merits['Base'] = get_scalar(merit0)
            self.plot_curve("Base", freqs, mag0, "Base Room")

            mag1 = np.load("example/mag_g1.npy")
            merit1 = np.load("example/merit_g1.npy")
            best_simple = np.load("example/best_dimensiones_g1.npy")
            self.merits['Simple'] = get_scalar(merit1)
            self.room_geometries['Simple'] = best_simple[[0, 1, 3, 4]]
            self.plot_curve("Simple", freqs, mag1, "Simple Geometry")

            mag4 = np.load("example/mag_g4.npy")[0]
            merit4 = np.load("example/merits_g4.npy")[0]
            best_complex = np.load("example/rooms_g4.npy")[0]
            self.merits['Compleja'] = get_scalar(merit4)
            self.room_geometries['Compleja'] = best_complex
            self.plot_curve("Compleja", freqs, mag4, "Complex Geometry")

            self.room_geometries['Base'] = []
            self.update_room_plan_plot()

            elapsed = time.time() - start_time
            for key in self.merits:
                self.terminal.append(f"[RESULT] {key} merit: {self.merits[key]:.3f}")
            self.terminal.append(f"[DONE] Completed in {elapsed:.2f} seconds.")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Something went wrong:\n{str(e)}")
            self.terminal.append(f"[ERROR] {str(e)}")

    def update_room_plan_plot(self):
        try:
            label = self.result_tabs.tabText(self.result_tabs.currentIndex()).split()[0]
            Lx = float(self.inputs['Lx'].text() or 2.5)
            Ly = float(self.inputs['Ly'].text() or 3.0)
            Dx = float(self.inputs['tol_Lx'].text() or 0.4)
            Dy = float(self.inputs['tol_Ly'].text() or 0.6)
            source_position = tuple(float(self.inputs[f'source_{a}'].text() or d) for a, d in zip('xyz', [1.9, 1.0, 1.3]))
            receiver_position = tuple(float(self.inputs[f'receiver_{a}'].text() or d) for a, d in zip('xyz', [1.25, 1.9, 1.2]))

            simple = self.room_geometries.get('Simple', [])
            complex_ = self.room_geometries.get(label, [])

            plt.figure()
            plot_room_iterative((Lx, Ly, Dx, Dy), source_position, receiver_position, simple, complex_)
            plt.savefig("room_temp.png")
            plt.close()
            self.room_image.setPixmap(QPixmap("room_temp.png").scaled(600, 400, Qt.KeepAspectRatio))
        except Exception as e:
            self.terminal.append(f"[ERROR] Room plot update failed: {str(e)}")

    def plot_curve(self, label, freqs, magnitudes, title):
        fig, canvas = self.curves[label]
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(freqs, magnitudes, label=title)
        ax.set_title(title)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.grid(True)
        ax.legend()
        canvas.draw()

    def export_results(self):
        try:
            path, _ = QFileDialog.getSaveFileName(self, "Save Results", "results.csv", "CSV Files (*.csv)")
            if not path:
                return
            with open(path, 'w') as f:
                f.write("Geometry, Merit\n")
                for label, merit in self.merits.items():
                    f.write(f"{label}, {merit:.3f}\n")
            self.terminal.append(f"[INFO] Results exported to {path}")
        except Exception as e:
            self.terminal.append(f"[ERROR] Export failed: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = BROAcousticsGUI()
    gui.show()
    sys.exit(app.exec_())
