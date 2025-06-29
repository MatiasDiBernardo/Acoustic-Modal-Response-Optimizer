
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout,
    QHBoxLayout, QFormLayout, QTabWidget, QMessageBox, QGroupBox,
    QComboBox, QTextEdit, QFileDialog, QCheckBox, QFrame, QSizePolicy, QScrollArea, QGridLayout, QStyleFactory
)
from PyQt5.QtGui import QDoubleValidator, QPixmap
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plots.graph_room_outline import plot_room_iterative

def get_scalar(val):
    try:
        return float(val.item()) if hasattr(val, 'item') and val.size == 1 else float(val[0])
    except Exception:
        return float(val)


class BROAcousticsGUI(QWidget):
    
    # CREACION DE VENTANA GENERAL: Configura la ventana principal de la aplicación
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optimizador BRO Acústica")
        self.setGeometry(100, 100, 1400, 800)

        self.layout_principal = QVBoxLayout(self)
        self.tabs_principales = QTabWidget()

        self.pestana_principal = QWidget()
        self.pestana_instrucciones = QWidget()

        self.tabs_principales.addTab(self.pestana_principal, "Optimización de Sala")
        self.tabs_principales.addTab(self.pestana_instrucciones, "Instrucciones")

        self.layout_principal.addWidget(self.tabs_principales)

        self.inicializar_pestana_principal()
        self.inicializar_pestana_instrucciones()

    #VENTANA DE INSTRUCCIONES: Configura la pestaña de instrucciones con un texto explicativo
    def inicializar_pestana_instrucciones(self):
        layout = QVBoxLayout()
        instrucciones = QLabel("""
        <h3>Instrucciones de uso:</h3>
        <ul>
        <li>Ingrese las dimensiones de la sala y sus tolerancias.</li>
        <li>Configure la posición del receptor y la fuente sonora.</li>
        <li>Presione "Generar Optimización" para ver los resultados.</li>
        <li>Use los checkboxes para mostrar/ocultar curvas en la respuesta modal.</li>
        <li>Utilice el botón de exportar para guardar los resultados en CSV.</li>
        </ul>
        """)
        instrucciones.setWordWrap(True)
        layout.addWidget(instrucciones)
        self.pestana_instrucciones.setLayout(layout)

    #VENTANA PRINCIPAL: Configura la pestaña principal con campos de entrada, botones y gráficos
    def inicializar_pestana_principal(self):
        layout_general = QHBoxLayout()

        grupo_entrada = QGroupBox("Configuración de Entrada")
        layout_entrada = QVBoxLayout()

        # PASO 1: Dimensiones y Tolerancias de Sala
        grupo_dimensiones = QGroupBox("Paso 1: Dimensiones y Tolerancias de Sala")
        layout_dimensiones = QFormLayout()
        self.entradas = {}
        
        # Crear campos de entrada para dimensiones y tolerancias
        for dim in ['Lx', 'Ly', 'Lz']:
            entrada_dim = QLineEdit()
            entrada_tol = QLineEdit()
            entrada_dim.setValidator(QDoubleValidator(0.01, 100.0, 5))
            entrada_tol.setValidator(QDoubleValidator(1, 1, 1))
            entrada_dim.setPlaceholderText("m")
            entrada_tol.setPlaceholderText("± m")
            self.entradas[dim] = entrada_dim
            self.entradas[f"tol_{dim}"] = entrada_tol

            fila = QHBoxLayout()
            fila.addWidget(entrada_dim)
            fila.addWidget(QLabel("±"))
            fila.addWidget(entrada_tol)
            contenedor = QWidget()
            contenedor.setLayout(fila)
            layout_dimensiones.addRow(dim, contenedor)
        grupo_dimensiones.setLayout(layout_dimensiones)
        layout_entrada.addWidget(grupo_dimensiones)

        #PASO 2: Posición del Receptor
        grupo_receptor = QGroupBox("Paso 2: Posición del Receptor")
        layout_receptor = QFormLayout()
        for eje in ['x', 'y', 'z']:
            campo = QLineEdit()
            campo.setValidator(QDoubleValidator(0.0, 100.0, 2))
            campo.setPlaceholderText("m")
            self.entradas[f"receptor_{eje}"] = campo
            layout_receptor.addRow(f"Receptor {eje.upper()}", campo)
        grupo_receptor.setLayout(layout_receptor)
        layout_entrada.addWidget(grupo_receptor)

        #PASO 3: Posición de la Fuente
        grupo_fuente = QGroupBox("Paso 3: Posición de la Fuente")
        layout_fuente = QFormLayout()
        for eje in ['x', 'y', 'z']:
            campo = QLineEdit()
            campo.setValidator(QDoubleValidator(0.0, 100.0, 2))
            campo.setPlaceholderText("m")
            self.entradas[f"fuente_{eje}"] = campo
            layout_fuente.addRow(f"Fuente {eje.upper()}", campo)
        grupo_fuente.setLayout(layout_fuente)
        layout_entrada.addWidget(grupo_fuente)

        # PASO 4: Botones de Control
        control_group = QGroupBox("Paso 4: Controles")
        control_layout = QVBoxLayout()
        self.boton_ejecutar = QPushButton("Generar Optimizacion")
        self.boton_ejecutar.clicked.connect(self.ejecutar_optimizacion)
        self.boton_pausa = QPushButton("Pausar")
        self.boton_pausa.setEnabled(False)
        self.boton_exportar = QPushButton("Exportar Resultados")
        self.boton_exportar.clicked.connect(self.exportar_resultados)
        self.selector_vel = QComboBox()
        self.selector_vel.addItems(["Baja", "Media", "Alta"])
        self.selector_vel.setCurrentIndex(1) # Media por defecto

        
        control_layout.addWidget(QLabel("Velocidad de Optimización:"))
        control_layout.addWidget(self.selector_vel)
        control_layout.addWidget(self.boton_ejecutar)
        control_layout.addWidget(self.boton_pausa)
        control_layout.addWidget(self.boton_exportar)
        control_layout.addStretch()
        control_group.setLayout(control_layout)
        layout_entrada.addWidget(control_group)

        # AREA DE TERMINAL
        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setFixedHeight(150)
        layout_entrada.addWidget(QLabel("Salida del Programa"))
        layout_entrada.addWidget(self.terminal)

        # TAB ENTRADA DE DATOS
        grupo_entrada.setLayout(layout_entrada)
        layout_general.addWidget(grupo_entrada, 1)

        # TAB RESULTADOS
        grupo_resultados = QGroupBox("Resultados")
        layout_resultados = QVBoxLayout()

        # Subgrupo Gráfico de Magnitud
        grupo_mag = QGroupBox("Gráfica de Magnitud")
        layout_mag = QVBoxLayout()

        checkbox_layout = QHBoxLayout()
        self.checkboxes = {}
        for key in ["Original", "Simple", "Compleja"]:
            checkbox = QCheckBox(key)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.actualizar_curvas_magnitud)
            self.checkboxes[key] = checkbox
            checkbox_layout.addWidget(checkbox)
        layout_mag.addLayout(checkbox_layout)

        self.fig_magnitud = Figure()
        self.canvas_magnitud = FigureCanvas(self.fig_magnitud)
        self.canvas_magnitud.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas_magnitud.updateGeometry()
        self.ax_magnitud = self.fig_magnitud.add_subplot(111)
        layout_mag.addWidget(self.canvas_magnitud)
        grupo_mag.setLayout(layout_mag)
        layout_resultados.addWidget(grupo_mag)

        # Subgrupo Plano de Sala con 3 Tabs
        grupo_plano = QGroupBox("Plano de Sala")
        layout_plano_tabs = QVBoxLayout()

        self.tabs_plano = QTabWidget()

        self.axs_plantas = {}
        self.canvas_plantas = {}
        self.labels_merito = {}

        for nombre in ["Original", "Simple", "Compleja"]:
            tab = QWidget()
            layout_tab = QHBoxLayout()
            
            #Figura del plano
            fig = Figure()
            canvas = FigureCanvas(fig)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            canvas.updateGeometry()
            ax = fig.add_subplot(111)

            layout_tab.addWidget(canvas)

            # Contenedor vertical para mérito + leyenda visual
            info_container = QWidget()
            info_layout = QVBoxLayout(info_container)

            # QLabel para texto de mérito
            info_label = QLabel()
            info_label.setWordWrap(True)
            info_label.setFrameStyle(QFrame.Panel | QFrame.Sunken)
            info_label.setAlignment(Qt.AlignTop)
            info_label.setMinimumWidth(200)
            info_layout.addWidget(info_label)

            # Canvas para leyenda visual (lo agregaremos en tiempo de ejecución)
            legend_canvas = None  # Se definirá en actualizar_plano
            info_layout.addStretch()

            layout_tab.addWidget(info_container)
            tab.setLayout(layout_tab)

            self.labels_merito[nombre] = info_label
            self.legend_canvas = self.legend_canvas if hasattr(self, 'legend_canvas') else {}
            self.legend_canvas[nombre] = legend_canvas
            self.info_layouts = self.info_layouts if hasattr(self, 'info_layouts') else {}
            self.info_layouts[nombre] = info_layout

            self.tabs_plano.addTab(tab, nombre)
            self.axs_plantas[nombre] = ax
            self.canvas_plantas[nombre] = canvas
            

        layout_plano_tabs.addWidget(self.tabs_plano)
        grupo_plano.setLayout(layout_plano_tabs)
        layout_resultados.addWidget(grupo_plano)


        grupo_resultados.setLayout(layout_resultados)
        layout_general.addWidget(grupo_resultados, 4)

        self.pestana_principal.setLayout(layout_general)

        self.frecuencias = np.arange(20, 200, 1)
        self.curvas = {}
        self.geometrias = {}
        self.meritos = {}

    # EJECUCION DE LA OPTIMIZACION: Ejecuta la optimización y actualiza las gráficas
    def ejecutar_optimizacion(self):
        try:
            self.terminal.clear()
            start_time = time.time() #Esto no se si lo hace alguna parte del código, pero lo dejo por si acaso
            self.terminal.append("[INFO] Iniciando optimización...")

            mag0 = np.load("example/mag0.npy")
            mag1 = np.load("example/mag_g1.npy")
            mag4 = np.load("example/mag_g4.npy")[0]

            self.curvas = {
                'Original': (self.frecuencias, mag0),
                'Simple': (self.frecuencias, mag1),
                'Compleja': (self.frecuencias, mag4),
            }

            self.geometrias['Original'] = []
            self.meritos['Original'] = get_scalar(np.load("example/merit_0.npy"))

            self.geometrias['Simple'] = np.load("example/best_dimensiones_g1.npy")[[0,1,3,4]]
            self.meritos['Simple'] = get_scalar(np.load("example/merit_g1.npy"))

            self.geometrias['Compleja'] = np.load("example/rooms_g4.npy")[0]
            self.meritos['Compleja'] = get_scalar(np.load("example/merits_g4.npy")[0])

            self.actualizar_curvas_magnitud()
            self.actualizar_plano()
            self.terminal.append("[INFO] Optimización completada.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error en la optimización:\n{str(e)}")
            self.terminal.append(f"[ERROR] {str(e)}")

    def actualizar_curvas_magnitud(self):
        self.ax_magnitud.clear()
       #self.ax_magnitud.set_title("Respuesta en Frecuencia")
        self.ax_magnitud.set_xlabel("Frecuencia (Hz)")
        self.ax_magnitud.set_ylabel("Magnitud")
        self.ax_magnitud.grid(True)

        # Colores fijos por curva
        colores = {
            "Original": "cyan",
            "Simple": "lime",
            "Compleja": "magenta"
        }

        # Curvas visibles según checkboxes
        for key in ["Original", "Simple", "Compleja"]:
            if self.checkboxes[key].isChecked():
                freqs, mag = self.curvas[key]
                self.ax_magnitud.plot(freqs, mag, label=key, color=colores[key])

        self.ax_magnitud.legend()
        self.fig_magnitud.tight_layout()
        self.canvas_magnitud.draw()

    # ACTUALIZACION DEL PLANO DE LA SALA: Grafica el plano de la sala con las geometrías seleccionadas
    # y muestra el índice de mérito correspondiente
    def actualizar_plano(self):
        try:
            Lx = float(self.entradas['Lx'].text() or 2.5)
            Ly = float(self.entradas['Ly'].text() or 3.0)
            Dx = float(self.entradas['tol_Lx'].text() or 0.4)
            Dy = float(self.entradas['tol_Ly'].text() or 0.6)
            pos_fuente = tuple(float(self.entradas[f'fuente_{a}'].text() or d) for a, d in zip('xyz', [1.9, 1.0, 1.3]))
            pos_receptor = tuple(float(self.entradas[f'receptor_{a}'].text() or d) for a, d in zip('xyz', [1.25, 1.9, 1.2]))

            for key in ["Original", "Simple", "Compleja"]:
                ax = self.axs_plantas[key]
                canvas = self.canvas_plantas[key]
                ax.clear()

                simple = self.geometrias['Simple'] if key != "Original" else []
                complexa = self.geometrias[key] if key == "Compleja" else []

                # Dibuja con leyenda interna
                plot_room_iterative((Lx, Ly, Dx, Dy), pos_fuente, pos_receptor, simple, complexa, ax=ax)

                # Extraer leyenda antes de borrarla
                handles, labels = ax.get_legend_handles_labels()
                legend = ax.get_legend()
                if legend:
                    legend.remove()  # Ocultar la leyenda del gráfico

                canvas.draw()

                # Texto del índice de mérito
                texto = f"<b>Índice de Mérito:</b><br>{self.meritos.get(key, 0):.3f}"
                self.labels_merito[key].setText(texto)

                # Eliminar viejo legend_canvas si existe
                if self.legend_canvas[key]:
                    self.legend_canvas[key].setParent(None)
                    self.info_layouts[key].removeWidget(self.legend_canvas[key])
                    self.legend_canvas[key].deleteLater()
                    self.legend_canvas[key] = None

                # Crear mini-leyenda visual si hay handles
                if handles:
                    fig_leg = Figure()
                    ax_leg = fig_leg.add_subplot(111)
                    ax_leg.axis("off")

                    # Agregar leyenda con estilos completos
                    leg = ax_leg.legend(handles, labels, loc='center left', frameon=False)

                    # Ajustar automáticamente el tamaño de la figura a la leyenda
                    fig_leg.tight_layout(pad=0.5)

                    # Crear canvas y actualizar
                    canvas_leg = FigureCanvas(fig_leg)
                    canvas_leg.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
                    canvas_leg.draw()

                    self.legend_canvas[key] = canvas_leg
                    self.info_layouts[key].insertWidget(1, canvas_leg)
    

                self.terminal.append(f"[INFO] Plano {key} graficado. Índice de mérito: {self.meritos.get(key, 0):.3f}")

        except Exception as e:
            self.terminal.append(f"[ERROR] No se pudo actualizar el plano: {str(e)}")

    def exportar_resultados(self):
        try:
            ruta, _ = QFileDialog.getSaveFileName(self, "Guardar Resultados", "resultados.csv", "CSV (*.csv)")
            if not ruta:
                return
            with open(ruta, 'w') as f:
                f.write("Geometría, Mérito\n")
                for clave, merito in self.meritos.items():
                    f.write(f"{clave}, {merito:.3f}\n")
            self.terminal.append(f"[INFO] Resultados exportados a {ruta}")
        except Exception as e:
            self.terminal.append(f"[ERROR] Fallo al exportar: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ventana = BROAcousticsGUI()
    ventana.show()
    sys.exit(app.exec_())
