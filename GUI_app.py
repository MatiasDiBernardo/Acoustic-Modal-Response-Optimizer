"""
GUI de Optimización de Respuesta Modal Acústica
===============================================
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
    QHBoxLayout, QFormLayout, QTabWidget, QMessageBox, QGroupBox,
    QComboBox, QTextEdit, QFileDialog, QCheckBox, QFrame, QSizePolicy, QScrollArea, QGridLayout, QStyleFactory
)
from PyQt5.QtGui import QDoubleValidator, QPixmap, QIcon
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plots.graph_room_outline import plot_room_iterative
from outline_optim import find_best_outline
from complex_outline_optim import find_complex_random_optim, calculate_initial
from plots.graph_mag_response import general_mag_response

def get_scalar(val):
    try:
        return float(val.item()) if hasattr(val, 'item') and val.size == 1 else float(val[0])
    except Exception:
        return float(val)


class BROAcousticsGUI(QWidget):
    
    # CREACION DE VENTANA GENERAL: Configura la ventana principal de la aplicación
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optimizador de Respuesta Modal Acústica")
        #self.setGeometry(100, 100, 1400, 800)
        self.resize(1200, 800)
        #self.adjustSize()

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)

        self.contenedor = QWidget()
        self.layout_principal = QVBoxLayout(self.contenedor)

        #self.layout_principal = QVBoxLayout()
        #self.setStyleSheet("background-color: #2b2b2b; color: white; font-family: Arial, sans-serif;")

        # Header layout para el logo y autores
        self.header_layout = QHBoxLayout()
        self.layout_principal.addLayout(self.header_layout)

        banner_layout = QHBoxLayout()
        logo = QLabel()
        pixmap = QPixmap('aux/untref_logo.png')
        pixmap = pixmap.scaled(100,100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        logo.setPixmap(pixmap)
        logo.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        autores = QLabel("""
        <div style='font-size:9pt; line-height:1.1em;'>
        <p><b>Desarrollado por:</p>
        <p>Matías Di Bernardo, Matías Vereertbrugghen, Camila Romina Lucana y María Victoria Alongi</p>
        <p><i>Instrumentos y Mediciones Acústicas - Ingeniería en Sonido - UNTREF @2025</i></p>
        </div>
        """)
        autores.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        #autores.setTextFormat(Qt.RichText)
        #autores.setWordWrap(True)
        autores.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        autores.setStyleSheet("color: light gray;")
        
        banner_layout.addWidget(logo, alignment=Qt.AlignLeft)
        banner_layout.addWidget(autores, stretch=1, alignment=Qt.AlignRight)
        banner_layout.addStretch()

        self.tabs_principales = QTabWidget()
        self.pestana_principal = QWidget()
        self.pestana_instrucciones = QWidget()

        self.layout_principal.addLayout(banner_layout)
        self.tabs_principales.addTab(self.pestana_principal, "Optimización de Sala")
        self.tabs_principales.addTab(self.pestana_instrucciones, "Instrucciones")

        self.layout_principal.addWidget(self.tabs_principales)
        self.layout_principal.addSpacing(5)
        
        self.inicializar_pestana_principal()
        self.inicializar_pestana_instrucciones()
        #self.setLayout(self.layout_principal)

        self. scroll.setWidget(self.contenedor)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.scroll)
        self.setLayout(main_layout)

    #VENTANA DE INSTRUCCIONES: Configura la pestaña de instrucciones con un texto explicativo
    def inicializar_pestana_instrucciones(self):
        layout = QVBoxLayout()
        
        # Logo UNTREF
        # logo = QLabel()
        # pixmap = QPixmap('aux/untref_logo.png')
        # pixmap = pixmap.scaledToWidth(150, Qt.SmoothTransformation)
        # logo.setPixmap(pixmap)
        # logo.setAlignment(Qt.AlignRight)
        # layout.addWidget(logo)

        instrucciones = QLabel("""
        <h3>Instrucciones de uso:</h3>
        <ul>
        <div style='line-height:1.6em;'>
        <li>Ingrese las dimensiones de la sala y sus tolerancias, en metros.</li>
        <li>Las dimensiones y tolerancias deben ser números decimales con hasta 2 dígitos.</li>
        <li>Ingrese las posiciones del receptor y la fuente sonora, en metros.</li>
        <li>Los campos de entrada aceptan números decimales con hasta 5 dígitos.</li>
        <li>Seleccione la velocidad de optimización deseada (Baja, Media, Alta).</>
        <li>Presione "Generar Optimización" para iniciar el proceso.</li>
        <li>Use los checkboxes para mostrar/ocultar curvas en la respuesta modal.</li>
        <li>Para ver el plano de la sala, asegúrese de que las dimensiones y posiciones sean correctas.</li>
        <li>El índice de mérito se mostrará en el plano de la sala.</li>
        <li>Para más información, consulte la documentación del proyecto.</li>
        </ul>
        """)
        instrucciones.setWordWrap(True)
        layout.addWidget(instrucciones)
        layout.addSpacing(100)

        # Créditos
        # creditos = QLabel("""
        # <div style="text-align: center;">
        # <h4>Desarrollado por:</h4>
        # <p>Matías Di Bernardo, Matías Vereertbrugghen, Camila Romina Lucana y María Victoria Alongi</p>
        # <p><i>Instrumentos y Mediciones Acústicas - Ingeniería en Sonido - UNTREF @2025</i></p>
        # </div>
        # """)
        # creditos.setStyleSheet("color: gray; ")
        # creditos.setWordWrap(True)
        # layout.addWidget(creditos)

        layout.addStretch()
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
            entrada_dim.setToolTip(f'Ingrese la dimensión {dim} de la sala en metros. Poner numeros con punto decimal (ej. 2.5). X: ancho, Y: largo, Z: altura')
            entrada_tol.setToolTip(f'Ingrese la tolerancia para {dim} en metros. Poner numeros con punto decimal (ej. 0.1). X: ancho, Y: largo, Z: altura')
            entrada_dim.setValidator(QDoubleValidator(0.01, 100.0, 5))
            entrada_tol.setValidator(QDoubleValidator(0, 3, 2))
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
            campo.setToolTip(f'Ingrese la posición del receptor en el eje {eje.upper()} en metros. Poner numeros con punto decimal (ej. 1.25). X: ancho, Y: largo, Z: altura')
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
            campo.setToolTip(f'Ingrese la posición de la fuente en el eje {eje.upper()} en metros. Poner numeros con punto decimal (ej. 1.25). X: ancho, Y: largo, Z: altura')
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
        self.boton_ejecutar.setToolTip("Inicia la optimización de la sala con los parámetros ingresados.")
        self.boton_ejecutar.clicked.connect(self.ejecutar_optimizacion)
        self.boton_pausa = QPushButton("Pausar")
        self.boton_pausa.setToolTip("Pausa la optimización en curso.")
        self.boton_pausa.setEnabled(False)
        self.boton_exportar = QPushButton("Exportar Resultados")
        self.boton_exportar.setToolTip("Exporta los resultados figura de merito de la optimización a un archivo CSV.")
        self.boton_exportar.clicked.connect(self.exportar_resultados)
        self.boton_borrar = QPushButton("Borrar Resultados")
        self.boton_borrar.setToolTip("Borra todos los resultados y campos de entrada para iniciar una nueva optimización.")
        self.boton_borrar.clicked.connect(self.borrar_todo)
        self.selector_vel = QComboBox()
        self.selector_vel.setToolTip("Seleccione la velocidad de optimización deseada. Baja: aproximadamente 10 minutos, Media: aproximadamente 5 minutos, Alta: aproximadamente 1 minuto.")
        self.selector_vel.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.selector_vel.addItems(["Baja", "Media", "Alta"])
        self.selector_vel.setCurrentIndex(1) # Media por defecto
        self.selector_par = QComboBox()
        self.selector_par.setToolTip("Seleccione la cantidad de paredes/quiebres deseada.")
        self.selector_par.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.selector_par.addItems(["2", "3", "4", '5'])
        self.selector_par.setCurrentIndex(0) # 2 por defecto

        
        control_layout.addWidget(QLabel("Velocidad de Optimización:"))
        control_layout.addWidget(self.selector_vel)
        control_layout.addWidget(QLabel("Cantidad de Paredes/Quiebres:"))
        control_layout.addWidget(self.selector_par)
        control_layout.addWidget(self.boton_ejecutar)
        control_layout.addWidget(self.boton_pausa)
        control_layout.addWidget(self.boton_exportar)
        control_layout.addWidget(self.boton_borrar)
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

        self.fig_magnitud = Figure(figsize=(5, 3))
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
            fig = Figure(figsize=(10, 5))
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
            self.terminal.append("[INFO] Iniciando optimización...")

            # === VALIDACIÓN DE ENTRADAS ===
            campos_requeridos = [
            'Lx', 'Ly', 'Lz', 'tol_Lx', 'tol_Ly', 'tol_Lz',
            'fuente_x', 'fuente_y', 'fuente_z',
            'receptor_x', 'receptor_y', 'receptor_z'
            ]

            valores = {}
            for clave in campos_requeridos:
                texto = self.entradas[clave].text().strip()
                if texto == "":
                    QMessageBox.warning(self, "Campo incompleto", f"El campo '{clave}' está vacío.")
                    self.terminal.append(f"[ERROR] Campo vacío: {clave}")
                    return
                try:
                    valor = float(texto)
                    if valor < 0:
                        raise ValueError("Valor negativo")
                    valores[clave] = valor
                except ValueError:
                    QMessageBox.warning(self, "Valor inválido", f"El campo '{clave}' debe ser un número positivo.")
                    self.terminal.append(f"[ERROR] Valor inválido en: {clave}")
                    return

            # Validaciones adicionales (pueden ajustarse)
            if valores['fuente_z'] > valores['Lz'] or valores['receptor_z'] > valores['Lz']:
                QMessageBox.warning(self, "Altura inválida", "La posición Z de la fuente o el receptor supera la altura de la sala.")
                return

            # Leer entradas del usuario (ya funcionan)
            Lx = valores['Lx']
            Ly = valores['Ly']
            Lz = valores['Lz']
            Dx = valores['tol_Lx']
            Dy = valores['tol_Ly']
            Dz = valores['tol_Lz']

            fuente = (valores['fuente_x'], valores['fuente_y'], valores['fuente_z'])
            receptor = (valores['receptor_x'], valores['receptor_y'], valores['receptor_z'])


            velocidad_ui = self.selector_vel.currentText()  # "Baja", "Media", "Alta"
            mapa_velocidades = {
                "Baja": "Slow",
                "Media": "Medium",
                "Alta": "Fast"
            }
            velocidad = mapa_velocidades.get(velocidad_ui, "Medium")  # Por defecto "Medium"
            cantidad_paredes = int(self.selector_par.currentText())

            self.terminal.append(f"[INFO] Parámetros leídos correctamente.")

            #Medicion de tiempo
            tiempo_inicio = time.time()

            # Simulación de optimización
            self.terminal.append("[INFO] Ejecutando optimización...")

            # Frecuencias
            freqs = np.arange(20, 200, 2) if velocidad == "Alta" else np.arange(20, 200, 1)
            self.frecuencias = freqs

            # Geometría original
            merit0, mag0 = calculate_initial(Lx, Ly, Lz, fuente, receptor)

            # Geometría simple
            best_simple_room, spacing_simple_room, merit1_sm, mag1_sm = find_best_outline(
                Lx, Ly, Lz, Dx, Dy, Dz, fuente, receptor, velocidad
            )

            Lx_new, Ly_new, Lz_new = best_simple_room
            Dx_new, Dy_new = spacing_simple_room
            dx_room = (Lx - Lx_new) / 2
            dy_room = (Ly - Ly_new) / 2

            new_fuente = (fuente[0] - dx_room, fuente[1] - dy_room, fuente[2])
            new_receptor = (receptor[0] - dx_room, receptor[1] - dy_room, receptor[2])

            merit1, mag1 = calculate_initial(Lx_new, Ly_new, Lz_new, new_fuente, new_receptor)

            # Geometría compleja
            best_complex_room, merit2, mag2 = find_complex_random_optim(
                Lx_new, Ly_new, Lz_new, Dx_new, Dy_new, new_fuente, new_receptor, cantidad_paredes, velocidad
            )

            # Guardar curvas y geometrías
            self.curvas = {
                'Original': (freqs, mag0),
                'Simple': (freqs, mag1),
                'Compleja': (freqs, mag2),
            }

            self.geometrias['Original'] = []
            self.geometrias['Simple'] = [Lx_new, Ly_new, Dx_new, Dy_new]
            self.geometrias['Compleja'] = best_complex_room

            self.meritos = {
                'Original': merit0,    #merit0 = (msfd, md, sd)
                'Simple': merit1,
                'Compleja': merit2,
            }

            tiempo_total = time.time() - tiempo_inicio
            self.terminal.append("[INFO] Optimizacion completada en {tiempo_total:.2f} segundos.")
            self.actualizar_curvas_magnitud()
            self.actualizar_plano()
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error en la lectura de entradas:\n{str(e)}")
            self.terminal.append(f"[ERROR] {str(e)}")


    def actualizar_curvas_magnitud(self):
        self.ax_magnitud.clear()
       #self.ax_magnitud.set_title("Respuesta en Frecuencia")
        self.ax_magnitud.set_xlabel("Frecuencia (Hz)")
        self.ax_magnitud.set_ylabel("Magnitud")
        #self.ax_magnitud.set_xscale("log") # Para que la escala sea logaritmica
        self.ax_magnitud.grid(True, which='both')

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
                if key in self.meritos:
                    msfd, md, sv = self.meritos[key]
                    texto = f"<b>Figura de Mérito:</b><br> MSFD: {msfd:.3f} | MD: {md:.3f} | SD: {sv:.3f}"
                else:
                    texto = "Figura de Mérito no disponible."

                self.labels_merito[key].setText(texto)

                # Eliminar viejo legend_canvas si existe
                if self.legend_canvas[key]:
                    self.legend_canvas[key].setParent(None)
                    self.info_layouts[key].removeWidget(self.legend_canvas[key])
                    self.legend_canvas[key].deleteLater()
                    self.legend_canvas[key] = None

                # Crear mini-leyenda visual si hay handles
                if handles:
                    fig_leg = Figure(figsize=(2, 1.5), tight_layout=True)
                    canvas_leg = FigureCanvas(fig_leg)
                    ax_leg = fig_leg.add_subplot(111)
                    ax_leg.axis("off")
                    leg = ax_leg.legend(handles, labels, loc='center', frameon=False)
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
                f.write("Geometría, MSFD, MD, SD\n")
                for clave, merito in self.meritos.items():
                    if isinstance(merito, (tuple, list)) and len(merito) == 3:
                        msfd, md, sd = merito
                    else:
                        msfd = md = sd = float(merito) #fallback si por algun motivo es un solo numero
                                           
                    f.write(f"{clave}, {msfd:.4f}, {md:.4f}, {sd:.4f}\n")
            self.terminal.append(f"[INFO] Resultados exportados a {ruta}")
        except Exception as e:
            self.terminal.append(f"[ERROR] Fallo al exportar: {str(e)}")

    def borrar_todo(self):
        #Limpiar campos de entrada
        for entrada in self.entradas.values():
            entrada.clear()
        
        #Limpiar gráfica magnitud
        self.ax_magnitud.clear()
        self.canvas_magnitud.draw()

        #Limpiar gráficos de planos
        for key in self.axs_plantas:
            self.axs_plantas[key].clear()
            self.canvas_plantas[key].draw()

            # Limpiar etiquetas de mérito
            self.labels_merito[key].setText("")

            # Borrar leyenda visual si existía
            if self.legend_canvas[key]:
                self.legend_canvas[key].setParent(None)
                self.info_layouts[key].removeWidget(self.legend_canvas[key])
                self.legend_canvas[key].deleteLater()
                self.legend_canvas[key] = None

        #Limpiar terminal
        self.terminal.clear()

        #Limpiar datos internos
        # self.curvas = {}
        # self.geometrias = {}
        # self.meritos = {}

        self.terminal.append("[INFO] Todo borrado. Listo para nueva optimización.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ventana = BROAcousticsGUI()
    ventana.show()
    sys.exit(app.exec_())