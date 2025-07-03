# Acoustic Modal Response Optimizer

## Descripción
Esta aplicación de código abierto permite optimizar de forma automática la distribución modal en salas de control acústico mediante un enfoque híbrido que combina búsqueda exhaustiva de proporciones geométricas y análisis por Elementos Finitos en frecuencia (FD‑FEM).
---

## Captura de pantalla de la interfaz

![Vista de la GUI](aux/GUI_screen.PNG)

*Figura 1. Interfaz gráfica de la herramienta.*

---

## Requisitos

- **Sistema operativo:** Linux (nativo) o Windows (con WSL).
- **Python:** versión 3.8 o superior.
- **Dependencias:** se instalan automáticamente con el script `install.sh` (incluye FEniCS, PyQt5 y demás librerías necesarias).

---

## Instalación y Ejecución

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/MatiasDiBernardo/Acoustic-Modal-Response-Optimizer.git
   cd Acoustic-Modal-Response-Optimizer
   ```
2. Instalar las dependencias
   ```bash
    chmod +x install.sh
    ./install.sh
   ```
3. Ejecutar la aplicación
   ```bash
    python3 main.py
   ```

Se recomienda tener un entorno nuevo antes de comenzar con el proceso de instalación para evitar conflictos inesperados.