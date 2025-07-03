#!/bin/bash
sudo apt update

# Install FeniCS
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt-get update
sudo apt-get install fenics

# Dependencies for Mesh Generation
sudo apt install libglu1-mesa -Y

# Dependencias de python (se recomienda instalar en un entorno nuevo)
pip install -r requirements.txt
