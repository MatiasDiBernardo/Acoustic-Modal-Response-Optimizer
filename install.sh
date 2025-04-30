#!/bin/bash
sudo apt update

# Install FeniCS
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt-get update
sudo apt-get install fenics

# Dependencies for Mesh Generation
sudo apt install libglu1-mesa -Y

# Dependencias de python (asumimos que ya esta instalado y que es un fresh environment)
pip install -r requirements.txt
