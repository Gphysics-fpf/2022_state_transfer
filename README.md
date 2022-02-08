# Instructions 

This is the set of simluations for the article "Universal deterministic quantum operations in microwave quantum links", Guillermo F. Peñas, Ricardo Puebla, Tomás Ramos, Peter Rabl, Juan José García-Ripoll, arXiv:2110.02092. 

It consists of the following files:
- `Codes/setup.py` Script that contains the most important objects and methods ans gets used in all the subsequent notebooks. In particular it contains the routine that creates the Hamiltonian matrix form a given set of parameters. 
- `Codes/00 State transfer simulations.ipynb` State transfer experiment between two nodes of a quantum link. A wide range of parameters is explored and our analitycal formulas are verified. These are the fidelity limits regarding diffraction and time dependent Stark shifts. Also, at the end of this notebook, we perform a numerical study of how the control gets distorted when including finite bandwidths.
- `Codes/01 Transfer Protocols Comparison.ipynb` Comparison of an excitation transfer between three different protocols, pulse shaping (the one the work orbitates around), quasiadiabatic STIRAP-like and direct-SWAP (trnafer of an excitation with constant coupling). First, the fidelity of such excitation transfer is studied, and later a further comparison relating the time robustness and the importance of truncation. 
- `Codes/02 Scattering and gate transfer simulations.ipynb` Scattering of a photon when it bouncess off one of the ends of the waveguide. In this notebook we forget about the emission and obsorption processes and just focuss on obtaining the phase profile of the interaction and compare it to our analytical results.
- `Codes/03 Phase gate simulations.ipynb` Simulation of a complete controlled phase gate. In this notebook we make use of all the concepts studied in the previous ones and combien the dynamic couplings with scattering to design a control phase gate.
- `Codes/99 All article plots.ipynb`Notebook the loads data for the data folder and fabricates with it the plots that can be found in arXiv:2110.02092
- 
- `Data` Folder in which the data of the simulations are stored.

To run the simulations execute the Jupyter notebooks in the order in which they are listed above. In or4der for the simulations to succed you will need:
- Python >= 3.9.
- Numpy ...
- Scipy ...
- Matplotlib ... 

[![DOI](https://zenodo.org/badge/451439295.svg)](https://zenodo.org/badge/latestdoi/451439295)
