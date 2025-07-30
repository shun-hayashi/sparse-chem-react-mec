## Supporting data files for "Sparse identification of chemical reaction mechanisms from limited concentration profiles"
[![DOI](https://img.shields.io/badge/DOI-10.1039%2FD5DD00293A-blue)](https://doi.org/10.1039/D5DD00293A)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15259062.svg)](https://doi.org/10.5281/zenodo.15259062)
<br>cite as:
Shun Hayashi, Digital Discovery, 2025, Accepted Manuscript, https://doi.org/10.1039/D5DD00293A

### Requirements
- python: 3.11.6
- numpy: 1.26.4
- scipy: 1.11.1
- pandas: 2.0.3
- numba: 0.58.1
- numbalsoda: 0.3.4
- pycma: 3.2.2

### Folder Structure
```
.
├── example.ipynb # Jupyter notebook file demonstrating an example of optimizing kinetic models
├── example # calculation results from example.ipynb
├── result # kinetic models developed in this study
├── src
│   ├── chemkinetics.py # main module for optimizing kinetic models
│   ├── load_data.py # load and normalize the experimental data
│   ├── loss.py # loss function
│   ├── lr.py # enumerate possible elementary steps
│   ├── report.py 
│   └── tools.py
└─── time_course # experimental data (absorbance of reaction mixtures with 20 initial conditions)
```
