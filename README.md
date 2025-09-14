# # pfc-dynamics-python

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Project Structure

```
pfc-dynamics-python/
├── pfc_dynamics/              # Main Python package
│   ├── simulation.py          # Data simulation functions
│   ├── estimation.py          # Parameter estimation algorithms
│   ├── utils.py               # Utility functions (vec, kronmult, etc.)
│   ├── tools_kron.py          # Kronecker product tools
│   └── minfunc.py             # Optimization algorithms (minFunc)
├── demo_learning.py           # Learning demo with plots
├── mtdr_demo.py              # mTDR analysis demo with plots
├── test_mmx_times.py         # Performance timing tests
├── matlab_code/              # Original MATLAB code
│   ├── functionFiles/         # MATLAB function files
│   └── minFunc_2012/         # MATLAB optimization library
├── EstimatedPars/            # Pre-computed results (.mat/.npz)
├── test_outputs/             # Generated plots and test outputs
│   ├── *.png                 # Demo plots 
│   ├── *.txt                 # Test output logs
│   └── plots_summary.md      # Documentation
├── requirements.txt          # Python dependencies
└── setup.py                  # Package installation
```

## Usage

```python
# Run demos
python demo_learning.py
python mtdr_demo.py       
python test_mmx_times.py  
```

Note:
All plots are automatically saved to `test_outputs/` directory with old images cleaned before each run.