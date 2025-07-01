# fpl-2025-runtime-pred
Data and code for FPL 2025 Paper "Open-Source FPGA Routing Runtime Prediction for Improved Productivity" by Gunter and Wilton.

Directories:

1) raw_results_data -- Raw data for figures in the paper
2) src -- Source code to reproduce raw data (will be printed to command line, unparsed)

Steps to reproduce results:

1) Open project root in a terminal
2) Using conda 24.3.0 run "conda env create -f fpl-2025-runtime-pred.yml"
3) "conda activate fpl-2025-runtime-pred"
4) Unzip data.zip in place
5) Unzip models.zip in place
5) python3 src/fpl2025.py


