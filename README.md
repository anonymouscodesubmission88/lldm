This is the official code implementation for the paper: "Long Term Time Series Forecasting With Latent Linear Operators".

All experiments in the paper can be found under lldm/experiments
All graphs were produced using visualizations code which can be found under lldm/experiments

In order to run the code, simply run the following lines from the terminal:
"

git clone https://github.com/anonymouscodesubmission88/lldm.git

cd lldm

conda create --name lldm_venv python=3.10

conda deactivate

conda activate lldm_venv

pip install --upgrade pip setuptools

pip install -e .

"


Now you can either run the scripts under lldm/experiments in an IDE (e.g., PyCharm) or directly through the terminal, for example:
"

cd lldm/experiments/illness/

python ./illness_lldm

"

Note that in order to edit the parameters of each experiment (e.g., prediction horizon, numer of layers, look-back window length etc. you need to directly edit the scripts).

Also, when downloading the data files for the electricity & traffic datasets, notice that the raw file might download as a .txt file. If that happens, simply change the file extension to .csv (the reason for that is that they were uploaded using Git LFS which does not store the actual raw files on the same repository as the code).
