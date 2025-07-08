# Shepp-Logan like phantom dataset + AFDPS

This repository contains the code for generating a Shepp-Logan like phantom dataset and training an AFDPS model on it ([https://arxiv.org/pdf/2506.03979](https://arxiv.org/pdf/2506.03979)).

This is a `pdm` project. 
To run it, it's very easy to use `pdm` to install the dependencies and run the code. 
You can of course try to run
```
python -m venv ./venv
source ./venv/bin/activate
pip install -e .
```
although I haven't tried this.

To install `pdm`, you can follow the instructions on [https://pdm-project.org/en/latest/](https://pdm-project.org/en/latest/). 
Once installed, run 
```bash
pdm install
```
to install the dependencies.

To run the code, you can use 
```bash
pdm run python scripts/generate_dataset.py
```
to generate the dataset, and 
```bash
pdm run python scripts/train_model.py
```
to train the model.

The code is structured as follows:

- `src/core`: contains the core code for the AFDPS model
- `src/forward_operator`: contains the forward operator code
- `src/training`: contains the training code
- `src/scripts`: contains the scripts for generating the dataset and training the model

I'll add a notebook soon that should be runnable with
```bash
pdm run jupyter notebook
```