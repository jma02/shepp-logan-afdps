# Shepp-Logan like phantom dataset + AFDPS
Much of the score matching code comes from [https://github.com/yang-song/score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch).   

This repository contains the code for generating a Shepp-Logan like phantom dataset and training an AFDPS model on it ([https://arxiv.org/pdf/2506.03979](https://arxiv.org/pdf/2506.03979)).

This is a `pdm` project. (Or at least I have been experimenting with `pdm`.)
To run it, it's very easy to use `pdm` to install the dependencies and run the code. 
You can of course run
```
python -m venv ./venv
source ./venv/bin/activate
pip install -e .
```
to begin developing and testing the code. 

- (07/09/2025: Looks like the commands above are more stable/reliable than `pdm` anyways.)
- If you want to use Song's `ncsnpp` architecture, you need to make sure your Python installation has
development headers enabled, since Song is fancy, and compiles his own cuda/C++ code.
- I recommend just using `ncsnv2`, since `ncsnpp` is very over engineered for our problem.
You can change this in `training/configs/config.py`.

To install `pdm`, you can follow the instructions on [https://pdm-project.org/en/latest/](https://pdm-project.org/en/latest/). If you don't want to do this, generally remove the prefixed `pdm run` from each command.
Once installed, run 
```bash
pdm install
```
to install the dependencies. (Or go with the venv commands above.)

To generate the data, run
```bash
pdm run python scripts/generate_dataset.py
```
or 
```bash
python scripts/generate_dataset.py
```
```bash
pdm run python training/train_score.py
```
or 
```bash
python training/train_score.py
```
to train the model.

I'll add a notebook soon that should be runnable with
```bash
pdm run jupyter notebook
```
or 
```bash
jupyter notebook
```

# Appendix
- (My side note on `pdm` is it's at the least able to resolve dependencies reliably, but it's actual implementation of virtual environments is a little iffy. See a similar pain point with a competing package manager `poetry`: [here](https://discuss.pytorch.org/t/pytorch-cannot-find-libcudnn/205696). You can always just use `pdm` to resolve dependencies, and then use `python -m venv ./venv` to create a virtual environment. This has worked well for me, and when `pdm` virtual environments work, they work well.¯\_(ツ)_/¯ )

- I think that my personal issues with `pdm` come from our GPU's oneAPI environment. Oh well.