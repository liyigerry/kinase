Exploring pocket-aware inhibitors of kinase by generative deep learning, molecular docking, and molecular dynamics simulations


Data

The original data comes from kinase data and is stored in the ./data/generated. sdf file.  The training data and test data are located in the `DL` folder.


Feature

The processing for generating molecular features, pocket features, and model parameter files is handled in the ./DL/make_pretrain_data.py file.


Models

Deep learning experiments were conducted on datasets composed of pocket-aware features. The training script is located in `./pretraining.py`, and the trained model is saved in the `DL` directory.


Molecular Dynamics (MD) Simulations

The interpretability files are located in the `MD_datasets` folder and include interpretability analyses of the reference and candidate molecule complexes. The `Morgan_fingerprinting.py` file in the `MD` folder performs clustering analysis on the candidate molecules, while `rg_rmsd.py`, `RRR.ipynb`, and others are used for plotting and analyzing MD simulation trajectories.
