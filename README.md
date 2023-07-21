# charmJ12Scripts
A set of scripts and folders to reproduce the analysis and pltos in the spin 1/2 charm baryon paper

---

# Conda Notes
## Switch to a faster environment solver
This is optional, but likely will solve the dependencies much much faster. See https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community
```conda update -n base conda```
```conda install -n base conda-libmamba-solver```
```conda config --set solver libmamba```

## Install Environment
```conda env create -f environment.yml```
## Activate/Use
```conda activate charm```
## Update (w. new packages)

1. Edit `environment.yml`
2. Deactivate conda environment with `conda deactivate`
3. Update conda environment with `conda env update -f=environment.yml`
