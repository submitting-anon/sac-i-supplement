## Setting up a Conda environment to run train and eval scripts

Here are instructions to set up a conda environment with the required dependencies in which to run and train and eval scripts. You will need to have conda installed.

##### Create Conda Env
```
conda create -n sac-i python=3 -y
```

##### Activate Conda Env

```
conda activate sac-i
```

##### Installing Python Packages
Once the conda environment is activated, install the requirements into it with the following commands:

```
conda install pytorch==1.3.1 -y && \
conda install -c conda-forge tensorboard -y && \
conda install -c conda-forge opencv -y && \
conda install -c anaconda numpy -y && \
conda install -c anaconda pandas -y && \
conda install -c conda-forge matplotlib -y && \
conda install -c conda-forge scikit-build -y && \
conda install -c conda-forge gym -y && \
conda install -c conda-forge gym-atari -y && \
conda install -c conda-forge gym-box2d -y

```
