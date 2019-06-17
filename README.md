# Autoenv

- This environment is transformed from [ngsim_env](https://github.com/sisl/ngsim_env) and written in python.

- This environment is an extensible environment that looks like rllab, we encourage any exploration on our environment, including but not limited to 
data source, feature extraction, action propagation, reward definition.

- Added AGen algorithm

## Installation:
### rllab3:
```bash
# install miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh # Linux
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh # Mac
# answer yes to everything
sh ./Miniconda3-latest-Linux-x86_64.sh
rm Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# install rllab
git clone https://github.com/rll/rllab.git
cd rllab
# this takes a while
conda env create -f environment.yml
conda env update
# If hdf5 is not installed, install it as it is required by AutoEnvs later in the process
conda install hdf5
# activate the rllab environment
source activate rllab3
python setup.py develop
cd ..
```

### installation instructions for the imitation learning algorithm
```bash
cd ~
git clone https://github.com/sisl/hgail.git
source activate rllab3
cd hgail
python setup.py develop
cd tests python runtests.py
cd ~/Autoenv
mkdir data
mkdir data/trajectories
mkdir data/experiments


```

### Downloading required data file

you can get the data from [here](https://drive.google.com/file/d/1CcPVu9sBsBa6SIRecLEXW16h72I1kHZU/view?usp=sharing).
After downloading the file, unzip it and save all the files in data directory.


## Run test
```bash
cd ~/Autoenv/
python adaption.py --n_proc 1 --params_filename itr_200.npz --use_multiagent False --n_envs 1 --adapt_steps 1

```





