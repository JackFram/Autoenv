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
sh ./Miniconda3-latest-Linux-x86_64.sh # Linux
sh ./Miniconda3-latest-MacOSX-x86_64.sh # Mac
# remove sh files
rm Miniconda3-latest-Linux-x86_64.sh # Linux
rm Miniconda3-latest-MacOSX-x86_64.sh  # Mac

source ~/.bashrc

# install rllab
git clone https://github.com/rll/rllab.git
cd rllab
# this takes a while
conda env create -f environment.yml
conda env update
# If hdf5 is not installed, install it as it is required by AutoEnvs later in the process
conda install hdf5
conda install tensorflow
# activate the rllab environment
conda activate rllab3
python setup.py develop
cd ..
```

### installation instructions for the imitation learning algorithm
```bash
cd ~
git clone https://github.com/sisl/hgail.git
conda activate rllab3
cd hgail
python setup.py develop
# if you have errors indicating you should upgrade your numpy
conda upgrade numpy


```
### installation instructions for AutoEnv
```bash
cd ~
git clone https://github.com/JackFram/Autoenv.git
cd ~/Autoenv
mkdir data
mkdir data/trajectories
mkdir data/experiments

```


### Downloading required data file

you can get the data from [here](https://drive.google.com/file/d/1nAj563dl4bETWfDqPZqTwriYqQ7BkLWR/view?usp=sharing).
After downloading the file, unzip it and save all the files in data directory.

```bash
cd ~/Autoenv/
cp -r /path/to/data ./
```


## Run test
```bash
conda activate rllab3
cd ~/Autoenv/
pip install -r requirements.txt
python adaption.py --n_proc 1 --params_filename itr_200.npz --use_multiagent False --n_envs 1 --adapt_steps 1

```





