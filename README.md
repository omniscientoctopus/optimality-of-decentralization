# Optimality of Decentralization

A repository for the code accompanying the paper "[Assessing the Optimality of Decentralized Inspection and Maintenance Policies for Stochastically Degrading Engineering Systems](https://pure.tudelft.nl/ws/portalfiles/portal/214136107/BNAIC2023_paper_13.pdf)" by Prateek Bhustali and Charalampos Andriotis.

This repository imports the [imprl](https://github.com/omniscientoctopus/imprl) library as a submodule at a specific commit to ensure reproducibility as the imprl library is being developed further for general use.

## Installation

### 1. Clone the repository (with submodules)

```bash
git clone --recurse-submodules -j8 https://github.com/omniscientoctopus/optimality-of-decentralization.git
```

### 2. Create a virtual environment

```bash
conda create --name opdec_env -y python==3.9
conda activate opdec_env
```

### 3. Install the dependencies

```bash
pip install poetry==1.8 # or conda install -c conda-forge poetry==1.8
poetry install
```

(optional) Install with dev dependencies to test the installation by
running the following command:

```bash
poetry install --with dev
pytest -v
```

(hopefully, all tests pass!)

### 3. Setup WandB

We use [wandb](https://wandb.ai/) to log the results. To use wandb, you need to create an account and set up the API key.

```bash
wandb login <API_KEY>
```

## Model checkpoints and inference data

We benchmark against $5$ different settings of the k-out-of-5 environment with $7$ algorithms, and $10$ seeds each. The checkpoints and inference data can be downloaded via the command line as follows:

```bash
cd results
wget https://surfdrive.surf.nl/files/index.php/s/eazbsUSRdCgRKxN/download
unzip download
```

The size of the compressed model checkpoints is ~3.1GB, and unzipped ~4.7GB (unzipping takes ~3 minutes).

The model checkpoints and inference data is stored in the following hierarchy:

```bash
results
└──model_checkpoints
    └── k_out_of_n
        └── hard-1-of-5
            └── DCMAC / runs
                └── 1gk287cb
                    ├── model_weights
                    ├── inference_log.json
                ├── 6q4ycugk
                ├── 6zachyyh
                ├── 9g474n58
                ├── 91pmfl02
                ├── avf174d6
                ├── p33ax53o
                ├── pv5rxm8b
                ├── rhe6sshi
                ├── xv5mvp17
            ├── DDMAC
            ├── IAC
            ├── IAC_PS
            └── IACC
        └── hard-2-of-5
            ...
        └── hard-3-of-5
            ...
        └── hard-4-of-5
            ...
        └── hard-5-of-5
```

where `inference_log.json` contains the returns computed using the model checkpoints. Given the large variance of the expected value, we use 10,000 MC rollouts.


## Reproducing experiments

To reproduce the results in the paper, we provide the required scripts in the experiments directory.

Example: Run DCMAC on the 1-out-of-5 environment with 5 seeds. 

We first create a configuration file and then initialize a wandb sweep. The `reproduce_DCMAC.yaml` file contains the hyperparameters used in the paper. The `project-name` is the name of the project in wandb where the sweep will be logged.

```bash
python experiments/create_config.py "hard-1-of-5" "DCMAC" 5
wandb sweep --project "<project-name>" experiments/configs/reproduce_DCMAC.yaml
```

To run the sweep, use the following command:

```bash
wandb agent -e "<entity>" -p "<project-name>" --count 1 <sweep-id>
```

This will start 1 run of the sweep using the `train.py` script on the entity `<entity>` and project `<project-name>`.
