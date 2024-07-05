import sys
import ruamel.yaml

yaml = ruamel.yaml.YAML()
yaml.indent(sequence=4, offset=2)

ENV_SETTING = sys.argv[1]
ALGORITHM = sys.argv[2]
NUM_SEEDS = sys.argv[3]

# YAML file location
location = "./experiments/configs"
file = f"reproduce_{ALGORITHM}.yaml"

# read yaml file
with open(f"{location}/{file}") as fp:
    data = yaml.load(fp)

# update yaml file
data["name"] = f"k_out_of_n-{ENV_SETTING}-{ALGORITHM}"  # sweep name
data["parameters"]["ENV_SETTING"]["value"] = ENV_SETTING
data["parameters"]["ALGORITHM"]["value"] = ALGORITHM
data["parameters"]["RANDOM_SEED"]["values"] = list(range(1, int(NUM_SEEDS) + 1))

# save yaml file with original formatting
with open(f"{location}/{file}", "w") as fp:
    yaml.dump(data, fp)
