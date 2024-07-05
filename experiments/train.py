"""
This script is called by a wandb sweep to train an agent.
"""

import os
import math
import numpy as np
import torch
import wandb
import concurrent.futures

import imprl.agents  # RL agents
import imprl.structural_envs as structural_envs  # gym environment
from imprl.runners.serial import training_rollout, evaluate_agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set environment variables
os.environ["WANDB_MODE"] = "online"


def parallel_rollout(args):
    config, checkpt_dir, ep = args
    env = structural_envs.make(setting=config["ENV_SETTING"])
    agent_class = imprl.agents.get_agent_class(config["ALGORITHM"])
    agent = agent_class(env, config, device)
    agent.load_weights(checkpt_dir, ep)
    return evaluate_agent(env, agent)


if __name__ == "__main__":

    wandb.init()

    # set seed for modules
    torch.manual_seed(wandb.config.RANDOM_SEED)
    np.random.seed(wandb.config.RANDOM_SEED)

    # logging and checkpointing
    logging_frequency = 100
    checkpt_frequency = 5_000
    inferencing_frequency = 5_000
    num_inference_episodes = 10_000
    best_cost = math.inf
    best_checkpt = 0

    is_time_to_checkpoint = (
        lambda ep: ep % checkpt_frequency == 0 or ep == wandb.config.NUM_EPISODES - 1
    )
    is_time_to_log = (
        lambda ep: ep % logging_frequency == 0 or ep == wandb.config.NUM_EPISODES - 1
    )
    is_time_to_infer = (
        lambda ep: ep % inferencing_frequency == 0
        or ep == wandb.config.NUM_EPISODES - 1
    )

    # path to store model weights
    _dir_name = wandb.run.dir.split("/")[-2]
    checkpt_dir = os.path.join("./experiments/data/", _dir_name, "model_weights")
    os.makedirs(checkpt_dir)
    print("Checkpoint directory: ", checkpt_dir)

    training_log = {}  # log for training metrics

    # Environment
    env_setting = wandb.config.ENV_SETTING
    env = structural_envs.make(setting=env_setting)

    # Agent
    algorithm = wandb.config.ALGORITHM
    agent_class = imprl.agents.get_agent_class(algorithm)
    LearningAgent = agent_class(env, wandb.config, device)

    _baseline = {
        "FailureReplace": env.baselines["FailureReplace"]["mean"],
        "TPI-CBM": env.baselines["TPI-CBM"]["mean"],
    }

    # training loop
    for ep in range(wandb.config.NUM_EPISODES):
        episode_cost = training_rollout(env, LearningAgent)

        LearningAgent.report()

        # CHECKPOINT
        if is_time_to_checkpoint(ep):
            LearningAgent.save_weights(checkpt_dir, ep)

        # INFERENCE
        if is_time_to_infer(ep):

            config = {}
            config.update(wandb.config)

            # parallel evaluation
            args_list = [
                (config, checkpt_dir, ep) for _ in range(num_inference_episodes)
            ]

            with concurrent.futures.ProcessPoolExecutor() as executor:
                # Submit all the tasks and gather the futures
                futures = [
                    executor.submit(parallel_rollout, args) for args in args_list
                ]
                # Wait for all futures to complete and extract the results
                list_func_evaluations = [
                    future.result()
                    for future in concurrent.futures.as_completed(futures)
                ]

            # Combine the results
            eval_costs = np.hstack(list_func_evaluations)

            _mean = np.mean(eval_costs)
            _stderr = np.std(eval_costs) / np.sqrt(len(eval_costs))

            if _mean < best_cost:
                best_cost = _mean
                best_checkpt = ep

            training_log.update(
                {
                    "inference_ep": ep,
                    "inference_mean": _mean,
                    "inference_stderr": _stderr,
                    "best_cost": best_cost,
                    "best_checkpt": best_checkpt,
                }
            )

        # LOGGING
        if is_time_to_log(ep):
            training_log.update(LearningAgent.logger)  # agent logger
            training_log.update(_baseline)  # baseline logger
            wandb.log(training_log, step=ep)  # log to wandb

    wandb.run.summary["best_cost"] = best_cost
    wandb.run.summary["best_checkpt"] = best_checkpt
