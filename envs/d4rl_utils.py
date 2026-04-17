import d4rl
import gymnasium
import gym
import numpy as np
import collections
from collections.abc import Mapping

from envs.env_utils import EpisodeMonitor
from utils.datasets import Dataset


def make_env(env_name):
    """Make D4RL environment."""
    env = gymnasium.make('GymV21Environment-v0', env_id=env_name)
    env = EpisodeMonitor(env)
    return env


def return_range(dataset):
    returns, lengths = [], []
    ep_ret, ep_len = 0., 0
    for r, d in zip(dataset['rewards'], dataset['terminals']):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == 1000:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0., 0
    # returns.append(ep_ret)    # incomplete trajectory
    lengths.append(ep_len)      # but still keep track of number of steps
    assert sum(lengths) == len(dataset['rewards'])
    return min(returns), max(returns)


# used to calculate the MC return for sparse-reward tasks.
# Assumes that the environment issues two reward values: reward_pos when the
# task is completed, and reward_neg at all the other steps.
ENV_REWARD_INFO = {
    "antmaze": {  # antmaze default is 0/1 reward
        "reward_pos": 1.0,
        "reward_neg": 0.0,
    },
    "adroit-binary": {  # adroit default is -1/0 reward
        "reward_pos": 0.0,
        "reward_neg": -1.0,
    },
}


def _determine_whether_sparse_reward(env_name):
    # return True if the environment is sparse-reward
    # determine if the env is sparse-reward or not
    if "antmaze" in env_name or env_name in [
        "pen-binary-v0",
        "door-binary-v0",
        "relocate-binary-v0",
        "pen-binary",
        "door-binary",
        "relocate-binary",
    ]:
        is_sparse_reward = True
    elif (
        "halfcheetah" in env_name
        or "hopper" in env_name
        or "walker" in env_name
        or "kitchen" in env_name
    ):
        is_sparse_reward = False
    else:
        raise NotImplementedError

    return is_sparse_reward

def _get_negative_reward(env_name, reward_scale, reward_bias):
    """
    Given an environment with sparse rewards (aka there's only two reward values,
    the goal reward when the task is done, or the step penalty otherwise).
    Args:
        env_name: the name of the environment
        reward_scale: the reward scale
        reward_bias: the reward bias. The reward_scale and reward_bias are not applied
            here to scale the reward, but to determine the correct negative reward value.

    NOTE: this function should only be called on sparse-reward environments
    """
    if "antmaze" in env_name:
        reward_neg = (
            ENV_REWARD_INFO["antmaze"]["reward_neg"] * reward_scale + reward_bias
        )
    elif env_name in [
        "pen-binary-v0",
        "door-binary-v0",
        "relocate-binary-v0",
    ]:
        reward_neg = (
            ENV_REWARD_INFO["adroit-binary"]["reward_neg"] * reward_scale + reward_bias
        )
    else:
        raise NotImplementedError(
            """
            If you want to try on a sparse reward env,
            please add the reward_neg value in the ENV_REWARD_INFO dict.
        """
        )

    return reward_neg


def concatenate_batches(batches):
    concatenated = {}
    for key in batches[0].keys():
        if isinstance(batches[0][key], Mapping):
            # to concatenate batch["observations"]["image"], etc.
            concatenated[key] = concatenate_batches([batch[key] for batch in batches])
        else:
            concatenated[key] = np.concatenate(
                [batch[key] for batch in batches], axis=0
            ).astype(np.float32)
    return concatenated


def calc_return_to_go(
    env_name,
    rewards,
    masks,
    gamma,
    reward_scale=None,
    reward_bias=None,
    infinite_horizon=False,
):
    """
    Calculat the Monte Carlo return to go given a list of reward for a single trajectory.
    Args:
        env_name: the name of the environment
        rewards: a list of rewards
        masks: a list of done masks
        gamma: the discount factor used to discount rewards
        reward_scale, reward_bias: the reward scale and bias used to determine
            the negative reward value for sparse-reward environments. If None,
            default from FLAGS values. Leave None unless for special cases.
        infinite_horizon: whether the MDP has inifite horizion (and therefore infinite return to go)
    """
    if len(rewards) == 0:
        return np.array([])

    # process sparse-reward envs
    if reward_scale is None or reward_bias is None:
        # scale and bias not applied, but used to determien the negative reward value
        assert reward_scale is None and reward_bias is None  # both should be unset

    is_sparse_reward = _determine_whether_sparse_reward(env_name)
    if is_sparse_reward:
        reward_neg = _get_negative_reward(env_name, reward_scale, reward_bias)

    if is_sparse_reward and np.all(np.array(rewards) == reward_neg):
        """
        If the env has sparse reward and the trajectory is all negative rewards,
        we use r / (1-gamma) as return to go.
        For exapmle, if gamma = 0.99 and the rewards = [-1, -1, -1],
        then return_to_go = [-100, -100, -100]
        """
        return_to_go = [float(reward_neg / (1 - gamma))] * len(rewards)
    else:
        # sum up the rewards backwards as the return to go
        return_to_go = [0] * len(rewards)
        prev_return = 0 if not infinite_horizon else float(rewards[-1] / (1 - gamma))
        for i in range(len(rewards)):
            return_to_go[-i - 1] = rewards[-i - 1] + gamma * prev_return * (
                masks[-i - 1]
            )
            prev_return = return_to_go[-i - 1]
    return np.array(return_to_go, dtype=np.float32)


def get_dataset(
    env,
    env_name,
    normalize_r,
):
    """Make D4RL dataset.

    Args:
        env: Environment instance.
        env_name: Name of the environment.
    """
    dataset = d4rl.qlearning_dataset(env)

    reward_scale = 1.0
    terminals = np.zeros_like(dataset['rewards'])  # Indicate the end of an episode.
    masks = np.zeros_like(dataset['rewards'])  # Indicate whether we should bootstrap from the next state.
    rewards = dataset['rewards'].copy().astype(np.float32)
    if 'antmaze' in env_name:
        for i in range(len(terminals) - 1):
            terminals[i] = float(
                np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6
            )
            masks[i] = 1 - dataset['terminals'][i]
        rewards = rewards - 1.0
    else:
        for i in range(len(terminals) - 1):
            if (
                np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6
                or dataset['terminals'][i] == 1.0
            ):
                terminals[i] = 1
            else:
                terminals[i] = 0
            masks[i] = 1 - dataset['terminals'][i]
        if normalize_r:
            min_ret, max_ret = return_range(dataset)
            reward_scale = 1000.0 / (max_ret - min_ret)
            print(f'Dataset returns have range [{min_ret}, {max_ret}]')
            rewards *= reward_scale
    masks[-1] = 1 - dataset['terminals'][-1]
    terminals[-1] = 1

    return Dataset.create(
        observations=dataset['observations'].astype(np.float32),
        actions=dataset['actions'].astype(np.float32),
        next_observations=dataset['next_observations'].astype(np.float32),
        terminals=terminals.astype(np.float32),
        rewards=rewards,
        masks=masks,
    ), reward_scale


def get_dataset_with_mc_calculation(
    env,
    env_name,
    normalize_r,
    reward_scale=1.0,
    reward_bias=0.0,
    gamma=0.99,
):
    # this funtion follows d4rl.qlearning_dataset
    # and adds the logic to calculate the return to go
    dataset = d4rl.qlearning_dataset(env)
    
    reward_scale = 1.0
    terminals = np.zeros_like(dataset['rewards'])  # Indicate the end of an episode.
    masks = np.zeros_like(dataset['rewards'])  # Indicate whether we should bootstrap from the next state.
    rewards = dataset['rewards'].copy().astype(np.float32)
    if 'antmaze' in env_name:
        for i in range(len(terminals) - 1):
            terminals[i] = float(
                np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6
            )
            masks[i] = 1 - dataset['terminals'][i]
        rewards = rewards - 1.0
    else:
        for i in range(len(terminals) - 1):
            if (
                np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]) > 1e-6
                or dataset['terminals'][i] == 1.0
            ):
                terminals[i] = 1
            else:
                terminals[i] = 0
            masks[i] = 1 - dataset['terminals'][i]
        if normalize_r:
            min_ret, max_ret = return_range(dataset)
            reward_scale = 1000.0 / (max_ret - min_ret)
            print(f'Dataset returns have range [{min_ret}, {max_ret}]')
            rewards *= reward_scale
    masks[-1] = 1 - dataset['terminals'][-1]
    terminals[-1] = 1
    
    dataset["terminals"] = terminals
    dataset["rewards"] = rewards

    N = dataset["rewards"].shape[0]
    data_ = collections.defaultdict(list)
    episodes_dict_list = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    # use_timeouts = False
    # if "timeouts" in dataset:
    #     use_timeouts = True

    # manually update the terminals for kitchen envs
    # if "kitchen" in env_name:
    #     # kitchen envs don't set the done signal correctly
    #     dataset["terminals"] = np.logical_or(
    #         dataset["terminals"], dataset["rewards"] == 4
    #     )

    # iterate over transitions, put them into trajectories
    episode_step = 0
    for i in range(N):

        done_bool = bool(dataset["terminals"][i])

        # if use_timeouts:
        #     is_final_timestep = dataset["timeouts"][i]
        # else:
        #     is_final_timestep = episode_step == env._max_episode_steps - 1

        # if (not terminate_on_end) and is_final_timestep or i == N - 1:
        if i == N - 1:
            # Skip this transition and don't apply terminals on the last step of an episode
            pass
        else:
            for k in dataset:
                if k in (
                    "actions",
                    "next_observations",
                    "observations",
                    "rewards",
                    "terminals",
                    "timeouts",
                ):
                    data_[k].append(dataset[k][i])
            if "next_observations" not in dataset.keys():
                data_["next_observations"].append(dataset["observations"][i + 1])
            episode_step += 1
        # if (done_bool or is_final_timestep) and episode_step > 0:
        if done_bool and episode_step > 0:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])

            episode_data["rewards"] = (
                episode_data["rewards"] * reward_scale + reward_bias
            )
            episode_data["mc_returns"] = calc_return_to_go(
                env_name,
                episode_data["rewards"],
                1 - episode_data["terminals"],
                gamma,
                reward_scale,
                reward_bias,
            )
            episodes_dict_list.append(episode_data)
            data_ = collections.defaultdict(list)
    dataset = concatenate_batches(episodes_dict_list)

    return Dataset.create(
        observations=dataset["observations"],
        actions=dataset["actions"],
        next_observations=dataset["next_observations"],
        terminals=dataset["terminals"].astype(np.float32),
        rewards=dataset["rewards"],
        masks=1 - dataset["terminals"].astype(np.float32),
        mc_returns=dataset["mc_returns"],
    )