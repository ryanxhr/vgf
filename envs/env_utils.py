import collections
import re
import time

import ogbench
import d4rl
import gym
import gymnasium
import numpy as np
from gymnasium.spaces import Box

from utils.datasets import Dataset


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env, filter_regexes=None):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0
        self.filter_regexes = filter_regexes if filter_regexes is not None else []

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Remove keys that are not needed for logging.
        for filter_regex in self.filter_regexes:
            for key in list(info.keys()):
                if re.match(filter_regex, key) is not None:
                    del info[key]

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['final_reward'] = reward
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self.unwrapped, 'get_normalized_score'):
                info['episode']['normalized_return'] = (
                    self.unwrapped.get_normalized_score(info['episode']['return']) * 100.0
                )

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self.get_observation(), reward, terminated, truncated, info


class NormalizeRewardWrapper(gymnasium.RewardWrapper):
    def __init__(self, env, scale):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale


class GymnasiumAPIWrapper(gymnasium.Wrapper):
    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            return out
        return out, {}
    
    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            return out
        obs, reward, done, info = out
        terminated, truncated = done, False
        return obs, reward, terminated, truncated, info


def make_env_and_datasets(env_name, frame_stack=None, action_clip_eps=1e-5, normalize_r=True):
    """Make offline RL environment and datasets.

    Args:
        env_name: Name of the environment or dataset.
        frame_stack: Number of frames to stack.
        action_clip_eps: Epsilon for action clipping.
        normalize_r: normalize reward to 0 to 1.
        normalize_s: normalize state for better weighted BC.
    Returns:
        A tuple of the environment, evaluation environment, training dataset, and validation dataset.
    """

    gym_all_envs = gym.envs.registry.all()
    d4rl_env_ids = [env_spec.id for env_spec in gym_all_envs]
    
    reward_scale = 1.0
    if env_name in d4rl_env_ids:
        # D4RL.
        from envs import d4rl_utils

        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset, reward_scale = d4rl_utils.get_dataset(env, env_name, normalize_r)
        train_dataset, val_dataset = dataset, None
    elif 'singletask' in env_name:
        # OGBench.
        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name)
        eval_env = ogbench.make_env_and_datasets(env_name, env_only=True)
        env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        eval_env = EpisodeMonitor(eval_env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        train_dataset = Dataset.create(**train_dataset)
        val_dataset = Dataset.create(**val_dataset)
    else:
        raise ValueError(f'Unsupported environment: {env_name}')

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)
        eval_env = FrameStackWrapper(eval_env, frame_stack)

    env.reset()
    eval_env.reset()

    # Clip dataset actions.
    if action_clip_eps is not None:
        train_dataset = train_dataset.copy(
            add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
        )
        if val_dataset is not None:
            val_dataset = val_dataset.copy(
                add_or_replace=dict(actions=np.clip(val_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
            )

    return env, eval_env, train_dataset, val_dataset, reward_scale


def make_env_and_datasets_mc(env_name, frame_stack=None, action_clip_eps=1e-5, normalize_r=True):
    """Make offline RL environment and datasets.

    Args:
        env_name: Name of the environment or dataset.
        frame_stack: Number of frames to stack.
        action_clip_eps: Epsilon for action clipping.
        normalize_r: normalize reward to 0 to 1.
        normalize_s: normalize state for better weighted BC.
    Returns:
        A tuple of the environment, evaluation environment, training dataset, and validation dataset.
    """

    gym_all_envs = gym.envs.registry.all()
    d4rl_env_ids = [env_spec.id for env_spec in gym_all_envs]
    
    reward_scale = 1.0
    if env_name in d4rl_env_ids:
        # D4RL.
        from envs import d4rl_utils

        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset_with_mc_calculation(env, env_name, normalize_r)
        train_dataset, val_dataset = dataset, None
    else:
        raise ValueError(f'Unsupported environment: {env_name}')

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)
        eval_env = FrameStackWrapper(eval_env, frame_stack)

    env.reset()
    eval_env.reset()

    # Clip dataset actions.
    if action_clip_eps is not None:
        train_dataset = train_dataset.copy(
            add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
        )
        if val_dataset is not None:
            val_dataset = val_dataset.copy(
                add_or_replace=dict(actions=np.clip(val_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
            )

    return env, eval_env, train_dataset, val_dataset, reward_scale


def make_env(env_name, frame_stack=None, add_episode_monitor=True, flatten=True, seed=None):
    """Make online RL environment and datasets.

    Args:
        env_name: Name of the environment or dataset.
        frame_stack: Number of frames to stack.
        action_clip_eps: Epsilon for action clipping.
        normalize_r: normalize reward to 0 to 1.
        normalize_s: normalize state for better weighted BC.
    Returns:
        A tuple of the environment, evaluation environment, training dataset, and validation dataset.
    """

    all_envs = gym.envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]

    if env_name in env_ids:
        # gym
        from gymnasium.wrappers import RescaleAction
        from gymnasium.wrappers import FlattenObservation

        env = gymnasium.make('GymV21Environment-v0', env_id=env_name)
        if flatten and isinstance(env.observation_space, gym.spaces.Dict):
            env = FlattenObservation(env)
        env = RescaleAction(env, -1.0, 1.0)
    else:
        # DMControl.
        from envs import dmc_utils
        from gym.wrappers import RescaleAction
        from gym.wrappers import FlattenObservation

        domain_name, task_name = env_name.split('-')
        env = dmc_utils.DMCEnv(domain_name=domain_name, task_name=task_name, task_kwargs={'random': seed})
        if flatten and isinstance(env.observation_space, gym.spaces.Dict):
            env = FlattenObservation(env)
        env = RescaleAction(env, -1.0, 1.0)

    env = GymnasiumAPIWrapper(env)

    if add_episode_monitor:
        env = EpisodeMonitor(env)

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)

    env.reset()
    return env
