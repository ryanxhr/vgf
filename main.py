import os
import platform

import json
import random
import time

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets, make_env, NormalizeRewardWrapper, make_env_and_datasets_mc
from utils.datasets import Dataset, ReplayBuffer
from utils.evaluation import flatten, evaluate, evaluate_vgf_multiple
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('project_name', 'univr_reproduce', 'Project name.')
flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'ogbench:cube-double-play-singletask-v0', 'Environment (dataset) name, format: <source>:<env_name>..')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 0, 'Number of online steps.')
flags.DEFINE_integer('start_training', 5000, 'Number of training steps to start training.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 10000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 20000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_string('off2on_style', 'iql', 'Different sampling for online fine-tuning: iql (D) or rlpd (half D and half R) or wsrl (R).')
# flags.DEFINE_integer('balanced_sampling', 0, 'Whether to use balanced sampling for online fine-tuning.')
flags.DEFINE_integer('normalize_r', 0, 'Whether to normalize reward.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")

config_flags.DEFINE_config_file('agent', 'agents/fql.py', lock_config=False)


def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project=FLAGS.project_name, group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Make environment and datasets.
    config = FLAGS.agent
    config.max_steps = FLAGS.offline_steps if FLAGS.online_steps == 0 else -1

    env_spec = FLAGS.env_name

    if ':' in env_spec:
        source, env_name = env_spec.split(':', 1)
    else:
        source = 'default'
        env_name = env_spec

    if source == 'd4rl' or source == 'ogbench':
        # offline or offline2online
        if config['agent_name'] == 'mcgf':
            env, eval_env, train_dataset, val_dataset, reward_scale = make_env_and_datasets_mc(env_name, frame_stack=FLAGS.frame_stack, normalize_r=FLAGS.normalize_r)
        else:
            env, eval_env, train_dataset, val_dataset, reward_scale = make_env_and_datasets(env_name, frame_stack=FLAGS.frame_stack, normalize_r=FLAGS.normalize_r)

        if FLAGS.normalize_r:
            env = NormalizeRewardWrapper(env, reward_scale)
            eval_env = NormalizeRewardWrapper(eval_env, reward_scale)
    else:
        # online
        env = make_env(env_name, frame_stack=FLAGS.frame_stack, seed=FLAGS.seed)
        eval_env = make_env(env_name, frame_stack=FLAGS.frame_stack, seed=FLAGS.seed + 42)

    if FLAGS.video_episodes > 0:
        assert 'singletask' in env_name, 'Rendering is currently only supported for OGBench environments.'
    if FLAGS.online_steps > 0:
        assert 'visual' not in env_name, 'Online fine-tuning is currently not supported for visual environments.'

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    env.action_space.seed(FLAGS.seed)
    env.observation_space.seed(FLAGS.seed)
    eval_env.action_space.seed(FLAGS.seed + 42)
    eval_env.observation_space.seed(FLAGS.seed + 42)

    if FLAGS.offline_steps != 0:
        # Set up datasets.
        train_dataset = Dataset.create(**train_dataset)
        if FLAGS.off2on_style == 'rlpd' or FLAGS.off2on_style == 'wsrl':
            # Create a separate empty replay buffer.
            example_transition = {k: v[0] for k, v in train_dataset.items()}
            replay_buffer = ReplayBuffer.create_from_transition(example_transition, size=FLAGS.buffer_size)
        else:
            # Use the training dataset as the replay buffer.
            train_dataset = ReplayBuffer.create_from_initial_dataset(
                dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
            )
            replay_buffer = train_dataset

        # Set p_aug and frame_stack.
        for dataset in [train_dataset, val_dataset, replay_buffer]:
            if dataset is not None:
                dataset.p_aug = FLAGS.p_aug
                dataset.frame_stack = FLAGS.frame_stack
                if config['agent_name'] == 'rebrac':
                    dataset.return_next_actions = True
    else:
        # Create an empty replay buffer.
        replay_buffer = ReplayBuffer.create(obs_space=env.observation_space, act_dim=env.action_space.shape[-1], size=FLAGS.buffer_size)

    # example_batch = train_dataset.sample(1)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    example_obs = np.zeros((1, obs_dim), dtype=np.float32)
    example_action = np.zeros((1, act_dim), dtype=np.float32)

    # Create agent.
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_obs,
        example_action,
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    step = 0
    done = True
    expl_metrics = dict()
    online_rng = jax.random.PRNGKey(FLAGS.seed)
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + FLAGS.online_steps + 1), smoothing=0.1, dynamic_ncols=True):
        if i <= FLAGS.offline_steps:
            # Offline RL.
            batch = train_dataset.sample(config['batch_size'])

            if config['agent_name'] == 'rebrac':
                agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
            else:
                agent, update_info = agent.update(batch)

            if config['agent_name'] == 'mcgf':
                # if i < 5 * 10000:
                #     agent = agent.pretrain_mc_b(batch)
                #     agent, update_info = agent.update(batch)
                #     # agent = agent.hard_update_mc_b(tau=0.005)
                # else:
                #     agent, update_info = agent.update(batch)
                #     if i % 500 == 0:
                #         agent = agent.hard_update_mc_b(tau=1.0)
                agent, update_info = agent.update(batch)
                agent = agent.hard_update_mc_b(tau=0.005)

        elif i == FLAGS.offline_steps+1 and i > 1:
            if config['agent_name'] == 'univr' or config['agent_name'] == 'obac':
                agent = agent.hard_update_actor_bc()
        else:
            # Online fine-tuning.
            online_rng, key = jax.random.split(online_rng)

            if done:
                step = 0
                ob, _ = env.reset()

            if i < FLAGS.start_training:
                action = env.action_space.sample()
            else:
                action = agent.sample_actions(observations=ob, temperature=1, seed=key, deterministic=False)
            action = np.array(action)

            next_ob, reward, terminated, truncated, info = env.step(action.copy())
            done = terminated or truncated

            # TDOO check if this is correct for ogbench offline2online
            if not terminated or 'TimeLimit.truncated' in info:
                mask = 1.0
            else:
                mask = 0.0

            # Adjust reward for D4RL antmaze or reward normalization is True
            if 'antmaze' in env_name and (
                'diverse' in env_name or 'play' in env_name or 'umaze' in env_name
            ):
                reward = reward - 1.0

            replay_buffer.add_transition(
                dict(
                    observations=ob,
                    actions=action,
                    rewards=reward,
                    terminals=float(done),
                    masks=mask,
                    next_observations=next_ob,
                )
            )
            ob = next_ob

            if done:
                expl_metrics = {f'exploration/{k}': np.mean(v) for k, v in flatten(info).items()}

            step += 1

            # Train agent after collecting sufficient data
            if (i+1) >= FLAGS.offline_steps + FLAGS.start_training:
                if FLAGS.off2on_style == 'rlpd':
                    # Half-and-half sampling from the offline dataset and online replay buffer.
                    dataset_batch = train_dataset.sample(config['batch_size'] // 2)
                    replay_batch = replay_buffer.sample(config['batch_size'] // 2)
                    batch = {k: np.concatenate([dataset_batch[k], replay_batch[k]], axis=0) for k in dataset_batch}
                else:
                    # Purely sampling from the online replay buffer.
                    batch = replay_buffer.sample(config['batch_size'])

                if config['agent_name'] == 'rebrac':
                    agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
                elif config['agent_name'] == 'univr' or config['agent_name'] == 'obac':
                    agent, update_info = agent.update(batch, stage='online')
                else:
                    agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            # if val_dataset is not None:
            #     val_batch = val_dataset.sample(config['batch_size'])
            #     _, val_info = agent.total_loss(val_batch, grad_params=None)
            #     train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            renders = []
            eval_metrics = {}

            if config['agent_name'] == 'vgf' or config['agent_name'] == 'avgf':
                step_list = [0, 1, 2, 3]
                all_stats, trajs, cur_renders = evaluate_vgf_multiple(
                    agent=agent,
                    env=eval_env,
                    eval_step_list=step_list,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                )
                renders.extend(cur_renders)

                for j, stats in enumerate(all_stats):
                    eval_vgf_steps = step_list[j]
                    for k, v in stats.items():
                        eval_metrics[f"evaluation_{eval_vgf_steps}/{k}"] = v

                if FLAGS.video_episodes > 0 and len(renders) > 0:
                    video = get_wandb_video(renders=renders)
                    eval_metrics["video"] = video
            else:
                eval_info, trajs, cur_renders = evaluate(
                    agent=agent,
                    env=eval_env,
                    config=config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=FLAGS.video_episodes,
                    video_frame_skip=FLAGS.video_frame_skip,
                )
                renders.extend(cur_renders)
                for k, v in eval_info.items():
                    eval_metrics[f'evaluation/{k}'] = v

                if FLAGS.video_episodes > 0:
                    video = get_wandb_video(renders=renders)
                    eval_metrics['video'] = video

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
