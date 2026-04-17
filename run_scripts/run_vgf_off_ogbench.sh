#!/bin/bash

# Script to reproduce vgf in d4rl results

GPU_LIST=(1 2 0)


for seed in 5; do

for bc_flow_steps in 10; do

for vgf_particles in 10; do

for vgf_lr in 0.01 0.05; do

for vgf_alpha in 0.2 0.5; do

for train_vgf_steps in 3; do

# for train_q_agg in 'mean'; do

for train_particle_select in 'mean'; do

for eval_particle_select in 'max'; do

for task_id in task4; do


# GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
# CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
#   --env_name="ogbench:antmaze-large-navigate-singletask-${task_id}-v0" \
#   --seed=$seed \
#   --offline_steps=2000000 \
#   --eval_interval=100000 \
#   --eval_episodes=50 \
#   --agent=agents/vgf.py \
#   --agent.discount=0.995 \
#   --agent.batch_size=256 \
#   --agent.activations='relu' \
#   --agent.actor_lr=0.0003 \
#   --agent.value_lr=0.0003 \
#   --agent.value_hidden_dims='(512, 512, 512, 512)' \
#   --agent.actor_hidden_dims='(512, 512, 512, 512)' \
#   --agent.layer_norm=True \
#   --agent.critic_loss='td' \
#   --agent.train_q_agg='min' \
#   --agent.eval_q_agg='min' \
#   --agent.bc_flow_steps=$bc_flow_steps \
#   --agent.vgf_particles=$vgf_particles \
#   --agent.train_particle_select=$train_particle_select \
#   --agent.eval_particle_select=$eval_particle_select \
#   --agent.train_vgf_steps=$train_vgf_steps \
#   --agent.vgf_lr=$vgf_lr \
#   --agent.vgf_alpha=$vgf_alpha \
#   --project_name='vgf_paper' \
#   --run_group='no entropy' &

# sleep 2
# let "task=$task+1"


# GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
# CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
#   --env_name="ogbench:antmaze-giant-navigate-singletask-${task_id}-v0" \
#   --seed=$seed \
#   --normalize_r=1 \
#   --offline_steps=3000000 \
#   --eval_interval=100000 \
#   --eval_episodes=50 \
#   --agent=agents/vgf.py \
#   --agent.discount=0.99 \
#   --agent.batch_size=256 \
#   --agent.activations='relu' \
#   --agent.actor_lr=0.0003 \
#   --agent.value_lr=0.0003 \
#   --agent.value_hidden_dims='(512, 512, 512, 512)' \
#   --agent.actor_hidden_dims='(512, 512, 512, 512)' \
#   --agent.layer_norm=True \
#   --agent.critic_loss='td' \
#   --agent.train_q_agg='min' \
#   --agent.vgf_q_agg='mean' \
#   --agent.bc_flow_steps=$bc_flow_steps \
#   --agent.vgf_particles=$vgf_particles \
#   --agent.train_particle_select=$train_particle_select \
#   --agent.eval_particle_select=$eval_particle_select \
#   --agent.train_vgf_steps=$train_vgf_steps \
#   --agent.vgf_lr=$vgf_lr \
#   --agent.vgf_alpha=$vgf_alpha \
#   --project_name='vgf_camera_ready_off' \
#   --run_group='ogbench' &

# sleep 2
# let "task=$task+1"


# GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
# CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
#   --env_name="ogbench:humanoidmaze-medium-navigate-singletask-${task_id}-v0" \
#   --seed=$seed \
#   --offline_steps=1000000 \
#   --eval_interval=100000 \
#   --eval_episodes=50 \
#   --agent=agents/vgf.py \
#   --agent.discount=0.995 \
#   --agent.actor_lr=0.0003 \
#   --agent.value_lr=0.0003 \
#   --agent.value_hidden_dims='(512, 512, 512, 512)' \
#   --agent.actor_hidden_dims='(512, 512, 512, 512)' \
#   --agent.layer_norm=True \
#   --agent.critic_loss='td' \
#   --agent.train_q_agg='mean' \
#   --agent.vgf_q_agg='mean' \
#   --agent.bc_flow_steps=$bc_flow_steps \
#   --agent.vgf_particles=$vgf_particles \
#   --agent.train_particle_select=$train_particle_select \
#   --agent.eval_particle_select=$eval_particle_select \
#   --agent.train_vgf_steps=$train_vgf_steps \
#   --agent.vgf_lr=$vgf_lr \
#   --agent.vgf_alpha=$vgf_alpha \
#   --project_name='vgf_camera_ready_off' \
#   --run_group='ogbench' &

# sleep 2
# let "task=$task+1"


# GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
# CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
#   --env_name="ogbench:humanoidmaze-large-navigate-singletask-${task_id}-v0" \
#   --seed=$seed \
#   --eval_interval=100000 \
#   --eval_episodes=50 \
#   --agent=agents/vgf.py \
#   --agent.discount=0.995 \
#   --agent.actor_lr=0.0003 \
#   --agent.value_lr=0.0003 \
#   --agent.value_hidden_dims='(512, 512, 512, 512)' \
#   --agent.actor_hidden_dims='(512, 512, 512, 512)' \
#   --agent.layer_norm=True \
#   --agent.critic_loss='td' \
#   --agent.train_q_agg='mean' \
#   --agent.vgf_q_agg='mean' \
#   --agent.bc_flow_steps=$bc_flow_steps \
#   --agent.vgf_particles=$vgf_particles \
#   --agent.train_particle_select=$train_particle_select \
#   --agent.eval_particle_select=$eval_particle_select \
#   --agent.train_vgf_steps=$train_vgf_steps \
#   --agent.vgf_lr=$vgf_lr \
#   --agent.vgf_alpha=$vgf_alpha \
#   --project_name='vgf_camera_ready_off' \
#   --run_group='ogbench' &

# sleep 2
# let "task=$task+1"


# GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
# CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
#   --env_name="ogbench:antsoccer-arena-navigate-singletask-${task_id}-v0" \
#   --seed=$seed \
#   --eval_interval=100000 \
#   --eval_episodes=50 \
#   --agent=agents/vgf.py \
#   --agent.discount=0.995 \
#   --agent.actor_lr=0.0003 \
#   --agent.value_lr=0.0003 \
#   --agent.value_hidden_dims='(512, 512, 512, 512)' \
#   --agent.actor_hidden_dims='(512, 512, 512, 512)' \
#   --agent.layer_norm=True \
#   --agent.critic_loss='td' \
#   --agent.train_q_agg='mean' \
#   --agent.vgf_q_agg='mean' \
#   --agent.bc_flow_steps=$bc_flow_steps \
#   --agent.vgf_particles=$vgf_particles \
#   --agent.train_particle_select=$train_particle_select \
#   --agent.eval_particle_select=$eval_particle_select \
#   --agent.train_vgf_steps=$train_vgf_steps \
#   --agent.vgf_lr=$vgf_lr \
#   --agent.vgf_alpha=$vgf_alpha \
#   --project_name='vgf_camera_ready_off' \
#   --run_group='ogbench' &

# sleep 2
# let "task=$task+1"


# GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
# CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
#   --env_name="ogbench:cube-single-play-singletask-${task_id}-v0" \
#   --seed=$seed \
#   --eval_interval=100000 \
#   --eval_episodes=50 \
#   --agent=agents/vgf.py \
#   --agent.discount=0.99 \
#   --agent.actor_lr=0.0003 \
#   --agent.value_lr=0.0003 \
#   --agent.value_hidden_dims='(512, 512, 512, 512)' \
#   --agent.actor_hidden_dims='(512, 512, 512, 512)' \
#   --agent.layer_norm=True \
#   --agent.critic_loss='td' \
#   --agent.train_q_agg='mean' \
#   --agent.vgf_q_agg='mean' \
#   --agent.bc_flow_steps=$bc_flow_steps \
#   --agent.vgf_particles=$vgf_particles \
#   --agent.train_particle_select=$train_particle_select \
#   --agent.eval_particle_select=$eval_particle_select \
#   --agent.train_vgf_steps=$train_vgf_steps \
#   --agent.vgf_lr=$vgf_lr \
#   --agent.vgf_alpha=$vgf_alpha \
#   --project_name='vgf_camera_ready_off' \
#   --run_group='ogbench' &

# sleep 2
# let "task=$task+1"


# GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
# CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
#   --env_name="ogbench:cube-double-play-singletask-${task_id}-v0" \
#   --seed=$seed \
#   --eval_interval=100000 \
#   --eval_episodes=50 \
#   --agent=agents/vgf.py \
#   --agent.discount=0.99 \
#   --agent.actor_lr=0.0003 \
#   --agent.value_lr=0.0003 \
#   --agent.value_hidden_dims='(512, 512, 512, 512)' \
#   --agent.actor_hidden_dims='(512, 512, 512, 512)' \
#   --agent.layer_norm=True \
#   --agent.critic_loss='td' \
#   --agent.train_q_agg='mean' \
#   --agent.vgf_q_agg='mean' \
#   --agent.bc_flow_steps=$bc_flow_steps \
#   --agent.vgf_particles=$vgf_particles \
#   --agent.train_particle_select=$train_particle_select \
#   --agent.eval_particle_select=$eval_particle_select \
#   --agent.train_vgf_steps=$train_vgf_steps \
#   --agent.vgf_lr=$vgf_lr \
#   --agent.vgf_alpha=$vgf_alpha \
#   --project_name='vgf_camera_ready_off' \
#   --run_group='ogbench' &

# sleep 2
# let "task=$task+1"


# GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
# CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
#   --env_name="ogbench:scene-play-singletask-${task_id}-v0" \
#   --seed=$seed \
#   --eval_interval=100000 \
#   --eval_episodes=50 \
#   --agent=agents/vgf.py \
#   --agent.discount=0.99 \
#   --agent.actor_lr=0.0003 \
#   --agent.value_lr=0.0003 \
#   --agent.value_hidden_dims='(512, 512, 512, 512)' \
#   --agent.actor_hidden_dims='(512, 512, 512, 512)' \
#   --agent.layer_norm=True \
#   --agent.critic_loss='td' \
#   --agent.train_q_agg='mean' \
#   --agent.vgf_q_agg='mean' \
#   --agent.bc_flow_steps=$bc_flow_steps \
#   --agent.vgf_particles=$vgf_particles \
#   --agent.train_particle_select=$train_particle_select \
#   --agent.eval_particle_select=$eval_particle_select \
#   --agent.train_vgf_steps=$train_vgf_steps \
#   --agent.vgf_lr=$vgf_lr \
#   --agent.vgf_alpha=$vgf_alpha \
#   --project_name='vgf_camera_ready_off' \
#   --run_group='ogbench' &

# sleep 2
# let "task=$task+1"


# GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
# CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
#   --env_name="ogbench:puzzle-3x3-play-singletask-${task_id}-v0" \
#   --seed=$seed \
#   --eval_interval=100000 \
#   --eval_episodes=50 \
#   --agent=agents/vgf.py \
#   --agent.discount=0.99 \
#   --agent.actor_lr=0.0003 \
#   --agent.value_lr=0.0003 \
#   --agent.value_hidden_dims='(512, 512, 512, 512)' \
#   --agent.actor_hidden_dims='(512, 512, 512, 512)' \
#   --agent.layer_norm=True \
#   --agent.critic_loss='td' \
#   --agent.train_q_agg='mean' \
#   --agent.vgf_q_agg='mean' \
#   --agent.bc_flow_steps=$bc_flow_steps \
#   --agent.vgf_particles=$vgf_particles \
#   --agent.train_particle_select=$train_particle_select \
#   --agent.eval_particle_select=$eval_particle_select \
#   --agent.train_vgf_steps=$train_vgf_steps \
#   --agent.vgf_lr=$vgf_lr \
#   --agent.vgf_alpha=$vgf_alpha \
#   --project_name='vgf_camera_ready_off' \
#   --run_group='mcmc' &

# sleep 2
# let "task=$task+1"


GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
  --env_name="ogbench:puzzle-4x4-play-singletask-${task_id}-v0" \
  --seed=$seed \
  --eval_interval=100000 \
  --eval_episodes=50 \
  --agent=agents/vgf.py \
  --agent.discount=0.99 \
  --agent.actor_lr=0.0003 \
  --agent.value_lr=0.0003 \
  --agent.value_hidden_dims='(512, 512, 512, 512)' \
  --agent.actor_hidden_dims='(512, 512, 512, 512)' \
  --agent.layer_norm=True \
  --agent.critic_loss='td' \
  --agent.train_q_agg='mean' \
  --agent.vgf_q_agg='mean' \
  --agent.bc_flow_steps=$bc_flow_steps \
  --agent.vgf_particles=$vgf_particles \
  --agent.train_particle_select=$train_particle_select \
  --agent.eval_particle_select=$eval_particle_select \
  --agent.train_vgf_steps=$train_vgf_steps \
  --agent.vgf_lr=$vgf_lr \
  --agent.vgf_alpha=$vgf_alpha \
  --agent.activations='relu' \
  --project_name='vgf_camera_ready_off' \
  --run_group='original' &

sleep 2
let "task=$task+1"

done

done

done

done

done

done

done

done

done

# done
