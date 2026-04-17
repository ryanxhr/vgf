#!/bin/bash

# Script to reproduce vgf in d4rl results

GPU_LIST=(1 2 3)

env_list=(
    # "d4rl:hopper-medium-v2"
    # "d4rl:halfcheetah-medium-v2"
    # "d4rl:walker2d-medium-v2"
	  # "d4rl:hopper-medium-replay-v2"
	  # "d4rl:halfcheetah-medium-replay-v2"
	  # "d4rl:walker2d-medium-replay-v2"
    # "d4rl:hopper-medium-expert-v2"
    # "d4rl:halfcheetah-medium-expert-v2"
    # "d4rl:walker2d-medium-expert-v2"
    # "d4rl:antmaze-umaze-v2"
	  # "d4rl:antmaze-umaze-diverse-v2"
	  "d4rl:antmaze-medium-play-v2"
    # "d4rl:antmaze-medium-diverse-v2"
	  # "d4rl:antmaze-large-play-v2"
	  "d4rl:antmaze-large-diverse-v2"
    # "d4rl:pen-cloned-v1"
    # "d4rl:door-cloned-v1"
    # "d4rl:hammer-cloned-v1"
    # "d4rl:relocate-cloned-v1"
	)


for seed in 22; do

for env in ${env_list[*]}; do

for bc_policy_type in 'flow'; do

for bc_flow_steps in 10; do

for vgf_particles in 5; do

for vgf_lr in 0.1; do

for vgf_alpha in 1.0; do

for train_vgf_steps in 5; do

for train_particle_select in 'mean'; do

for eval_particle_select in 'max'; do

# TD
GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
  --env_name=$env \
  --seed=$seed \
  --normalize_r=1 \
  --offline_steps=1000000 \
  --online_steps=1000000 \
  --eval_interval=50000 \
  --eval_episodes=50 \
  --agent=agents/a_vgf.py \
  --agent.discount=0.99 \
  --agent.actor_lr=0.0003 \
  --agent.value_lr=0.0003 \
  --agent.value_hidden_dims='(512, 512, 512, 512)' \
  --agent.actor_hidden_dims='(512, 512, 512, 512)' \
  --agent.layer_norm=True \
  --agent.critic_loss='td' \
  --agent.train_q_agg='min' \
  --agent.vgf_q_agg='mean' \
  --agent.bc_policy_type=$bc_policy_type \
  --agent.bc_flow_steps=$bc_flow_steps \
  --agent.vgf_particles=$vgf_particles \
  --agent.train_particle_select=$train_particle_select \
  --agent.eval_particle_select=$eval_particle_select \
  --agent.train_vgf_steps=$train_vgf_steps \
  --agent.vgf_lr=$vgf_lr \
  --agent.vgf_alpha=$vgf_alpha \
  --agent.activations='relu' \
  --agent.q_grad_norm=False \
  --project_name='vgf_camera_ready_off2on' \
  --run_group='a_vgf;gradient_clip' &

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

done