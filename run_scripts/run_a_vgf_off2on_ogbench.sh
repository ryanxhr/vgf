#!/bin/bash

# Script to reproduce vgf in d4rl results

GPU_LIST=(0 1 2 3)


for seed in 99; do

for bc_flow_steps in 10; do

for vgf_particles in 5; do

# for vgf_lr in 0.05; do

# for vgf_alpha in 1.0; do

# for train_vgf_steps in 1; do

# for train_q_agg in 'mean'; do

for train_particle_select in 'mean'; do

for eval_particle_select in 'max'; do

# for task_id in task4; do


GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
  --env_name="ogbench:antmaze-large-navigate-singletask-v0" \
  --seed=$seed \
  --offline_steps=1000000 \
  --online_steps=1000000 \
  --eval_interval=100000 \
  --eval_episodes=50 \
  --agent=agents/a_vgf.py \
  --agent.discount=0.995 \
  --agent.batch_size=256 \
  --agent.actor_lr=0.0003 \
  --agent.value_lr=0.0003 \
  --agent.value_hidden_dims='(512, 512, 512, 512)' \
  --agent.actor_hidden_dims='(512, 512, 512, 512)' \
  --agent.layer_norm=True \
  --agent.critic_loss='td' \
  --agent.train_q_agg='min' \
  --agent.vgf_q_agg='mean' \
  --agent.bc_flow_steps=$bc_flow_steps \
  --agent.vgf_particles=$vgf_particles \
  --agent.train_particle_select=$train_particle_select \
  --agent.eval_particle_select=$eval_particle_select \
  --agent.train_vgf_steps=5 \
  --agent.vgf_lr=0.1 \
  --agent.q_grad_norm=False \
  --agent.activations='relu' \
  --project_name='vgf_camera_ready_off2on' \
  --run_group='a_vgf;gradient_clip' &

sleep 2
let "task=$task+1"


GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
  --env_name="ogbench:antmaze-giant-navigate-singletask-v0" \
  --seed=$seed \
  --normalize_r=1 \
  --offline_steps=1000000 \
  --online_steps=1000000 \
  --eval_interval=100000 \
  --eval_episodes=50 \
  --agent=agents/a_vgf.py \
  --agent.discount=0.995 \
  --agent.batch_size=256 \
  --agent.actor_lr=0.0003 \
  --agent.value_lr=0.0003 \
  --agent.value_hidden_dims='(512, 512, 512, 512)' \
  --agent.actor_hidden_dims='(512, 512, 512, 512)' \
  --agent.layer_norm=True \
  --agent.critic_loss='td' \
  --agent.train_q_agg='min' \
  --agent.vgf_q_agg='mean' \
  --agent.bc_flow_steps=$bc_flow_steps \
  --agent.vgf_particles=$vgf_particles \
  --agent.train_particle_select=$train_particle_select \
  --agent.eval_particle_select=$eval_particle_select \
  --agent.train_vgf_steps=5 \
  --agent.vgf_lr=0.1 \
  --agent.q_grad_norm=False \
  --agent.activations='relu' \
  --project_name='vgf_camera_ready_off2on' \
  --run_group='a_vgf;gradient_clip' &

sleep 2
let "task=$task+1"


GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
  --env_name="ogbench:humanoidmaze-medium-navigate-singletask-v0" \
  --seed=$seed \
  --offline_steps=1000000 \
  --online_steps=1000000 \
  --eval_interval=100000 \
  --eval_episodes=50 \
  --agent=agents/a_vgf.py \
  --agent.discount=0.995 \
  --agent.batch_size=256 \
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
  --agent.train_vgf_steps=1 \
  --agent.vgf_lr=0.05 \
  --agent.q_grad_norm=False \
  --agent.activations='relu' \
  --project_name='vgf_camera_ready_off2on' \
  --run_group='a_vgf;gradient_clip' &

sleep 2
let "task=$task+1"


GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
  --env_name="ogbench:humanoidmaze-large-navigate-singletask-v0" \
  --seed=$seed \
  --offline_steps=1000000 \
  --online_steps=1000000 \
  --eval_interval=100000 \
  --eval_episodes=50 \
  --agent=agents/a_vgf.py \
  --agent.discount=0.995 \
  --agent.batch_size=256 \
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
  --agent.train_vgf_steps=1 \
  --agent.vgf_lr=0.05 \
  --agent.q_grad_norm=False \
  --agent.activations='relu' \
  --project_name='vgf_camera_ready_off2on' \
  --run_group='a_vgf;gradient_clip' &

sleep 2
let "task=$task+1"


GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
  --env_name="ogbench:antsoccer-arena-navigate-singletask-v0" \
  --seed=$seed \
  --offline_steps=1000000 \
  --online_steps=1000000 \
  --eval_interval=100000 \
  --eval_episodes=50 \
  --agent=agents/a_vgf.py \
  --agent.discount=0.995 \
  --agent.batch_size=256 \
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
  --agent.train_vgf_steps=2 \
  --agent.vgf_lr=0.05 \
  --agent.q_grad_norm=False \
  --agent.activations='relu' \
  --project_name='vgf_camera_ready_off2on' \
  --run_group='a_vgf;gradient_clip' &

sleep 2
let "task=$task+1"


GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
  --env_name="ogbench:cube-single-play-singletask-v0" \
  --seed=$seed \
  --offline_steps=1000000 \
  --online_steps=1000000 \
  --eval_interval=100000 \
  --eval_episodes=50 \
  --agent=agents/a_vgf.py \
  --agent.discount=0.99 \
  --agent.batch_size=256 \
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
  --agent.train_vgf_steps=1 \
  --agent.vgf_lr=0.05 \
  --agent.q_grad_norm=False \
  --agent.activations='relu' \
  --project_name='vgf_camera_ready_off2on' \
  --run_group='a_vgf;gradient_clip' &

sleep 2
let "task=$task+1"


GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
  --env_name="ogbench:cube-double-play-singletask-v0" \
  --seed=$seed \
  --offline_steps=1000000 \
  --online_steps=1000000 \
  --eval_interval=100000 \
  --eval_episodes=50 \
  --agent=agents/a_vgf.py \
  --agent.discount=0.99 \
  --agent.batch_size=256 \
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
  --agent.train_vgf_steps=1 \
  --agent.vgf_lr=0.05 \
  --agent.q_grad_norm=False \
  --agent.activations='relu' \
  --project_name='vgf_camera_ready_off2on' \
  --run_group='a_vgf;gradient_clip' &

sleep 2
let "task=$task+1"


GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
  --env_name="ogbench:scene-play-singletask-v0" \
  --seed=$seed \
  --offline_steps=1000000 \
  --online_steps=1000000 \
  --eval_interval=100000 \
  --eval_episodes=50 \
  --agent=agents/a_vgf.py \
  --agent.discount=0.99 \
  --agent.batch_size=256 \
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
  --agent.train_vgf_steps=1 \
  --agent.vgf_lr=0.1 \
  --agent.q_grad_norm=False \
  --agent.activations='relu' \
  --project_name='vgf_camera_ready_off2on' \
  --run_group='a_vgf;gradient_clip' &

sleep 2
let "task=$task+1"


GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
  --env_name="ogbench:puzzle-3x3-play-singletask-v0" \
  --seed=$seed \
  --offline_steps=1000000 \
  --online_steps=1000000 \
  --eval_interval=100000 \
  --eval_episodes=50 \
  --agent=agents/a_vgf.py \
  --agent.discount=0.99 \
  --agent.batch_size=256 \
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
  --agent.train_vgf_steps=5 \
  --agent.vgf_lr=0.1 \
  --agent.q_grad_norm=False \
  --agent.activations='relu' \
  --project_name='vgf_camera_ready_off2on' \
  --run_group='a_vgf;gradient_clip' &

sleep 2
let "task=$task+1"


GPU_DEVICE=${GPU_LIST[task%${#GPU_LIST[@]}]}
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python main.py \
  --env_name="ogbench:puzzle-4x4-play-singletask-v0" \
  --seed=$seed \
  --offline_steps=1000000 \
  --online_steps=1000000 \
  --eval_interval=100000 \
  --eval_episodes=50 \
  --agent=agents/a_vgf.py \
  --agent.discount=0.99 \
  --agent.batch_size=256 \
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
  --agent.train_vgf_steps=5 \
  --agent.vgf_lr=0.1 \
  --agent.q_grad_norm=False \
  --agent.activations='relu' \
  --project_name='vgf_camera_ready_off2on' \
  --run_group='a_vgf;gradient_clip' &

sleep 2
let "task=$task+1"

# done

done

done

done

done

done

# done

# done

# done

# done
