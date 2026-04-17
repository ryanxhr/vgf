
import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import jax.random as jr
import ml_collections
import optax
from functools import partial

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value, Actor


def iql_loss(adv, expectile):
    """Compute the IQL loss."""
    # adv = jnp.minimum(adv, 5.0)  # clip to prevent from gradeint exploding
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * (adv**2)


def sql_loss(x, expectile):
    """Compute the SQL loss."""
    # x = jnp.minimum(x, 5.0)  # clip to prevent from gradeint exploding
    sp_term = x / (2 * expectile) + 1.0
    sp_weight = jnp.where(sp_term > 0, 1., 0.)
    return expectile * sp_weight * (sp_term**2) - x


def target_update(model, target_model, tau):
    """Update the target network."""
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), model.params, target_model.params
    )
    return target_model.replace(params=new_target_params)


def rbf_kernel(X, Y, sigma=None):
    """
    X: [B, n, d], Y: [B, m, d]
    returns K_XY: [B, n, m] with RBF kernel entries
    """
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    X2 = jnp.sum(X * X, axis=-1, keepdims=True)                                 # [B, n, 1]
    Y2 = jnp.sum(Y * Y, axis=-1, keepdims=True).transpose(0, 2, 1)              # [B, 1, m]
    XY = jnp.matmul(X, Y.transpose(0, 2, 1))                                    # [B, n, m]
    dnorm2 = X2 + Y2 - 2.0 * XY                                                 # [B, n, m]
    dnorm2 = jnp.maximum(dnorm2, 0.0)

    if sigma is None:
        # Median heuristic per batch
        h = jnp.median(dnorm2, axis=(1,2)) / (2.0 * jnp.log(X.shape[1] + 1.0))  # [B]
        sigma_val = jnp.sqrt(jnp.maximum(h, 1e-12))                             # [B]
        # reshape and broadcast to [B, n, m]
        sigma_val = sigma_val[:, None, None]
    else:
        sigma_val = jnp.asarray(sigma)
        if sigma_val.ndim == 0:
            sigma_val = jnp.broadcast_to(sigma_val, (X.shape[0], 1, 1))

    gamma = 1.0 / (1e-6 + 2.0 * (sigma_val ** 2))
    K_XY = jnp.exp(-gamma * dnorm2)                                             # [B, n, m]
    return K_XY, dnorm2, gamma*2

# def rbf_kernel(X, Y, sigma=None, min_sigma=1e-6, clip_exp_low=-80.0):
#     """
#     X: [B,n,d], Y: [B,m,d]
#     return: K_XY [B,n,m], d2 [B,n,m], twice_gamma [B,1,1]
#     """
#     diff = X[:, :, None, :] - Y[:, None, :, :]
#     d2 = jnp.sum(diff * diff, axis=-1)  # [B,n,m]

#     if sigma is None:
#         B, n, _ = X.shape
#         if n == Y.shape[1]:
#             if n > 1:
#                 iu, ju = jnp.triu_indices(n, k=1)
#                 d2_ut = d2[:, iu, ju]     # [B, L], L = n*(n-1)//2
#                 med = jnp.median(d2_ut, axis=1)
#             else: 
#                 med = jnp.full((B,), min_sigma**2, dtype=d2.dtype)
#         else:
#             med = jnp.median(d2, axis=(1, 2))
#         h = med / (2.0 * jnp.log(n + 1.0))
#         sigma_val = jnp.sqrt(jnp.maximum(h, min_sigma**2))[:, None, None]
#     else:
#         sigma_val = jnp.asarray(sigma)
#         if sigma_val.ndim == 0:
#             sigma_val = jnp.broadcast_to(sigma_val, (X.shape[0], 1, 1))

#     gamma = 1.0 / (2.0 * sigma_val**2)             # [B,1,1]
#     arg = -gamma * d2                               # <= 0
#     arg = jnp.clip(arg, clip_exp_low, 0.0) 
#     K = jnp.exp(arg)
#     return K, d2, 2.0 * gamma


class SVGD_VGF:
    def __init__(self, q, alpha, q_agg, optimizer: optax.GradientTransformation):
        """
        q: state-action value function
        optimizer: an optax optimizer, e.g. optax.adam(1e-2)
        """
        self.q = q
        self.alpha = alpha
        self.q_agg = q_agg
        self.optim = optimizer
        self.opt_state = None

    def init(self, particles):
        """Initialize optimizer state for the particle array X."""
        self.opt_state = self.optim.init(particles)
        return particles, self.opt_state

    def phi(self, obs, particles):
        # obs: [B, D], particles: [B, N, D]
        # Score terms
        def sum_q(action):
            obs_flatten = obs.reshape(-1, obs.shape[-1])                        # [B*N, D]
            action_flatten = action.reshape(-1, action.shape[-1])               # [B*N, D]
            qs = self.q(obs_flatten, action_flatten)
            q = jnp.min(qs, axis=0) if self.q_agg == 'min' else jnp.mean(qs, axis=0)
            # q normalization to stabilize gradient (doesn't work)
            # q /= jax.lax.stop_gradient(jnp.mean(jnp.abs(q)))
            return jnp.sum(q)
        score = jax.grad(sum_q)(particles)                                 # [B, N, D]

        # Kernel terms
        particles_stop = jax.lax.stop_gradient(particles)
        K_xx, K_dist, K_gamma = rbf_kernel(particles, particles_stop)                            # [B, N, N]
        K_xx = jax.lax.stop_gradient(K_xx)

        # grad_K := -∂/∂X sum_{i,j} K(X_i, X_j_stop)
        def sum_K(x):
            return jnp.sum(rbf_kernel(x, particles_stop)[0])
        grad_K = -jax.grad(sum_K)(particles)                                    # [B, N, D]

        # φ(X) = (K_xx * score + grad_K) / N
        phi_val = (K_xx @ score / self.alpha + grad_K) / particles.shape[1]
        # phi_val = score / particles.shape[1]  # mcmc   

        # # Tr = [\nabla_{X_i} k(X_i, X_j)]^\top [\nabla_{X_j} Q(s, X_j)] + \alpha * k(X_i, X_j)[||X_i-X_j||^2 / \sigma^4 - d / \sigma^2]
        # term_1 = grad_K @ score.transpose((0, 2, 1))     # [B, N, N]
        # term_2 = self.alpha * K_xx * (particles.shape[-1] * K_gamma - K_dist * K_gamma**2)    # [B, N, N]
        # trace = (term_1 + term_2).mean(axis=(1, 2))    # [B,]
        # return phi_val, K_xx @ score, grad_K, trace         
        return phi_val      

    def step(self, obs, particles, opt_state):
        """In Optax, we pass "grads" = -phi(X), which yields X ← X + lr * phi(X)"""
        phi_val = self.phi(obs, particles)
        grads = -phi_val

        updates, new_opt_state = self.optim.update(grads, opt_state, params=particles)
        new_particles = optax.apply_updates(particles, updates)
        return new_particles, new_opt_state


class VGFAgent(flax.struct.PyTreeNode):
    """Value Gradient Flow (VGF) agent."""

    rng: Any
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    actor_bc: TrainState
    config: Any = nonpytree_field()

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, _ = jax.random.split(self.rng)

        def value_loss_fn(value_params):
            _, noise_rng = jax.random.split(self.rng)
            target_actions = self.sample_bc_actions(batch['observations'], seed=noise_rng)
            # target_actions = batch['actions']            
            qs = self.target_critic(batch['observations'], actions=target_actions)
            v = self.value(batch['observations'], params=value_params)
            if self.config['train_q_agg'] == 'min':
                q = qs.min(axis=0)
            else:
                q = qs.mean(axis=0)
            
            value_loss = iql_loss(q - v, self.config['expectile']).mean()  # iql

            return value_loss, {
                'value_loss': value_loss,
                'v_mean': v.mean(),
                'adv_mean': (q-v).mean(),
            }

        def critic_loss_fn(critic_params):
            batch_size, action_dim = batch['actions'].shape
            _, noise_rng = jax.random.split(self.rng)

            # compute bc actions
            next_obs_rep = jnp.repeat(jnp.expand_dims(batch['next_observations'], 1), self.config['vgf_particles'], axis=1)
            next_obs_rep = next_obs_rep.reshape(-1, next_obs_rep.shape[-1])
            bc_next_actions = self.sample_bc_actions(next_obs_rep, noise_rng)
            bc_next_actions = bc_next_actions.reshape(batch_size, self.config['vgf_particles'], -1)
            next_obs_rep = next_obs_rep.reshape(batch_size, self.config['vgf_particles'], -1)

            # value gradient flow
            svgd = SVGD_VGF(self.critic, self.config['vgf_alpha'], self.config['vgf_q_agg'], optax.adam(learning_rate=self.config['vgf_lr']))
            particles, opt_state = svgd.init(bc_next_actions)
            # phi_q_list, phi_k_list = [], []

            for _ in range(self.config['train_vgf_steps']):
                particles, new_opt_state = svgd.step(next_obs_rep, particles, opt_state)
                particles = jnp.clip(particles, -1, 1)
                # phi_q_list.append(phi_q)
                # phi_k_list.append(phi_k)
                opt_state = new_opt_state

            next_obs_flatten = next_obs_rep.reshape(-1, next_obs_rep.shape[-1])
            next_actions_flatten = particles.reshape(-1, particles.shape[-1])
            next_qs = self.target_critic(next_obs_flatten, actions=next_actions_flatten)      # [2, B*N]
            if self.config['train_q_agg'] == 'min':
                next_q = next_qs.min(axis=0)
            else:
                next_q = next_qs.mean(axis=0)
            
            # particle selection
            next_q = next_q.reshape(-1, self.config['vgf_particles'])  
            if self.config['train_particle_select'] == 'max':
                next_q = jnp.max(next_q, axis=1)    # shape (B,)
            else:
                next_q = jnp.mean(next_q, axis=1)   # shape (B,)
            
            target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q
            q1, q2 = self.critic(batch['observations'], actions=batch['actions'], params=critic_params)

            if self.config['critic_loss']  == 'td':  # TD update
                critic_loss = ((target_q - q1) ** 2 + (target_q - q2) ** 2).mean()
            elif self.config['critic_loss'] == 'sql-q':
                critic_loss = (sql_loss(target_q - q1, self.config['expectile']) + sql_loss(target_q - q2, self.config['expectile'])).mean()
            elif self.config['critic_loss'] == 'iql-q':
                critic_loss = (iql_loss(target_q - q1, self.config['expectile']) + iql_loss(target_q - q2, self.config['expectile'])).mean()

            q_info = {
                'critic_loss': critic_loss,
                'q_mean': q1.mean(),
            }
            # if self.config['critic_loss']  == 'td':
            # q_info.update({f"phi_q_{i}": v.mean() for i, v in enumerate(phi_q_list)})
            # q_info.update({f"phi_k_{i}": v.mean() for i, v in enumerate(phi_k_list)})
            q_info['bc_flow_var'] = jnp.var(bc_next_actions, axis=1, ddof=1).mean()
            q_info['vgf_var'] = jnp.var(particles, axis=1, ddof=1).mean()
            # 'actor_actions_abs_mean': jnp.abs(actor_actions).mean(),
            # 'dataset_actions_abs_mean': jnp.abs(batch['actions']).mean(),
            # 'mse': jnp.mean((actor_actions - batch['actions']) ** 2),

            return critic_loss, q_info

        def actor_bc_loss_fn(actor_bc_params):
            """bc flow loss."""
            batch_size, action_dim = batch['actions'].shape

            if self.config['bc_policy_type'] == 'flow': 
                _, x_rng, t_rng = jax.random.split(self.rng, 3)

                # BC flow loss.
                x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
                x_1 = batch['actions']
                t = jax.random.uniform(t_rng, (batch_size, 1))
                x_t = (1 - t) * x_0 + t * x_1
                vel = x_1 - x_0

                pred = self.actor_bc(batch['observations'], x_t, t, params=actor_bc_params)
                bc_loss = jnp.mean((pred - vel) ** 2)
            elif self.config['bc_policy_type'] == 'gau': 
                dist = self.actor_bc(batch['observations'], params=actor_bc_params)
                log_prob = dist.log_prob(batch['actions'])
                bc_loss = -jnp.mean(log_prob)
            else:
                raise ValueError(f"Unknown bc_policy_type: {self.config['bc_policy_type']}")

            return bc_loss, {
                'bc_loss': bc_loss,
            }

        new_value, value_info = self.value, {}  
        new_critic, critic_info = self.critic.apply_loss_fn(loss_fn=critic_loss_fn)
        new_target_critic = target_update(self.critic, self.target_critic, self.config['tau'])
        new_actor_bc, actor_bc_info = self.actor_bc.apply_loss_fn(loss_fn=actor_bc_loss_fn)

        return self.replace(rng=new_rng, critic=new_critic, target_critic=new_target_critic, value=new_value, actor_bc=new_actor_bc), {
            **value_info, **critic_info, **actor_bc_info
        }

    @partial(jax.jit, static_argnames=("eval_vgf_steps",))
    def sample_actions(
        self,
        observations,
        seed=None,
        eval_vgf_steps=0,
        temperature=1.0,
        deterministic=True,
    ):
        """(Evaluation) Sample actions via value gradient flow."""
        obs = jnp.expand_dims(observations, 0)
        obs_rep = jnp.repeat(jnp.expand_dims(obs, 1), self.config['vgf_particles'], axis=1)
        obs_rep = obs_rep.reshape(-1, obs_rep.shape[-1])

        # generate bc flow actions: [1, particle_num, action_dim]
        _, noise_seed = jax.random.split(seed)
        bc_actions = self.sample_bc_actions(obs_rep, noise_seed)
        bc_actions = bc_actions.reshape(-1, self.config['vgf_particles'], bc_actions.shape[-1])
        obs_rep = obs_rep.reshape(-1, self.config['vgf_particles'], obs_rep.shape[-1])

        # value gradient flow
        svgd = SVGD_VGF(self.critic, self.config['vgf_alpha'], self.config['vgf_q_agg'], optax.adam(learning_rate=self.config['vgf_lr']))
        particles, opt_state = svgd.init(bc_actions)

        for _ in range(eval_vgf_steps):
            particles, new_opt_state = svgd.step(obs_rep, particles, opt_state)
            particles = jnp.clip(particles, -1, 1)
            opt_state = new_opt_state

        obs_flatten = obs_rep.reshape(-1, obs_rep.shape[-1])
        particles_flatten = particles.reshape(-1, particles.shape[-1])
        qs = self.critic(obs_flatten, particles_flatten)
        if self.config['train_q_agg'] == 'min':
            q = jnp.min(qs, axis=0)
        else:
            q = jnp.mean(qs, axis=0)

        # particle selection (1, N, act_dim)
        q = q.reshape(-1, self.config['vgf_particles'])  
        if self.config['eval_particle_select'] == 'max':
            best_idx = jnp.argmax(q, axis=1)   
        elif self.config['eval_particle_select'] == 'softmax':
            logits = (q - jnp.max(q, axis=1, keepdims=True)) / self.config['softmax_temp']      # temperature = 1
            best_idx = jr.categorical(seed, logits, axis=1)             
        else:  # random selection
            B, N = q.shape
            best_idx = jr.randint(seed, (B,), minval=0, maxval=N)

        actions = particles[jnp.arange(particles.shape[0]), best_idx]   # (1, D)
        return actions.squeeze() 

    @jax.jit
    def sample_bc_actions(
        self,
        observations,
        seed,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_encoder')(observations)

        if self.config['bc_policy_type'] == 'gau': 
            dist = self.actor_bc(observations)
            actions = dist.sample(seed=seed)
        elif self.config['bc_policy_type'] == 'flow':
            noises = jax.random.normal(
                seed,
                (
                    *observations.shape[: -len(self.config['ob_dims'])],
                    self.config['action_dim'],
                ), 
            )
            actions = noises
            # Euler method.
            for i in range(self.config['bc_flow_steps']):
                t = jnp.full((*observations.shape[:-1], 1), i / self.config['bc_flow_steps'])
                vels = self.actor_bc(observations, actions, t, is_encoded=True)
                actions = actions + vels / self.config['bc_flow_steps']
        else:
            raise ValueError(f"Unknown policy_type: {self.config['bc_policy_type']}")
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = encoder_module()
            encoders['critic'] = encoder_module()
            encoders['actor_bc'] = encoder_module()

        # Define bc actors
        if config['bc_policy_type'] == 'flow':
            actor_bc_def = ActorVectorField(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                layer_norm=config['actor_layer_norm'],
                activations=config['activations'],
                encoder=encoders.get('actor_bc'),
            )
            actor_bc_params = actor_bc_def.init(actor_key, ex_observations, ex_actions, ex_times)['params']
        elif config['bc_policy_type'] == 'gau':
            actor_bc_def = Actor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                layer_norm=config['actor_layer_norm'],
                activations=config['activations'],
                state_dependent_std=False,
                const_std=None,
                tanh_squash=config['bc_use_tanh'],
                encoder=encoders.get('actor_bc'),
            )
            actor_bc_params = actor_bc_def.init(actor_key, ex_observations)['params']
        else:
            raise ValueError(f"Unknown policy_type: {config['bc_policy_type']}")
        actor_bc = TrainState.create(actor_bc_def, actor_bc_params, tx=optax.adam(learning_rate=config['actor_lr']))

        # Define value functions
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            activations=config['activations'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        critic_params = critic_def.init(critic_key, ex_observations, ex_actions)['params']
        critic = TrainState.create(critic_def, critic_params, tx=optax.adam(learning_rate=config['value_lr']))
        target_critic = TrainState.create(critic_def, critic_params)

        value_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            activations=config['activations'],
            num_ensembles=1,
            encoder=encoders.get('value'),
        )
        value_params = value_def.init(value_key, ex_observations)['params']
        value = TrainState.create(value_def, value_params, tx=optax.adam(learning_rate=config['value_lr']))

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, critic=critic, target_critic=target_critic, value=value, actor_bc=actor_bc, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='vgf',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            value_lr=3e-4,  # Value learning rate.
            actor_lr=3e-4,  # BC actor learning rate.
            batch_size=256,  # Batch size.
            activations='relu',  # 'relu' or 'gelu'.
            value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
            actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            critic_loss='td',  # Critic loss type ('td' or 'ivr-q').
            expectile=0.9,  # SQL alpha or IQL expectile if choosing ivr in critic loss.
            train_q_agg='min',  # Aggregation method for target Q values.
            vgf_q_agg='mean',  # Aggregation method for Q values during evaluation.
            bc_policy_type='flow',  # ('gau' or 'flow') gaussian or flow policy as bc actor.
            bc_flow_steps=5,  # Number of bc flow steps.
            bc_use_tanh=False,  # Whether to use tanh squash for the bc actor.
            vgf_particles=10,  # Number of vgf particles.
            train_particle_select='mean',  # ('max' or 'mean').
            eval_particle_select='max',  # ('max', 'softmax' or 'mean').
            train_vgf_steps=1,  # Number of vgf steps during training.
            # eval_vgf_steps=1,  # Number of vgf steps during evaluation.
            vgf_lr=1e-1,  # Learning rate of vgf.
            vgf_alpha=1.0,  # Coefficient of the entropy.
            softmax_temp=1.0,
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
