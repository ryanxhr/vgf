from typing import Any, Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp
import jax

def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.relu  # nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False
    dropout_rate: Optional[float] = None
    ln_first: bool = True

    @nn.compact
    def __call__(self, x, training=False):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.ln_first:
                    # different from fql
                    if self.layer_norm:
                        x = nn.LayerNorm()(x)
                    x = self.activations(x)
                else:
                    x = self.activations(x)
                    if self.layer_norm:
                        x = nn.LayerNorm()(x)
                # if self.dropout_rate is not None and self.dropout_rate > 0:
                #     x = nn.Dropout(rate=self.dropout_rate)(
                #         x, deterministic=not training)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class Actor(nn.Module):
    """Gaussian actor network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    activations: str = 'relu' 
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: Optional[float] = None
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        if self.activations == 'relu':
            activations = nn.relu
        elif self.activations == 'gelu':
            activations = nn.gelu
        elif self.activations == 'swish':
            activations = nn.swish
        else:
            raise ValueError(f"Activation {self.activations} not found in flax.nn")

        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm, activations=activations)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if self.const_std is not None:
                self.log_stds = self.param('log_stds', nn.initializers.constant(jnp.log(self.const_std)), (self.action_dim,))
                self.log_stds = jax.lax.stop_gradient(self.log_stds)
            else:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        temperature=1.0,
    ):
        """Return action distributions.

        Args:
            observations: Observations.
            temperature: Scaling factor for the standard deviation.
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)

        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        if self.tanh_squash:
            distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))
        else:
            # means = nn.tanh(means)
            distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)

        return distribution


class Value(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        num_ensembles: Number of ensemble components.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = False
    activations: str = 'relu' 
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.activations == 'relu':
            activations = nn.relu
        elif self.activations == 'gelu':
            activations = nn.gelu
        elif self.activations == 'swish':
            activations = nn.swish
        else:
            raise ValueError(f"Activation {self.activations} not found in flax.nn")

        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm, activations=activations)

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v


class WFunction(nn.Module):
    """weight network.

    This module can be used for W(s, a, adv) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = False
    w_residual: bool = False
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        value_net = mlp_class((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, actions, adv):
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)
        w = self.value_net(inputs).squeeze(-1)
        
        return w + adv if self.w_residual else w


class ActorVectorField(nn.Module):
    """Vector field network for actor flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    activations: str = 'relu' 
    encoder: nn.Module = None

    def setup(self) -> None:
        if self.activations == 'relu':
            activations = nn.relu
        elif self.activations == 'gelu':
            activations = nn.gelu
        elif self.activations == 'swish':
            activations = nn.swish
        else:
            raise ValueError(f"Activation {self.activations} not found in flax.nn")

        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm, activations=activations)

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = self.mlp(inputs)

        return v


class ValueVectorField(nn.Module):
    """Vector field network for value flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = False
    activations: str = 'relu' 
    encoder: nn.Module = None

    def setup(self) -> None:
        if self.activations == 'relu':
            activations = nn.relu
        elif self.activations == 'gelu':
            activations = nn.gelu
        elif self.activations == 'swish':
            activations = nn.swish
        else:
            raise ValueError(f"Activation {self.activations} not found in flax.nn")

        self.mlp = MLP((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm, activations=activations)

    @nn.compact
    def __call__(self, observations, actions, values, times=None, is_encoded=False):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            values: Values.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions, values], axis=-1)
        else:
            inputs = jnp.concatenate([observations, actions, values, times], axis=-1)

        v = self.mlp(inputs)

        return v


class ValueGrad(nn.Module):
    """Value Gradient Network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    activations: str = 'relu' 
    encoder: nn.Module = None

    def setup(self) -> None:
        if self.activations == 'relu':
            activations = nn.relu
        elif self.activations == 'gelu':
            activations = nn.gelu
        elif self.activations == 'swish':
            activations = nn.swish
        else:
            raise ValueError(f"Activation {self.activations} not found in flax.nn")

        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm, activations=activations)

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
            
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = self.mlp(inputs)

        return v