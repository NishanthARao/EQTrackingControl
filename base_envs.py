import numpy as np
import scipy.linalg
from flax import struct
import jax
import jax.numpy as jnp
from jax import random as jrandom
from jax import lax
from jax import jit

from typing import Any, Dict, Optional, Tuple, Union
from functools import partial

from gymnax.environments import spaces

@struct.dataclass
class EnvState:
    time: float

@struct.dataclass
class PointState(EnvState):
    pos: jnp.ndarray
    vel: jnp.ndarray
    ref_pos: jnp.ndarray
    ref_vel: jnp.ndarray
    ref_acc: jnp.ndarray
    lqr_cmd: jnp.ndarray

@struct.dataclass
class PointVelocityState(EnvState):
    pos: jnp.ndarray
    vel: jnp.ndarray
    ref_pos: jnp.ndarray
    ref_vel: jnp.ndarray
    ref_acc: jnp.ndarray

@struct.dataclass
class PointRandomWalkState(EnvState):
    rnd_key: jnp.ndarray
    pos: jnp.ndarray
    vel: jnp.ndarray
    ref_pos: jnp.ndarray
    ref_vel: jnp.ndarray
    ref_acc: jnp.ndarray


@struct.dataclass
class PointLissajousTrackingState(EnvState):
    pos: jnp.ndarray
    vel: jnp.ndarray
    ref_pos: jnp.ndarray
    ref_vel: jnp.ndarray
    ref_acc: jnp.ndarray
    amplitudes: jnp.ndarray
    frequencies: jnp.ndarray
    phases: jnp.ndarray


class PointParticleBase:
    def __init__ (self, ref_pos=None, equivariant=False, state_cov_scalar=0.5, ref_cov_scalar=3.0, dt=0.05, max_time=100.0, terminate_on_error=True, 
                  reward_q_pos = 1e-2, reward_q_vel = 1e-2, reward_r = 1e-4, termination_bound = 10., terminal_reward = 0.0, reward_reach=False, 
                  use_des_action_in_reward=True, clip_actions=True, **kwargs):
        self.state_mean = jnp.array([0., 0., 0.])
        self.state_cov = jnp.eye(3) * state_cov_scalar
        self.ref_mean = jnp.array([0., 0., 0.])
        self.ref_cov = jnp.eye(3) * ref_cov_scalar
        self.predefined_ref_pos = ref_pos

        self.max_time = max_time
        self.dt = dt
        self.equivariant = equivariant
        
        self.terminate_on_error = terminate_on_error
        self.termination_bound = termination_bound
        self.terminal_reward = terminal_reward
        self.reach_reward = reward_reach
        self.reward_q_pos = reward_q_pos
        self.reward_q_vel = reward_q_vel
        self.reward_r = reward_r
        self.use_des_action_in_reward = use_des_action_in_reward
        self.clip_actions = clip_actions

        self.epsilon_ball_radius = 1e-2

        self.other_args = kwargs

        # For LQR calculations
        A = np.eye(6,)
        A[:3, 3:] = dt * np.eye(3,)

        B = np.zeros((6, 3))
        B[3:, :] = dt * np.eye(3,)

        Q = 10 * np.eye(6)
        R = 1 * np.eye(3)

        self.gamma = 0.995

        S = scipy.linalg.solve_discrete_are((self.gamma ** 0.5) * A, B, Q, 1/(self.gamma) * R)

        K = np.linalg.inv(R + self.gamma * B.T @ S @ B) @ B.T @ S @ A

        self.K = jnp.asarray(K)

    def _sample_random_ref_pos(self, key):
        """
        Helper function to sample a random reference position from a multivariate normal distribution
        """
        key, new_key = jrandom.split(key)
        return jrandom.multivariate_normal(new_key, self.ref_mean, self.ref_cov)
    
    def _get_predefined_ref_pos(self, key):
        """
        Helper function to get the predefined reference position
        """
        return jnp.array(self.predefined_ref_pos) if self.predefined_ref_pos is not None else jnp.zeros_like(self.ref_mean)
    
    def _maybe_reset(self, key, env_state, done):
        '''
        Reset helper to work with the jax conditional flow. 
        If the done flag is True, run self._reset with the key. 
        If done is False, return the env_state as is.
        '''
        return lax.select(done, self._reset(key), env_state)
    
    def _is_terminal(self, env_state,):
        """
        Helper function to check if the environment state is terminal.
        If self.terminate_on_error is True, then the environment is terminal if the particle is outside the world bounds or exceeds the velocity error.
        """
            
        # if any part of the state is outside the bounds of +- 5, then the episode is done
        # or if the time is greater than the max time

        time_exceeded = env_state.time > self.max_time

        world_cond = self._is_terminal_error(env_state)
        world_cond = lax.select(self.terminate_on_error, world_cond, False)

        return jnp.logical_or(world_cond, time_exceeded)
    
    def _is_terminal_error(self, env_state):
        # outside_world_bounds = jnp.any(jnp.linalg.norm(env_state.ref_pos - env_state.pos)**2 > self.termination_bound)
        # exceeded_error_velocity = jnp.any(jnp.linalg.norm(env_state.ref_vel - env_state.vel)**2 > self.termination_bound)
        outside_world_bounds = jnp.any(jnp.abs(env_state.ref_pos - env_state.pos) > self.termination_bound)
        exceeded_error_velocity = jnp.any(jnp.abs(env_state.ref_vel - env_state.vel) > self.termination_bound)
        # exceeded_error_velocity = False

        return jnp.logical_or(outside_world_bounds, exceeded_error_velocity)

    def _is_terminal_reach(self, env_state):

        return jnp.any(jnp.linalg.norm(env_state.ref_pos - env_state.pos) ** 2 < self.epsilon_ball_radius)

    def _get_reward(self, env_state, action, des_action=0.0):
        '''
        Get reward from the environment state. 
        Reward is defined as the "LQR" cost function: scaled position error and scaled velocity error
        '''
        state = env_state
        termination_from_error = self._is_terminal_error(state)

        terminal_reward = lax.select(termination_from_error, self.terminal_reward, 0.0)

        #dest_reached = self._is_terminal_reach(state)
        #dest_reach_reward = lax.select(dest_reached, self.reach_reward, 0.0)
        dest_reach_reward_modified = 1.0 - jnp.tanh(0.01 * jnp.linalg.norm(env_state.ref_pos - env_state.pos) / (self.epsilon_ball_radius * 0.5))
        
        dest_reach_reward = lax.select(self.reach_reward, dest_reach_reward_modified, 0.0)

        reward_no_des_act = -self.reward_q_pos * (jnp.linalg.norm(state.ref_pos - state.pos)**2) - self.reward_q_vel * (jnp.linalg.norm(state.ref_vel - state.vel)**2) - self.reward_r * (jnp.linalg.norm(action)**2) + terminal_reward + dest_reach_reward
        reward_yes_des_act = -self.reward_q_pos * (jnp.linalg.norm(state.ref_pos - state.pos)**2) - self.reward_q_vel * (jnp.linalg.norm(state.ref_vel - state.vel)**2) - self.reward_r * (jnp.linalg.norm(action - des_action)**2) + terminal_reward + dest_reach_reward

        final_reward = lax.select(self.use_des_action_in_reward, reward_yes_des_act, reward_no_des_act)

        #return -self.reward_q * (jnp.linalg.norm(state.ref_pos - state.pos)**2 + jnp.linalg.norm(state.ref_vel - state.vel)**2) - self.reward_r * (jnp.linalg.norm(action)**2) + terminal_reward + dest_reach_reward
        #return -self.reward_q_pos * (jnp.linalg.norm(state.ref_pos - state.pos)**2) - self.reward_q_vel * (jnp.linalg.norm(state.ref_vel - state.vel)**2) - self.reward_r * (jnp.linalg.norm(action)**2) + terminal_reward
        ## Add (a - a_{des}) -> Important for random walk envs
        #return -self.reward_q_pos * (jnp.linalg.norm(state.ref_pos - state.pos)**2) - self.reward_q_vel * (jnp.linalg.norm(state.ref_vel - state.vel)**2) - self.reward_r * (jnp.linalg.norm(action - des_action)**2) + terminal_reward
        return final_reward

    @property
    def num_actions(self) -> int:
        return 3
    
    def action_space(self) -> spaces.Box:
        low = jnp.array([-1., -1., -1.])
        high = jnp.array([1., 1., 1.])
        return spaces.Box(low, high, (3,), jnp.float32)