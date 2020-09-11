import numpy as np
import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import policy_step as ps

class PublicGoodsEnv(py_environment.PyEnvironment):

  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(2,), dtype=np.float32, minimum=0, maximum = 1, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(12,2), dtype=np.float32, minimum=0, name='observation')
    s = np.zeros((12, 2))
    s[11] = [1,1]
    self._state = s
    self._episode_ended = False
    self._counter = 0
    self._END = 10
    self._MULT = 1.5


  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    s = np.zeros((12, 2))
    s[11] = [1,1]
    self._state = s
    self._episode_ended = False
    self._counter = 0
    return ts.restart(np.array(self._state, dtype=np.float32))

  def _step(self, action):

    self._counter += 1

    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self.reset()
    
    self._state[self._counter - 1, 0] = action[0]
    self._state[self._counter - 1, 1] = action[1]

    # Make sure episodes don't go on forever.
    if self._counter > self._END:
      self._episode_ended = True

    s1 = self._state[11, 0]
    a1 = self._state[self._counter - 1, 0] * s1
    s2 = self._state[11, 1]
    a2 = self._state[self._counter - 1, 1] * s2
    self._state[11, 0] = s1 - a1 + (a1 + a2)*self._MULT / 2
    self._state[11, 1] = s2 - a2 + (a1 + a2)*self._MULT / 2

    if self._episode_ended:
      reward_self = (self._state[11, 0] - s1)/s1
      reward_other = (self._state[11, 1] - s2)/s2
      my_reward = reward_fun(reward_self, reward_other)

      ret = ts.termination(np.array(self._state, dtype=np.float32), my_reward)
      self._counter += 1

      return ret
    else:
      reward_self = (self._state[11, 0] - s1)/s1
      reward_other = (self._state[11, 1] - s2)/s2
      my_reward = reward_fun(reward_self, reward_other)
      ret = ts.transition(
          np.array(self._state, dtype=np.float32), reward=my_reward, discount=1.0)
      
      for i in range(2):
        self._state[self._counter + 1, i] += self._state[self._counter, i]

      #print("transition:", ret)
      return ret
      
def construct_intended_action(policy0, policy1, time_step):
  action_step0 = policy0.action(time_step) 

  obs = time_step.observation.numpy()[0]
  r, _ = np.shape(obs)
  for i in range(r):
    my_obs = obs[i]
    obs[i] = [my_obs[1], my_obs[0]]
  obs = tf.constant(np.array(obs, dtype=np.float32), shape=(1, 12,2), name="observation")

  step_type = time_step.step_type.numpy()[0]
  step_type = tf.constant(step_type, dtype=tf.int32, shape=(1,), name="step_type")      

  reward = time_step.reward.numpy()[0]
  reward = tf.constant(reward, dtype=tf.float32, shape=(1,), name="reward")    

  discount = time_step.discount.numpy()[0]
  discount = tf.constant(discount, dtype=tf.float32, shape=(1,), name="discount")

  time_step = ts.TimeStep(step_type, reward, discount, obs)
  #print("time step edited", time_step)
  action_step1 = policy1.action(time_step)

  action0 = action_step0.action.numpy()[0]
  action1 = action_step1.action.numpy()

  new_action = tf.convert_to_tensor(np.array([[action0[0], action1[0]]], dtype=np.float32))
  return new_action, action_step0.state, action_step0.info

def construct_intended_action_squashed_shape(policy0, policy1, time_step):
  action_step0 = policy0.action(time_step) 

  obs = time_step.observation.numpy()[0]
  r, _ = np.shape(obs)
  for i in range(r):
    my_obs = obs[i]
    obs[i] = [my_obs[1], my_obs[0]]
  obs = tf.constant(np.array(obs, dtype=np.float32), shape=(12,2), name="observation")

  step_type = time_step.step_type.numpy()[0]
  step_type = tf.constant(step_type, dtype=tf.int32, shape=(), name="step_type")      

  reward = time_step.reward.numpy()[0]
  reward = tf.constant(reward, dtype=tf.float32, shape=(), name="reward")    

  discount = time_step.discount.numpy()[0]
  discount = tf.constant(discount, dtype=tf.float32, shape=(), name="discount")

  time_step = ts.TimeStep(step_type, reward, discount, obs)
  #print("time step edited", time_step)
  action_step1 = policy1.action(time_step)

  action0 = action_step0.action.numpy()[0]
  action1 = action_step1.action.numpy()

  new_action = tf.convert_to_tensor(np.array([[action0[0], action1[0]]], dtype=np.float32))
  return new_action, action_step0.state, action_step0.info

def construct_fixed_action(policy0, action1, time_step):
  action_step0 = policy0.action(time_step) 

  action0 = action_step0.action.numpy()[0]

  new_action = tf.convert_to_tensor(np.array([[action0[0], action1]], dtype=np.float32))
  return new_action