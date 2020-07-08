class PublicGoodsEnv(py_environment.PyEnvironment):
  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(2,), dtype=np.float32, minimum=0, maximum = 1, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(11,2), dtype=np.float32, name='observation')
    s = np.full((11, 2), -1, dtype=np.float32)
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
    s = np.full((11, 2), -1, dtype=np.float32)
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

    s1 = 1
    a1 = self._state[self._counter - 1, 0] * s1
    s2 = 1
    a2 = self._state[self._counter - 1, 1] * s2
    s1_final = s1 - a1 + (a1 + a2)*self._MULT / 2
    s2_final = s2 - a2 + (a1 + a2)*self._MULT / 2

    if self._episode_ended:
      reward_self = s1_final - s1
      reward_other = s2_final - s2
      my_reward = reward_fun(reward_self, reward_other)

      ret = ts.termination(np.array(self._state, dtype=np.float32), my_reward)
      self._counter += 1

      return ret
    else:
      reward_self = s1_final - s1
      reward_other = s2_final - s2
      my_reward = reward_fun(reward_self, reward_other)
      ret = ts.transition(
          np.array(self._state, dtype=np.float32), reward=my_reward, discount=1.0)
      #print("transition:", ret)
      return ret
