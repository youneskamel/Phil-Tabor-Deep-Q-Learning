import cv2
import numpy as np
import gym
from gym.spaces import Box

class RepeatActionAndMaxFrame(gym.Wrapper):
    '''
    Initializer for an OpenAI wrapper used to repeat actions for several frames and
    deal with flickering in the Atari games. This technique is documented in the Atari RL litterature

    args : env : gym.env()
            n_fram_skip() : frames to skip (repeat action for the given value of frames)
    '''
    def __init__(self, env=None, n_frame_skip=4):
        super().__init__()
        self.env = env
        self.n_frame_skip = n_frame_skip
        self.frame_buffer = []
      
    ''' 
    Wraper for env.step() functiion
    We skip 4 frames and take the max from the two last frames
    to counter flickering. Necessary for Atari games

    Args : action
    '''
    def step(self, action):
        total_rewards = 0
        self.done = False 
        for i in range(self.n_frame_skip):
            obs, reward, done, info = self.env.step(action)
            total_rewards += reward
            # on retourne le frame max entre les deux dernier
            # osef des deux premiers, les elements cligonetent toutes les 
            # 2 frames et on veut juste empecher ca
            idx = i % 2
            self.frame_buffer[idx] = obs
            if done :
                break

        max_frame = np.max(self.frame_buffer)
        return max_frame, total_rewards, done, info
    ''' 
    Wraper for env.reset() functiion
    
    Args : none
    '''
    def reset(self):
        obs = self.env.reset()
        self.frame_buffer = []
        self.frame_buffer.append(obs)
        return obs

class PreprocessFrame(gym.ObservationWrapper):

    '''
    Intiliazer function for the frame preprocessing class

    Args : shape : the shape of the frame from atari
            env : gym.env() from openAI
    '''
    def __init__(self, shape, env=None):
        super().__init__(env)
        # Atari has color channels at the end, pytorch needs them at the beginning
        # so we give self.shape the dimmesions of the shape argument but with color channels first
        self.shape = tuple(shape[2], shape[0], shape[1])
        # We use gym.spaces.Box to reshape the obs space
        # self.observation_space is inherited from gym.ObservationWrapper
        self.observation_space = Box(low=0, high=1.0, shape=self.shape,dtype=np.float32)
     
    '''
    Function called by Gym.ObservationWrapper.__init__()
    It puts the frames in grayscale and reshapes them

    Args : obs : the state
    '''
    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Reminder : Python slice notation : x[start_index, end_index, step_size]
        obs = cv2.resize(obs, self.shape[1:], interpolation=cv2.INTER_AREA)
        obs = np.array(obs, dtype=np.unit8).reshape(self.shape) / 255
        return obs
     
''''''
class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, n_frame_skip):


    def reset(self):
     

    def observation(self, observation):
    

def make_env(env_name, shape=(84,84,1), n_frame_skip=4, clip_rewards=False,
             no_ops=0, fire_first=False):
 