import sys,os,time
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
import algo_envs.algo_base as AlgoBase
import gymnasium as gym

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCriticBox2dNet(nn.Module):
    def __init__(self,obs_dim,act_dim,hide_dim):
        super(ActorCriticBox2dNet,self).__init__()
        
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hide_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hide_dim, hide_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hide_dim, act_dim))
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hide_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hide_dim, hide_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hide_dim, 1))
        )

    def get_logits(self,states):
        return self.actor(states)
    
    def get_action(self,states):
        logits = self.actor(states)
        distris = Categorical(logits=logits)
        actions = distris.sample()
        return actions
    
    def get_value(self,states):
        return self.critic(states)
    
    def forward(self,states):
        actions = self.get_action(states)
        value = self.critic(states)
        return actions,value
    
if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    net = ActorCriticBox2dNet(obs_dim=env.observation_space.shape[0],act_dim=env.action_space.n,hide_dim=64)
    net.load_state_dict(torch.load('ppo_box2d_CartPole_215921.pth'))
    state,_ = env.reset()
    for _ in range(1000):
        rs = []
        while True:
            state = torch.as_tensor(state,dtype=torch.float32)
            with torch.no_grad():
                action = net.get_action(state)
            action = action.numpy()
            state, reward_n, is_done, truncated, _ = env.step(action)
            rs.append(reward_n)
            if is_done or truncated:
                print('rewards:',str(sum(rs)))
                state,_ = env.reset()
                break