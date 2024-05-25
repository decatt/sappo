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

class Adversary(nn.Module):
    def __init__(self, state_dim, hidden_size, attacker_limit=0.05):
        super(Adversary, self).__init__()
        self.attacker_limit = attacker_limit
        self.action = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim)
        )

        self.critc = nn.Sequential(
            nn.Linear(state_dim*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def get_action(self, state):
        action = self.action(state)
        if len(action.shape) == 1:
            action_min = torch.min(action)
            action_max = torch.max(action)
        else:
            action_min,_ = torch.min(action, dim=1, keepdim=True)
            action_max,_ = torch.max(action, dim=1, keepdim=True)
        action = (action - action_min) / (action_max - action_min) - 0.5
        action = action * self.attacker_limit*2
        return action
    
    def get_q_value(self, state, action):
        q_value = self.critc(torch.cat([state, action], dim=1))
        return q_value
    
    def forward(self, state):
        action = self.get_action(state)
        return action
    

class MadAttacker(nn.Module):
    def __init__(self, state_dim, hidden_size, attacker_limit=0.05):
        super(MadAttacker, self).__init__()
        self.attacker_limit = attacker_limit
        self.action = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim)
        )

    def get_action(self, state):
        action = self.action(state)
        if len(action.shape) == 1:
            action_min = torch.min(action)
            action_max = torch.max(action)
        else:
            action_min,_ = torch.min(action, dim=1, keepdim=True)
            action_max,_ = torch.max(action, dim=1, keepdim=True)
        action = (action - action_min) / (action_max - action_min) - 0.5
        action = action * self.attacker_limit*2
        return action

    def forward(self, state):
        action = self.get_action(state)
        return action
    
class PPOMujocoNet(AlgoBase.AlgoBaseNet):
    def __init__(self,obs_dim,act_dim,hide_dim,use_noise=True):
        super(PPOMujocoNet,self).__init__()
        
        if use_noise:
            self.noise_layer_out = AlgoBase.NoisyLinear(hide_dim,act_dim)
            self.noise_layer_hide = AlgoBase.NoisyLinear(hide_dim,hide_dim)
                            
            #normal mu
            self.mu = nn.Sequential(
                    AlgoBase.layer_init(nn.Linear(obs_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                    nn.ReLU(),
                    self.noise_layer_hide,
                    nn.ReLU(),
                    self.noise_layer_out,
                    nn.Tanh()
                )
        else:
            #normal mu
            self.mu = nn.Sequential(
                    AlgoBase.layer_init(nn.Linear(obs_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                    nn.ReLU(),
                    AlgoBase.layer_init(nn.Linear(hide_dim, act_dim)),
                    nn.Tanh()
                )
                
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.clamp(torch.as_tensor(log_std),-20,2))
        
        self.value = nn.Sequential(
                AlgoBase.layer_init(nn.Linear(obs_dim, hide_dim)),
                nn.ReLU(),
                AlgoBase.layer_init(nn.Linear(hide_dim, hide_dim)),
                nn.ReLU(),
                AlgoBase.layer_init(nn.Linear(hide_dim, 1))
            )
                
    def get_distris(self,states):
        mus = self.mu(states)
        distris = Normal(mus,torch.exp(self.log_std))
        return distris

    def get_value(self,states):
        return self.value(states)
        
    def forward(self,states,actions):
        values = self.value(states)
        distris = self.get_distris(states)
        log_probs = distris.log_prob(actions) 
        return values,log_probs,distris.entropy()
    
    def get_sample_data(self,states):
        distris = self.get_distris(states)
        actions = distris.sample()
        log_probs = distris.log_prob(actions)
        return actions,log_probs
    
    def get_check_data(self,states):
        states = states.type(torch.float32)
        distris = self.get_distris(states)
        mus = self.mu(states)
        log_probs = distris.log_prob(distris.mean)
        return mus,distris.entropy(),log_probs
    
    def get_calculate_data(self,states,actions):
        values = self.value(states)
        distris = self.get_distris(states)
        log_probs = distris.log_prob(actions) 
        return values,log_probs,distris.entropy()
    
    def sample_noise(self):
        self.noise_layer_out.sample_noise()
        self.noise_layer_hide.sample_noise()

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
    env = gym.make('LunarLander-v2')
    net = ActorCriticBox2dNet(8,4,64)
    net.load_state_dict(torch.load('ppo_box2d_LunarLander_48229.pth'))
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