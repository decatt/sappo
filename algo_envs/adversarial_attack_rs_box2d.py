import sys,os,time
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import torch 
import torch.nn as nn
import gymnasium as gym
import numpy as np
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from types import SimpleNamespace
import argparse
import torch.utils.tensorboard as tensorboard
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from auto_LiRPA.perturbations import *
from algo_envs.nets import Adversary, ActorCriticBox2dNet

import random

parser = argparse.ArgumentParser("Adversarial Attack Training")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--num_steps", type=int, default=512)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--env_name", type=str, default="Acrobot")
parser.add_argument("--use_gpu", type=bool, default=False)
parser.add_argument("--enable_lr_decay", type=bool, default=False)
parser.add_argument("--max_version", type=int, default=1000000)
parser.add_argument("--enable_mini_batch", type=bool, default=False)
parser.add_argument("--mini_batch_size", type=int, default=32)
parser.add_argument("--enable_adv_norm", type=bool, default=True)
parser.add_argument("--clip_coef", type=float, default=0.2)
parser.add_argument("--max_clip_coef", type=float, default=4)
parser.add_argument("--enable_clip_max", type=bool, default=False) 
parser.add_argument("--vf_coef", type=float, default=0.5)
parser.add_argument("--ent_coef", type=float, default=0.01)
parser.add_argument("--grad_norm", type=float, default=0.5)
parser.add_argument("--enable_grad_norm", type=bool, default=True)
parser.add_argument("--num_trainer", type=int, default=1)
parser.add_argument("--attacker_limit", type=int, default=0.05)
parser.add_argument("--agent_model_path", type=str, default="ppo_box2d_Acrobot_384336.pth")
parser.add_argument("--eps_steps",type=int,default=100)
args = parser.parse_args()

env_config = {
    'LunarLander':SimpleNamespace(**{'env_name': "LunarLander-v2",'obs_dim':8,'act_dim':4,'hide_dim':64}),
    'CartPole':SimpleNamespace(**{'env_name': "CartPole-v1",'obs_dim':4,'act_dim':2,'hide_dim':64}),
    'Acrobot':SimpleNamespace(**{'env_name': "Acrobot-v1",'obs_dim':6,'act_dim':3,'hide_dim':64}),
}

current_env_name = args.env_name

# Set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

    
class AttackAgent:
    def __init__(self, sample_net:ActorCriticBox2dNet, attack_net:Adversary,config_dict,is_checker=False):
        self.sample_net = sample_net
        self.attack_net = attack_net
        self.config_dict = config_dict
        self.num_steps = args.num_steps
        self.num_envs = args.num_envs

        env_name = env_config[current_env_name].env_name
    
        if not is_checker:
            self.envs = [gym.make(env_name) for _ in range(self.num_envs)]
            self.states = [self.envs[i].reset()[0] for i in range(self.num_envs)]
            self.exps_list = [[] for _ in range(self.num_envs)]
        else:
            print("PPOMujocoNormalShare check mujoco env is",env_name)
            self.envs = gym.make(env_name)
            self.states,_ = self.envs.reset()
            self.states = [self.states]
            self.num_steps = 1024
    
    @torch.no_grad()
    def get_sample_actions(self, state):
        states_v = torch.Tensor(np.array(state))
        state_noise = self.attack_net.get_action(states_v)
        attacked_state = states_v + state_noise
        logits = self.sample_net.get_logits(attacked_state)
        distris = Categorical(logits=logits)
        actions = distris.sample()
        log_probs = distris.log_prob(actions)
        return actions.cpu().numpy(), state_noise.cpu().numpy(), log_probs.cpu().numpy()
    
    @torch.no_grad()
    def get_check_action(self, state):
        states_v = torch.Tensor(np.array(state))
        state_noise = self.attack_net.get_action(states_v)
        attacked_state = states_v + state_noise
        logits = self.sample_net.get_logits(attacked_state)
        distris = Categorical(logits=logits)
        #get max probability action
        mu = distris.probs.argmax(dim=-1)
        entropy = distris.entropy()
        log_prob = distris.log_prob(mu)
        return mu.cpu().numpy()[0],entropy.cpu().numpy()[0],log_prob.cpu().numpy()[0]

    def sample_env(self):
        while len(self.exps_list[0]) < self.num_steps:
            
            actions,attacker_noises,log_probs = self.get_sample_actions(self.states)
            for i in range(self.num_envs):
                next_state_n, reward_n, done_n, truncated_n, _ = self.envs[i].step(actions[i])
                reward_n = -1*reward_n           
                if done_n or truncated_n or len(self.exps_list[i]) >= self.num_steps:
                    next_state_n,_ = self.envs[i].reset()
                    done_n = True
                self.exps_list[i].append([self.states[i],attacker_noises[i],reward_n,done_n,self.config_dict['train_version']])
                self.states[i] = next_state_n

        # Starting training
        train_exps = self.exps_list
        # Deleting the length before self.pae_length
        self.exps_list = [[] for _ in range(self.num_envs)]
        samples = []
        for i in range(self.num_envs):
            samples.append(train_exps[i])
        return samples
    
    def check_env(self):
        step_record_dict = dict()
        is_done = False
        steps = 0
        mus = []
        rewards = []
        entropys = []
        log_probs = []

        while True:
            #self.envs.render()
            mu,entropy,log_prob = self.get_check_action(self.states)
            next_state_n, reward_n, is_done, is_truncated, _ = self.envs.step(mu)
            if is_done:
                next_state_n,_ = self.envs.reset()
            self.states = [next_state_n]
            rewards.append(reward_n)
            mus.append(mu)
            entropys.append(entropy)
            log_probs.append(log_prob)
            
            steps += 1
            if is_done or is_truncated:
                break
        
        step_record_dict['sum_rewards'] = np.sum(rewards)
        step_record_dict['mean_entropys'] = np.mean(entropys)
        step_record_dict['mean_mus'] = np.mean(mus)
        step_record_dict['mean_log_probs'] = np.mean(log_probs)
        
        return step_record_dict

class AttackerTraning:
    def __init__(self, attack_net:Adversary, config_dict, calculate_index):
        self.attack_net = attack_net
        self.config_dict = config_dict
        self.calculate_number = self.config_dict['num_trainer']
        self.calculate_index = calculate_index
        self.train_version = 0

        if args.use_gpu and torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_index = self.calculate_index % device_count
            self.device = torch.device('cuda',device_index)
        else:
            self.device = torch.device('cpu')

        hidden_size = env_config[current_env_name].hide_dim
        state_dim = env_config[current_env_name].obs_dim
        self.calculate_net = Adversary(state_dim,hidden_size).to(self.device)
        self.calculate_net.load_state_dict(self.attack_net.state_dict())
        self.calculate_net.to(self.device)
        self.optimizer = torch.optim.Adam(self.attack_net.parameters(), lr=args.lr)
        
        self.states_list = None
        self.actions_list = None
        self.rewards_list = None
        self.dones_list = None
        self.advantage_list = None
        self.returns_list = None
        self.noise_list = None

    def begin_batch_train(self, samples_list: list):
        s_states = [np.array([s[0] for s in samples]) for samples in samples_list]
        s_actions = [np.array([s[1] for s in samples]) for samples in samples_list]
        s_rewards = [np.array([s[2] for s in samples]) for samples in samples_list]
        s_dones = [np.array([s[3] for s in samples]) for samples in samples_list]
        
        self.states = [torch.Tensor(states).to(self.device) for states in s_states]
        self.actions = [torch.Tensor(actions).to(self.device) for actions in s_actions]
        self.rewards = s_rewards
        self.dones = s_dones

        self.states_list = torch.cat(self.states)
        self.actions_list = torch.cat(self.actions) 
        self.rewards_list = torch.cat([torch.Tensor(rewards).to(self.device) for rewards in s_rewards])
        self.dones_list = torch.cat([torch.Tensor(dones).to(self.device) for dones in s_dones])
    
    def end_batch_train(self):
        self.states_list = None
        self.actions_list = None
        self.rewards_list = None
        self.dones_list = None
        self.advantage_list = None
        self.returns_list = None

        train_version = self.config_dict[self.calculate_index]
        self.decay_lr(train_version)

    def decay_lr(self, version):
        if args.enable_lr_decay:
            lr_now = args.lr * (1 - version*1.0 / args.max_version)
            if lr_now <= 1e-6:
                lr_now = 1e-6
            
            if self.optimizer is not None:
                for param in self.optimizer.param_groups:
                    param['lr'] = lr_now
    
    def compute_loss(self, states, actions, rewards, dones):
        lambda_RS = 0.1
        gamma = args.gamma

        next_states = []
        next_actions = []

        for i in range(len(actions)):
            if dones[i]:
                next_states.append(states[i])
                next_actions.append(actions[i])
            elif i < len(actions)-1:
                next_states.append(states[i+1])
                next_actions.append(actions[i+1])
            else:
                next_states.append(states[i])
                next_actions.append(actions[i])

        next_states = torch.stack(next_states).to(self.device)
        next_actions = torch.stack(next_actions).to(self.device)

        target_Q_values = rewards + gamma * self.calculate_net.get_q_value(next_states, next_actions)
        current_Q_values = self.calculate_net.get_q_value(states, actions)
        TD_error = torch.pow(target_Q_values - current_Q_values, 2)

        steps = args.eps_steps
        #batch_action_means = actions.detach()
        batch_q_means = self.calculate_net.get_q_value(states, actions).detach()
        # upper and lower bounds for clipping
        states_ub = states + args.attacker_limit
        states_lb = states - args.attacker_limit
        step_eps = args.attacker_limit / steps
        # SGLD noise factor. We set (inverse) beta=1e-5 as gradients are relatively small here.
        beta = 1e-5
        noise_factor = np.sqrt(2 * step_eps * beta)
        noise = torch.randn_like(states) * noise_factor
        var_states = (states.clone() + noise.sign() * step_eps).detach().requires_grad_()
        """for i in range(steps):
            # Find a nearby state new_phi that maximize the difference
            diff = (warpped_net.get_action(var_states) - batch_action_means)
            kl = (diff * diff).sum(axis=-1, keepdim=True).mean()
            # Need to clear gradients before the backward() for policy_loss
            kl.backward(retain_graph=True)
            # Reduce noise at every step.
            noise_factor = np.sqrt(2 * step_eps * beta) / (i+2)
            # Project noisy gradient to step boundary.
            update = (var_states.grad + noise_factor * torch.randn_like(var_states)).sign() * step_eps
            var_states.data += update
            # clip into the upper and lower bounds
            var_states = torch.max(var_states, states_lb)
            var_states = torch.min(var_states, states_ub)
            var_states = var_states.detach().requires_grad_()
        #self.calculate_net.zero_grad()
        diff = (warpped_net.get_action(var_states.requires_grad_(False)) - batch_action_means)"""
        for i in range(steps):
            actions_ = self.calculate_net.get_action(var_states)
            diff = (self.calculate_net.get_q_value(var_states, actions_) - batch_q_means)
            kl = (diff * diff).sum(axis=-1, keepdim=True).mean()
            kl.backward(retain_graph=True)
            noise_factor = np.sqrt(2 * step_eps * beta) / (i+2)
            update = (var_states.grad + noise_factor * torch.randn_like(var_states)).sign() * step_eps
            var_states.data += update
            var_states = torch.max(var_states, states_lb)
            var_states = torch.min(var_states, states_ub)
            var_states = var_states.detach().requires_grad_()
        actions_ = self.calculate_net.get_action(var_states.requires_grad_(False))
        diff = (self.calculate_net.get_q_value(var_states.requires_grad_(False), actions_) - batch_q_means)
        loss_RS  = (diff * diff).sum(axis=-1, keepdim=True).mean()

        loss_TD = TD_error.mean()

        total_loss = loss_TD + loss_RS*lambda_RS
        return total_loss
    

    def generate_grads(self): 
        self.calculate_net.load_state_dict(self.attack_net.state_dict())
        self.calculate_net.to(self.device)
        mini_batch_size = args.mini_batch_size
        grad_norm = args.grad_norm
        if args.enable_mini_batch:
            mini_batch_number = self.states_list.shape[0] // mini_batch_size
        else:
            mini_batch_number = 1
            mini_batch_size = self.states_list.shape[0]

        for i in range(mini_batch_number):
            start_index = i*mini_batch_size
            end_index = (i+1)* mini_batch_size
            
            mini_states = self.states_list[start_index:end_index]
            mini_actions = self.actions_list[start_index:end_index]
            mini_rewards = self.rewards_list[start_index:end_index]
            mini_dones = self.dones_list[start_index:end_index]
            
            # Calculate the loss
            loss = self.compute_loss(mini_states, mini_actions, mini_rewards, mini_dones)

            self.optimizer.zero_grad()
            loss.backward()

            grads = [
                param.grad.data.cpu().numpy()
                if param.grad is not None else None
                for param in self.calculate_net.parameters()
            ]
            
            # Updating network parameters
            for param, grad in zip(self.attack_net.parameters(), grads):
                if grad is not None:
                    param.grad = torch.Tensor(grad).to(self.device)

            if args.enable_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.attack_net.parameters(), grad_norm)

            self.optimizer.step()

if __name__ == "__main__":
    attacker_limits = [0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3]
    for attacker_limit in attacker_limits:
        args.attacker_limit = attacker_limit
        for _ in range(1):
            print("Start Training")
            comment = "SAPPO Mujoco Attacker Training"
            seed = random.randint(0,100000)
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            writer = tensorboard.SummaryWriter(comment=comment)
            # initialize training network
            ppo_agent_net = ActorCriticBox2dNet(env_config[current_env_name].obs_dim, env_config[current_env_name].act_dim, env_config[current_env_name].hide_dim)
            state_dim = env_config[current_env_name].obs_dim
            hidden_size = env_config[current_env_name].hide_dim
            attacker_net = Adversary(state_dim, hidden_size)
            config_dict = {}
            config_dict[0] = 0
            config_dict['num_trainer'] = 1
            config_dict['train_version'] = 0

            attacker_net = Adversary(state_dim, hidden_size, args.attacker_limit)
            model_path = args.agent_model_path
            if os.path.exists(model_path):
                ppo_agent_net.load_state_dict(torch.load(model_path))
                print("Load agent model from",model_path)
            else:
                print("Can't find agent model from",model_path)

            sample_agent = AttackAgent(ppo_agent_net, attacker_net, config_dict)
            check_agent = AttackAgent(ppo_agent_net, attacker_net, config_dict, True)
            trainer = AttackerTraning(attacker_net, config_dict, 0)

            MAX_VERSION = 50
            REPEAT_TIMES = 10
        
            """check_rewards = []
            for check_time in range(1000):
                infos = check_agent.check_env()
                check_rewards.append(infos['sum_rewards'])
            print("mean_check_rewards:", np.mean(check_rewards))
        """
            for _ in range(MAX_VERSION):
                # Sampling training data and calculating time cost
                start_time = time.time()
                samples_list = sample_agent.sample_env()
                end_time = time.time()-start_time
                print('sample_time:',end_time)
                
                # Calculating policy gradients and time cost
                start_time = time.time()
                trainer.begin_batch_train(samples_list)
                for _ in range(REPEAT_TIMES):
                    trainer.generate_grads()
                trainer.end_batch_train()
                end_time = time.time()-start_time
                print('calculate_time:',end_time)
                
                # Updating model version
                config_dict[0] = config_dict[0] + 1
                config_dict['train_version'] = config_dict[0]
                
                # Evaluating agent
                infos = check_agent.check_env()
                for k,v in infos.items():
                    writer.add_scalar(k,v,config_dict[0])
                    
                print("version:",config_dict[0],"sum_rewards:",infos['sum_rewards'])

            """check_rewards = []
            for check_time in range(1000):
                infos = check_agent.check_env()
                check_rewards.append(infos['sum_rewards'])
            print("mean_check_rewards:", np.mean(check_rewards))"""

            model_path = "saved_model\\"+args.env_name+"\\"+str(args.attacker_limit)+"\\adversarial_attack_rs_seed_"+str(seed)+".pth"
            torch.save(attacker_net.state_dict(), model_path)
