import sys,os,time
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import torch 
import torch.nn as nn
import gymnasium as gym
import numpy as np
from torch.distributions.normal import Normal
from torch.nn import functional as F
from types import SimpleNamespace
from ppo_mujoco import PPOMujocoNet
from algo_base import calculate_gae
import argparse
import torch.utils.tensorboard as tensorboard
from algo_envs.nets import MadAttacker as Adversary

parser = argparse.ArgumentParser("Adversarial Attack Training")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--num_steps", type=int, default=512)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--env_name", type=str, default="Walker2d")
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
parser.add_argument("--attacker_limit", type=int, default=0.001)
parser.add_argument("--agent_model_path", type=str, default="ppo_mujoco_Walker2d.pth")
parser.add_argument("--eps_steps",type=int,default=100)
args = parser.parse_args()

env_config = {
    'Swimmer':SimpleNamespace(**{'env_name': "Swimmer-v4",'obs_dim':8,'act_dim':2,'hide_dim':64}),
    'HalfCheetah':SimpleNamespace(**{'env_name': "HalfCheetah-v4",'obs_dim':17,'act_dim':6,'hide_dim':64}),
    'Ant':SimpleNamespace(**{'env_name': "Ant-v4",'obs_dim':27,'act_dim':8,'hide_dim':256}),
    'Hopper':SimpleNamespace(**{'env_name': "Hopper-v4",'obs_dim':11,'act_dim':3,'hide_dim':64}),
    'Pusher':SimpleNamespace(**{'env_name': "Pusher-v4",'obs_dim':23,'act_dim':7,'hide_dim':64}),
    'Humanoid':SimpleNamespace(**{'env_name': "Humanoid-v4",'obs_dim':376,'act_dim':17,'hide_dim':512}),
    'Walker2d':SimpleNamespace(**{'env_name': "Walker2d-v4",'obs_dim':17,'act_dim':6,'hide_dim':64}),
}

current_env_name = args.env_name

# Set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

#adversarial attack Q network
   
class AttackAgent:
    def __init__(self, sample_net:PPOMujocoNet, attack_net:Adversary,config_dict,is_checker=False):
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
            self.num_steps = 1024
    
    @torch.no_grad()
    def get_sample_actions(self, state):
        states_v = torch.Tensor(np.array(state))
        state_noise = self.attack_net.get_action(states_v)
        attacked_state = states_v + state_noise
        actions,log_probs = self.sample_net.get_sample_data(attacked_state)
        return actions.cpu().numpy(),log_probs.cpu().numpy(),state_noise.cpu().numpy()
    
    @torch.no_grad()
    def get_check_action(self, state):
        states_v = torch.Tensor(np.array(state))
        #state_noise = self.attack_net.get_action(states_v)
        state_noise = torch.rand_like(states_v) * 2 * args.attacker_limit - args.attacker_limit
        attacked_state = states_v + state_noise
        mu,entropy,log_prob = self.sample_net.get_check_data(attacked_state)
        return mu.cpu().numpy(),entropy.cpu().numpy(),log_prob.cpu().numpy()

    def sample_env(self):
        while len(self.exps_list[0]) < self.num_steps:
            
            actions,log_probs,attacker_noises = self.get_sample_actions(self.states)
            for i in range(self.num_envs):
                next_state_n, reward_n, done_n, _, _ = self.envs[i].step(actions[i])                
                if done_n or len(self.exps_list[i]) >= self.num_steps:
                    next_state_n,_ = self.envs[i].reset()
                    done_n = True
                self.exps_list[i].append([self.states[i],actions[i],reward_n,done_n,log_probs[i],self.config_dict['train_version']])
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

        for _ in range(1000):
            #self.envs.render()
            mu,entropy,log_prob = self.get_check_action(self.states)
            next_state_n, reward_n, is_done, _, _ = self.envs.step(mu)
            if is_done:
                next_state_n,_ = self.envs.reset()
            self.states = next_state_n
            rewards.append(reward_n)
            mus.append(mu)
            entropys.append(entropy)
            log_probs.append(log_prob)
            
            steps += 1
            if is_done:
                break
        
        step_record_dict['sum_rewards'] = np.sum(rewards)
        step_record_dict['mean_entropys'] = np.mean(entropys)
        step_record_dict['mean_mus'] = np.mean(mus)
        step_record_dict['mean_log_probs'] = np.mean(log_probs)
        
        return step_record_dict

class AttackerTraning:
    def __init__(self, attack_net:Adversary, sample_net:PPOMujocoNet, config_dict, calculate_index):
        self.attack_net = attack_net
        self.sample_net = sample_net
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

        self.calculate_net = Adversary(env_config[current_env_name].obs_dim, env_config[current_env_name].hide_dim, args.attacker_limit).to(self.device)
        self.calculate_net.load_state_dict(self.attack_net.state_dict())
        self.calculate_net.to(self.device)
        self.optimizer = torch.optim.Adam(self.sample_net.parameters(), lr=args.lr)
        
        self.states_list = None
        self.actions_list = None
        self.rewards_list = None
        self.dones_list = None
        self.old_log_probs_list = None
        self.advantage_list = None
        self.returns_list = None
        self.noise_list = None
        self.noise_log_probs_list = None

    def begin_batch_train(self, samples_list: list):
        s_states = [np.array([s[0] for s in samples]) for samples in samples_list]
        s_actions = [np.array([s[1] for s in samples]) for samples in samples_list]
        s_log_probs = [np.array([s[4] for s in samples]) for samples in samples_list]
        s_rewards = [np.array([s[2] for s in samples]) for samples in samples_list]
        s_dones = [np.array([s[3] for s in samples]) for samples in samples_list]
        
        self.states = [torch.Tensor(states).to(self.device) for states in s_states]
        self.actions = [torch.Tensor(actions).to(self.device) for actions in s_actions]
        self.old_log_probs = [torch.Tensor(log_probs).to(self.device) for log_probs in s_log_probs]
        self.rewards = s_rewards
        self.dones = s_dones

        self.states_list = torch.cat(self.states)
        self.actions_list = torch.cat(self.actions) 
        self.old_log_probs_list = torch.cat(self.old_log_probs)
    
    def end_batch_train(self):
        self.states_list = None
        self.actions_list = None
        self.rewards_list = None
        self.dones_list = None
        self.old_log_probs_list = None
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

    
    def generate_grads(self): 
        train_version = self.config_dict[self.calculate_index]
        self.calculate_net.load_state_dict(self.attack_net.state_dict())
        self.calculate_net.to(self.device)
        mini_batch_size = args.mini_batch_size
        vf_coef = args.vf_coef
        ent_coef = args.ent_coef
        grad_norm = args.grad_norm
        if args.enable_mini_batch:
            mini_batch_number = self.states_list.shape[0] // mini_batch_size
        else:
            mini_batch_number = 1
            mini_batch_size = self.states_list.shape[0]

        losss = []

        for i in range(mini_batch_number):
            start_index = i*mini_batch_size
            end_index = (i+1)* mini_batch_size
            
            mini_states = self.states_list[start_index:end_index]
            
            self.calculate_net.load_state_dict(self.attack_net.state_dict())

            mini_attacker_noises = self.calculate_net.get_action(mini_states)

            mini_actions = self.actions_list[start_index:end_index]

            mini_log_probs = self.old_log_probs_list[start_index:end_index]

            # find target noise
            max_diffs = torch.zeros(mini_batch_size,1).to(self.device)
            target_noise = torch.zeros_like(mini_attacker_noises)
            for _ in range(args.eps_steps):
                #random noise in the range of [- args.attacker_limit, args.attacker_limit]
                noise = torch.rand_like(mini_attacker_noises) * 2 * args.attacker_limit - args.attacker_limit
                attacked_states = mini_states + noise
                _,mini_attacked_log_probs = self.sample_net.get_sample_data(attacked_states)
                diffs = ((mini_attacked_log_probs - mini_log_probs)**2).sum(axis=-1, keepdim=True)
                for diff_index in range(mini_batch_size):
                    if diffs[diff_index] > max_diffs[diff_index]:
                        max_diffs[diff_index] = diffs[diff_index]
                        target_noise[diff_index] = noise[diff_index]

            # calculate loss
            loss = F.mse_loss(mini_attacker_noises, target_noise)

            losss.append(loss.item())

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
        return np.mean(losss)


if __name__ == "__main__":
    comment = "SAPPO Mujoco Attacker Training"
    writer = tensorboard.SummaryWriter(comment=comment)
    # initialize training network
    ppo_agent_net = PPOMujocoNet(env_config[current_env_name].obs_dim, env_config[current_env_name].act_dim, env_config[current_env_name].hide_dim)
    attacker_net = Adversary(env_config[current_env_name].obs_dim, env_config[current_env_name].hide_dim, args.attacker_limit)
    config_dict = {}
    config_dict[0] = 0
    config_dict['num_trainer'] = 1
    config_dict['train_version'] = 0

    model_path = args.agent_model_path
    if os.path.exists(model_path):
        ppo_agent_net.load_state_dict(torch.load(model_path))
        print("Load agent model from",model_path)
    else:
        print("Can't find agent model from",model_path)

    """for param in ppo_agent_net.parameters():
        param.requires_grad = False"""

    sample_agent = AttackAgent(ppo_agent_net, attacker_net, config_dict)
    check_agent = AttackAgent(ppo_agent_net, attacker_net, config_dict, True)
    trainer = AttackerTraning(attacker_net, ppo_agent_net, config_dict, 0)

    MAX_VERSION = 10
    REPEAT_TIMES = 10

    for _ in range(MAX_VERSION):
        # Sampling training data and calculating time cost
        samples_list = sample_agent.sample_env()
        
        # Calculating policy gradients and time cost
        trainer.begin_batch_train(samples_list)
        loss_list = []
        for _ in range(REPEAT_TIMES):
            loss = trainer.generate_grads()
            loss_list.append(loss)
        print("loss:",np.mean(loss_list))
        trainer.end_batch_train()
        
        # Updating model version
        config_dict[0] = config_dict[0] + 1
        config_dict['train_version'] = config_dict[0]
        
        # Evaluating agent
        infos = check_agent.check_env()
        for k,v in infos.items():
            writer.add_scalar(k,v,config_dict[0])
            
        print("version:",config_dict[0],"sum_rewards:",infos['sum_rewards'])

model_path = "saved_model\\adversarial_attack_mad_01_"+args.env_name+".pth"
torch.save(attacker_net.state_dict(), model_path)
