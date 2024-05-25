import sys,os,time
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import torch 
import torch.nn as nn
import gymnasium as gym
import numpy as np
from torch.distributions.categorical import Categorical
from torch.nn import functional as F
from types import SimpleNamespace
import algo_envs.algo_base as AlgoBase
import argparse
import yaml
from algo_envs.nets import ActorCriticBox2dNet
import wandb
import random


parser = argparse.ArgumentParser("PPO Box2d Normal Share GAE")
parser.add_argument("--env_name",type=str,default="Acrobot")
parser.add_argument("--gae_lambda",type=float,default=0.98)
parser.add_argument("--gamma",type=float,default=0.99)
parser.add_argument("--clip_coef",type=float,default=0.2)
parser.add_argument("--max_clip_coef",type=float,default=4)
parser.add_argument("--vf_coef",type=float,default=4)
parser.add_argument("--ent_coef",type=float,default=0.01)
parser.add_argument("--max_version",type=int,default=int(1e6))
parser.add_argument("--learning_rate",type=float,default=1e-4)
parser.add_argument("--ratio_coef",type=float,default=0.5)
parser.add_argument("--grad_norm",type=float,default=0.5)
parser.add_argument("--pg_loss_type",type=int,default=1)
parser.add_argument("--enable_clip_max",type=bool,default=True)
parser.add_argument("--enable_ratio_decay",type=bool,default=False)
parser.add_argument("--enable_entropy_decay",type=bool,default=False)
parser.add_argument("--enable_lr_decay",type=bool,default=False)
parser.add_argument("--enable_grad_norm",type=bool,default=False)
parser.add_argument("--enable_adv_norm",type=bool,default=True)
parser.add_argument("--enable_mini_batch",type=bool,default=True)
parser.add_argument("--pae_length",type=int,default=256)
parser.add_argument("--num_envs",type=int,default=32)
parser.add_argument("--num_steps",type=int,default=512)
parser.add_argument("--use_gpu",type=bool,default=False)
parser.add_argument("--mini_batch_size",type=int,default=128)
args = parser.parse_args()

#path = 'algo_envs\configs\ppo_mujoco.yaml'
path = None
if path is not None:
    #read yaml file fot hyperparameters
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in config.items():
            setattr(args, k, v)

train_envs = {
    'LunarLander':SimpleNamespace(**{'env_name': "LunarLander-v2",'obs_dim':8,'act_dim':4,'hide_dim':64}),
    'BipedalWalker':SimpleNamespace(**{'env_name': "BipedalWalker-v3",'obs_dim':24,'act_dim':4,'hide_dim':64}),
    'CartPole':SimpleNamespace(**{'env_name': "CartPole-v1",'obs_dim':4,'act_dim':2,'hide_dim':64}),
    'MountainCar':SimpleNamespace(**{'env_name': "MountainCar-v0",'obs_dim':2,'act_dim':3,'hide_dim':64}),
    'Acrobot':SimpleNamespace(**{'env_name': "Acrobot-v1",'obs_dim':6,'act_dim':3,'hide_dim':64}),
}

# current environment name
current_env_name = args.env_name

class PPOBox2dAgent(AlgoBase.AlgoBaseAgent):
    def __init__(self,sample_net:ActorCriticBox2dNet,model_dict,is_checker):
        super(PPOBox2dAgent,self).__init__()
        self.sample_net = sample_net
        self.model_dict = model_dict
        self.num_steps = args.num_steps
        self.num_envs = args.num_envs
        self.pae_length = args.pae_length
        self.rewards = []
        
        env_name = train_envs[current_env_name].env_name
    
        if not is_checker:
            self.envs = gym.vector.SyncVectorEnv([lambda:gym.make(env_name) for _ in range(self.num_envs)])
            self.states,_ = self.envs.reset()
            self.exps_list = [[] for _ in range(self.num_envs)]
        else:
            print("PPOBox2dNormalShare check mujoco env is",env_name)
            self.envs = gym.make(env_name)
            self.states,_ = self.envs.reset()
            self.num_steps = 1024
        
    def sample_env(self):
        while len(self.exps_list[0]) < self.num_steps:
            
            actions,log_probs = self.get_sample_actions(self.states)
            next_states_n, rewards_n, dones_n, truncateds_n, _ = self.envs.step(actions)
            for i in range(self.num_envs):
                next_state_n = next_states_n[i]
                reward_n = rewards_n[i]
                done_n = dones_n[i]
                truncated_n = truncateds_n[i]
                if done_n or truncated_n or len(self.exps_list[i]) >= self.num_steps:
                    done_n = True
                    next_states_n[i],_ = self.envs.envs[i].reset()
                    
                self.exps_list[i].append([self.states[i],actions[i],reward_n,done_n,log_probs[i],self.model_dict['train_version']])
            self.states = next_states_n
        # Starting training
        train_exps = self.exps_list
        # Deleting the length before self.pae_length
        self.exps_list = [ exps[self.pae_length:self.num_steps] for exps in self.exps_list ]
        return train_exps
    
    def check_env(self):
        step_record_dict = dict()
        
        is_done = False
        steps = 0
        actions = []
        rewards = []
        entropys = []
        log_probs = []

        while True:
            #self.envs.render()
            action,entropy,log_prob = self.get_check_action(self.states)
            next_state_n, reward_n, is_done, truncated, _ = self.envs.step(action)
            if is_done or truncated:
                next_state_n,_ = self.envs.reset()
            self.states = next_state_n
            rewards.append(reward_n)
            actions.append(action)
            entropys.append(entropy)
            log_probs.append(log_prob)
            
            steps += 1
            if is_done or truncated:
                break
            #if steps >= self.num_steps:
            #    break
        
        step_record_dict['sum_rewards'] = np.sum(rewards)
        step_record_dict['mean_entropys'] = np.mean(entropys)
        step_record_dict['mean_actions'] = np.mean(actions)
        step_record_dict['mean_log_probs'] = np.mean(log_probs)
        
        return step_record_dict
            
    @torch.no_grad()
    def get_sample_actions(self,states):
        states_v = torch.Tensor(np.array(states))
        logits = self.sample_net.get_logits(states_v)
        distris = Categorical(logits=logits)
        actions = distris.sample()
        log_probs = distris.log_prob(actions)
        return actions.cpu().numpy(),log_probs.cpu().numpy()
    
    @torch.no_grad()
    def get_check_action(self,state):
        state_v = torch.Tensor(np.array(state))
        logits = self.sample_net.get_logits(state_v)
        distris = Categorical(logits=logits)
        action = distris.sample()
        log_prob = distris.log_prob(action)
        entropy = distris.entropy()
        return action.cpu().numpy(),entropy.cpu().numpy(),log_prob.cpu().numpy()
            
class PPOMujocoCalculate(AlgoBase.AlgoBaseCalculate):
    def __init__(self,share_model:ActorCriticBox2dNet,model_dict,calculate_index):
        super(PPOMujocoCalculate,self).__init__()
        self.model_dict = model_dict
        self.share_model = share_model
        self.pae_length = args.pae_length
        self.calculate_number = self.model_dict['num_trainer']
        self.calculate_index = calculate_index
        self.train_version = 0
        
        if args.use_gpu and torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_index = self.calculate_index % device_count
            self.device = torch.device('cuda',device_index)
        else:
            self.device = torch.device('cpu')
                        
        self.calculate_net = ActorCriticBox2dNet(train_envs[current_env_name].obs_dim,train_envs[current_env_name].act_dim,train_envs[current_env_name].hide_dim)
        self.calculate_net.to(self.device)
        #self.calculate_net.load_state_dict(self.share_model.state_dict())
    
        self.share_optim = torch.optim.Adam(params=self.share_model.parameters(), lr=args.learning_rate)
        
        # Clear training data
        self.states_list = None
        self.actions_list = None
        self.rewards_list = None
        self.dones_list = None
        self.old_log_probs_list = None
        self.advantage_list = None
        self.returns_list = None
        
    def begin_batch_train(self, samples_list: list): 
        """store training data
        Example:
            >>> train_net = PPOMujocoNormalShareGAENet()
            >>> calculate = PPOMujocoNormalShareGAECalculate(train_net,model_dict,calculate_index)
            >>> calculate.begin_batch_train(samples)
        """  

        s_states = [np.array([s[0] for s in samples]) for samples in samples_list]
        s_actions = [np.array([s[1] for s in samples]) for samples in samples_list]
        s_log_probs = [np.array([s[4] for s in samples]) for samples in samples_list]
        
        s_rewards = [np.array([s[2] for s in samples]) for samples in samples_list]
        s_dones = [np.array([s[3] for s in samples]) for samples in samples_list]
        
        #s_versions = [s[6] for s in samples]
        
        self.states = [torch.Tensor(states).to(self.device) for states in s_states]
        self.actions = [torch.Tensor(actions).to(self.device) for actions in s_actions]
        self.old_log_probs = [torch.Tensor(log_probs).to(self.device) for log_probs in s_log_probs]
        self.rewards = s_rewards
        self.dones = s_dones
        
        self.states_list = torch.cat([states[0:self.pae_length] for states in self.states])
        self.actions_list = torch.cat([actions[0:self.pae_length] for actions in self.actions])
        self.old_log_probs_list = torch.cat([old_log_probs[0:self.pae_length] for old_log_probs in self.old_log_probs])
        
    def end_batch_train(self):
        """clear training data, update learning rate and noisy network
        Example:
            >>> train_net = PPOMujocoNormalShareGAENet()
            >>> calculate = PPOMujocoNormalShareGAECalculate(train_net,model_dict,calculate_index)
            >>> calculate.end_batch_train(samples)
        """
        
        self.states_list = None
        self.actions_list = None
        self.rewards_list = None
        self.dones_list = None
        self.old_log_probs_list = None
        self.advantage_list = None
        self.returns_list = None
        
        train_version = self.model_dict[self.calculate_index]
        self.decay_lr(train_version)
        

    def calculate_samples_gae(self):
        """Calculate samples generalized advantage estimator, more details see in algo_base.calculate_gae
        """
        gamma = args.gamma
        gae_lambda = args.gae_lambda
        
        np_advantages = []
        np_returns = []
        
        for states,rewards,dones in zip(self.states,self.rewards,self.dones):
            with torch.no_grad():
                values = self.calculate_net.get_value(states)
                            
            advantages,returns = AlgoBase.calculate_gae(values.cpu().numpy().reshape(-1),rewards,dones,gamma,gae_lambda)
            np_advantages.extend(advantages[0:self.pae_length])
            np_returns.extend(returns[0:self.pae_length])
            
        return np.array(np_advantages), np.array(np_returns)
            
    def decay_lr(self, version):
        if args.enable_lr_decay:
            lr_now = args.learning_rate * (1 - version*1.0 / args.max_version)
            if lr_now <= 1e-6:
                lr_now = 1e-6
            
            if self.share_optim is not None:
                for param in self.share_optim.param_groups:
                    param['lr'] = lr_now
                                                                                   
    def generate_grads(self): 
        train_version = self.model_dict[self.calculate_index]
        vf_coef = args.vf_coef
        pg_loss_type = args.pg_loss_type
        grad_norm = args.grad_norm
        mini_batch_size = args.mini_batch_size

        ent_coef = args.ent_coef
        ratio_coef = self.get_ratio_coef(train_version)
    
        self.calculate_net.load_state_dict(self.share_model.state_dict())
                        
        #start = timer()
        np_advantages,np_returns = self.calculate_samples_gae()
        
        #run_time = timer() - start
        #print("CPU function took %f seconds." % run_time)
        
        if args.enable_adv_norm:
            np_advantages = (np_advantages - np_advantages.mean()) / (np_advantages.std() + 1e-8)
                                                    
        advantage_list = torch.Tensor(np_advantages.reshape(-1,1)).to(self.device)    
        returns_list = torch.Tensor(np_returns.reshape(-1,1)).to(self.device)
        
        if args.enable_mini_batch:
            mini_batch_number = advantage_list.shape[0] // mini_batch_size
        else:
            mini_batch_number = 1
            mini_batch_size = advantage_list.shape[0]

        for i in range(mini_batch_number):
            start_index = i*mini_batch_size
            end_index = (i+1)* mini_batch_size
            
            mini_states = self.states_list[start_index:end_index]
            mini_actions = self.actions_list[start_index:end_index]
            mini_old_log_probs = self.old_log_probs_list[start_index:end_index]
            
            self.calculate_net.load_state_dict(self.share_model.state_dict())
                
            mini_new_values = self.calculate_net.get_value(mini_states)
            mini_new_logits = self.calculate_net.get_logits(mini_states)
            mini_new_distris = Categorical(logits=mini_new_logits)
            mini_new_log_probs = mini_new_distris.log_prob(mini_actions)
            mini_entropys = mini_new_distris.entropy()
            
            mini_advantage = advantage_list[start_index:end_index]
            mini_returns = returns_list[start_index:end_index]

            ratio = torch.exp(mini_new_log_probs-mini_old_log_probs)

            pg_loss = self.get_pg_loss(ratio,mini_advantage)
                
            # Policy loss
            pg_loss = -torch.mean(pg_loss)
            
            v_loss = F.mse_loss(mini_returns, mini_new_values) * vf_coef
            
            e_loss = -torch.mean(mini_entropys) * ent_coef
            
            loss = pg_loss + v_loss + e_loss
            
            self.calculate_net.zero_grad()

            loss.backward()
            
            grads = [
                param.grad.data.cpu().numpy()
                if param.grad is not None else None
                for param in self.calculate_net.parameters()
            ]
            
            # Updating network parameters
            for param, grad in zip(self.share_model.parameters(), grads):
                param.grad = torch.FloatTensor(grad)

            if args.enable_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.share_model.parameters(),grad_norm)  
            self.share_optim.step()
    
    def get_pg_loss(self,ratio,advantage):
        clip_coef = args.clip_coef
        max_clip_coef = args.max_clip_coef
        enable_clip_max =  args.enable_clip_max
        
        positive = torch.where(ratio >= 1.0 + clip_coef, 0 * advantage,advantage)
        if enable_clip_max:
            negtive = torch.where(ratio <= 1.0 - clip_coef,0 * advantage,torch.where(ratio >= max_clip_coef, 0 * advantage,advantage))
        else:
            negtive = torch.where(ratio <= 1.0 - clip_coef,0 * advantage,advantage)
        
        return torch.where(advantage>=0,positive,negtive)*ratio
    
    def get_ent_coef(self,version):
        if args.enable_entropy_decay:
            ent_coef = args.ent_coef * (1 - version*1.0 / args.max_version)
            if ent_coef <= 1e-8:
                ent_coef = 1e-8
            return ent_coef
        else:
            return args.ent_coef

    def get_ratio_coef(self,version):
        """increase ratio from 0 to 0.95 in mixed environment"""
        if args.enable_ratio_decay:
            ratio_coef = version/args.max_version
            if ratio_coef >= 1.0:
                ratio_coef = 0.95       
            return ratio_coef   
        
        else:
            return args.ratio_coef
        
if __name__ == "__main__":
    # initialize training network
    train_net = ActorCriticBox2dNet(train_envs[current_env_name].obs_dim,train_envs[current_env_name].act_dim,train_envs[current_env_name].hide_dim)
    model_dict = {}
    model_dict[0] = 0
    model_dict['num_trainer'] = 1
    model_dict['train_version'] = 0

    # initialize a RL agent, smaple agent used for sampling training data, check agent used for evaluating 
    # and calculate used for calculating gradients
    sample_agent = PPOBox2dAgent(train_net,model_dict,is_checker=False)
    check_agent = PPOBox2dAgent(train_net,model_dict,is_checker=True)
    calculate = PPOMujocoCalculate(train_net,model_dict,0)

    # hyperparameters
    MAX_VERSION = 200
    REPEAT_TIMES = 10

    seed = random.randint(0,1000000)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    for _ in range(MAX_VERSION):
        # Sampling training data and calculating time cost
        start_time = time.time()
        samples_list = sample_agent.sample_env()
        end_time = time.time()-start_time
        print('sample_time:',end_time)
        samples = []
        
        for s in samples_list:
            samples.append(s)
        # Calculating policy gradients and time cost
        start_time = time.time()
        calculate.begin_batch_train(samples)
        for _ in range(REPEAT_TIMES):
            calculate.generate_grads()
        calculate.end_batch_train()
        end_time = time.time()-start_time                    
        print('calculate_time:',end_time)
        
        # Updating model version
        model_dict[0] = model_dict[0] + 1
        model_dict['train_version'] = model_dict[0]
        
        # Evaluating agent
        infos = check_agent.check_env()
            
        sum_rewards = infos['sum_rewards']
        print("version:",model_dict[0],"sum_rewards:",sum_rewards)

    torch.save(train_net.state_dict(), 'ppo_box2d_'+current_env_name+'_'+str(seed)+'.pth')

    

    
