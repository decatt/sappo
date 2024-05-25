import sys,os,time
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import torch 
import torch.nn as nn
import gymnasium as gym
import numpy as np
from torch.distributions.normal import Normal
from torch.nn import functional as F
from types import SimpleNamespace
import algo_envs.algo_base as AlgoBase
import argparse
import yaml

#torch.tensorboard
import torch.utils.tensorboard as tensorboard

parser = argparse.ArgumentParser("SAPPO Mujoco Training")
parser.add_argument("--env_name",type=str,default="Hopper")
parser.add_argument("--gae_lambda",type=float,default=0.95)
parser.add_argument("--gamma",type=float,default=0.99)
parser.add_argument("--clip_coef",type=float,default=0.2)
parser.add_argument("--max_clip_coef",type=float,default=4)
parser.add_argument("--vf_coef",type=float,default=4)
parser.add_argument("--ent_coef",type=float,default=0.01)
parser.add_argument("--max_version",type=int,default=int(1e6))
parser.add_argument("--use_noise",type=bool,default=True)
parser.add_argument("--learning_rate",type=float,default=2.5e-4)
parser.add_argument("--ratio_coef",type=float,default=0.5)
parser.add_argument("--grad_norm",type=float,default=0.5)
parser.add_argument("--pg_loss_type",type=int,default=2)
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
parser.add_argument("--adpr_coef",type=float,default=1)
parser.add_argument("--attacker_noise_limit",type=float,default=0.05)
parser.add_argument("--eps_steps",type=int,default=10)

args = parser.parse_args()

path = None
if path is not None:
    #read yaml file fot hyperparameters
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in config.items():
            setattr(args, k, v)

env_config = {
    'Swimmer':SimpleNamespace(**{'env_name': "Swimmer-v4",'obs_dim':8,'act_dim':2,'hide_dim':64}),
    'HalfCheetah':SimpleNamespace(**{'env_name': "HalfCheetah-v4",'obs_dim':17,'act_dim':6,'hide_dim':64}),
    'Ant':SimpleNamespace(**{'env_name': "Ant-v4",'obs_dim':27,'act_dim':8,'hide_dim':256}),
    'Hopper':SimpleNamespace(**{'env_name': "Hopper-v4",'obs_dim':11,'act_dim':3,'hide_dim':64}),
    'Pusher':SimpleNamespace(**{'env_name': "Pusher-v4",'obs_dim':23,'act_dim':7,'hide_dim':64}),
    'Humanoid':SimpleNamespace(**{'env_name': "Humanoid-v4",'obs_dim':376,'act_dim':17,'hide_dim':512}),
    'Walker2d':SimpleNamespace(**{'env_name': "Walker2d-v4",'obs_dim':17,'act_dim':6,'hide_dim':64}),
}

# current environment name
current_env_name = args.env_name

class PPOMujocoNet(AlgoBase.AlgoBaseNet):
    def __init__(self):
        super(PPOMujocoNet,self).__init__()
        
        obs_dim = env_config[current_env_name].obs_dim
        act_dim = env_config[current_env_name].act_dim
        hide_dim = env_config[current_env_name].hide_dim
        
        if args.use_noise:
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
        if args.use_noise:
            self.noise_layer_out.sample_noise()
            self.noise_layer_hide.sample_noise()
    
class PPOMujocoUtils(AlgoBase.AlgoBaseUtils):
    pass
                    
class PPOMujocoAgent(AlgoBase.AlgoBaseAgent):
    def __init__(self,sample_net:PPOMujocoNet,config_dict,is_checker):
        super(PPOMujocoAgent,self).__init__()
        self.sample_net = sample_net
        self.config_dict = config_dict
        self.num_steps = args.num_steps
        self.num_envs = args.num_envs
        self.pae_length = args.pae_length
        self.rewards = []
        
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
        
    def sample_env(self):
        while len(self.exps_list[0]) < self.num_steps:
            
            actions,log_probs = self.get_sample_actions(self.states)
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
        self.exps_list = [ exps[self.pae_length:self.num_steps] for exps in self.exps_list ]
        return train_exps
    
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
            
    @torch.no_grad()
    def get_sample_actions(self,states):
        states_v = torch.Tensor(np.array(states))
        actions,log_probs = self.sample_net.get_sample_data(states_v)
        return actions.cpu().numpy(),log_probs.cpu().numpy()
    
    @torch.no_grad()
    def get_check_action(self,state):
        state_v = torch.Tensor(np.array(state))
        # add random noise to the state
        epsilon_t = args.attacker_noise_limit
        perturbations = np.random.uniform(-epsilon_t, epsilon_t, size=state.shape)
        state_v = state_v + perturbations
        mu,entropy,log_prob = self.sample_net.get_check_data(state_v)
        return mu.cpu().numpy(),entropy.cpu().numpy(),log_prob.cpu().numpy()        
            
class PPOMujocoCalculate(AlgoBase.AlgoBaseCalculate):
    def __init__(self,share_model:PPOMujocoNet,config_dict,calculate_index):
        super(PPOMujocoCalculate,self).__init__()
        self.config_dict = config_dict
        self.share_model = share_model
        self.pae_length = args.pae_length
        self.calculate_number = self.config_dict['num_trainer']
        self.calculate_index = calculate_index
        self.train_version = 0
        
        if args.use_gpu and torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_index = self.calculate_index % device_count
            self.device = torch.device('cuda',device_index)
        else:
            self.device = torch.device('cpu')
                        
        self.calculate_net = PPOMujocoNet()
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
        
        self.states_list = torch.cat([states[0:self.pae_length] for states in self.states])
        self.actions_list = torch.cat([actions[0:self.pae_length] for actions in self.actions])
        self.old_log_probs_list = torch.cat([old_log_probs[0:self.pae_length] for old_log_probs in self.old_log_probs])
        
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
        
        # Resetting sample noise
        if self.calculate_index == self.calculate_number - 1:
            self.share_model.sample_noise()

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
        train_version = self.config_dict[self.calculate_index]
        vf_coef = args.vf_coef
        ent_coef = args.ent_coef
        grad_norm = args.grad_norm
        adpr_coef = args.adpr_coef
        pg_loss_type = args.pg_loss_type
        mini_batch_size = args.mini_batch_size
        

        ratio_coef = self.get_ratio_coef(train_version)
    
        self.calculate_net.load_state_dict(self.share_model.state_dict())
                        
        np_advantages,np_returns = self.calculate_samples_gae()

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
                
            mini_new_values,mini_new_log_probs,mini_entropys = self.calculate_net(mini_states,mini_actions)
                        
            mini_advantage = advantage_list[start_index:end_index]
            mini_returns = returns_list[start_index:end_index]

            batch_action_means = self.calculate_net.mu(mini_states)

            adpr = getstate_kl_bound_sgld(self.calculate_net, mini_states, batch_action_means, args.attacker_noise_limit, args.eps_steps, torch.exp(self.calculate_net.log_std))

            loss_adpr = adpr.mean()*adpr_coef

            ratio1 = torch.exp(mini_new_log_probs-mini_old_log_probs)

            ratio2 = ratio1.prod(1,keepdim=True).expand_as(ratio1)

            ratio3 = ratio1 * ratio_coef + ratio2 * (1.0 - ratio_coef)

            #discrete
            if pg_loss_type == 0:
                pg_loss = self.get_pg_loss(ratio1,mini_advantage)
                
            #prod
            elif pg_loss_type == 1:
                pg_loss = self.get_pg_loss(ratio2,mini_advantage)
                
            #mixed
            elif pg_loss_type == 2:
                pg_loss = self.get_pg_loss(ratio3,mini_advantage)
                
            #last_mixed
            elif pg_loss_type == 3:
                pg_loss1 = self.get_pg_loss(ratio1,mini_advantage)
                pg_loss2 = self.get_pg_loss(ratio2,mini_advantage)
                pg_loss = (pg_loss1+pg_loss2)/2
                            
            # Policy loss
            pg_loss = -torch.mean(pg_loss)
            
            v_loss = F.mse_loss(mini_returns, mini_new_values) * vf_coef
            
            e_loss = -torch.mean(mini_entropys) * ent_coef
            
            loss = pg_loss + v_loss + e_loss + loss_adpr
            
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

    """def state_adversarial_policy_regularizer(self):
        R_ppo_theta = 0
        s_i = self.states_list
        with torch.no_grad():
            pi_s_i = self.calculate_net.get_distris(s_i)
        s_i_perturbeds = B_p(s_i, args.attacker_noise_limit)
        pi_s_i_perturbeds = self.calculate_net.get_distris(s_i_perturbeds)
        kl_divergence = KL_divergence(pi_s_i, pi_s_i_perturbeds)
        R_ppo_theta = kl_divergence.mean()
        return R_ppo_theta"""

@torch.no_grad()
def B_p(state, epsilon_t):
    """
    Generates a set of perturbed states within a neighborhood defined by epsilon_t.
    """
    perturbations = np.random.uniform(-epsilon_t, epsilon_t, size=state.shape)
    perturbed_states = state + perturbations
    # if perturbed_states is numpy array
    # perturbed_states = torch.tensor(perturbed_states, dtype=torch.float32)
    if isinstance(perturbed_states, np.ndarray):
        perturbed_states = torch.tensor(perturbed_states, dtype=torch.float32)
    elif isinstance(perturbed_states, torch.Tensor):
        perturbed_states = perturbed_states.type(torch.float32)
    return perturbed_states


def KL_divergence(p, q):
    return torch.distributions.kl.kl_divergence(p, q)

def getstate_kl_bound_sgld(net, batch_states, batch_action_means, eps, steps, stdev):
    warpped_net = net
    batch_action_means = batch_action_means.detach()
    # upper and lower bounds for clipping
    states_ub = batch_states + eps
    states_lb = batch_states - eps
    step_eps = eps / steps
    # SGLD noise factor. We set (inverse) beta=1e-5 as gradients are relatively small here.
    beta = 1e-5
    noise_factor = np.sqrt(2 * step_eps * beta)
    noise = torch.randn_like(batch_states) * noise_factor
    var_states = (batch_states.clone() + noise.sign() * step_eps).detach().requires_grad_()
    for i in range(steps):
        # Find a nearby state new_phi that maximize the difference
        diff = (warpped_net.mu(var_states) - batch_action_means) / stdev
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
    net.zero_grad()
    diff = (warpped_net.get_sample_data(var_states.requires_grad_(False))[0] - batch_action_means) / stdev
    return (diff * diff).sum(axis=-1, keepdim=True)
        
if __name__ == "__main__":
    comment = "SAPPO Mujoco Training"
    writer = tensorboard.SummaryWriter(comment=comment)
    # initialize training network
    train_net = PPOMujocoNet()
    config_dict = {}
    config_dict[0] = 0
    config_dict['num_trainer'] = 1
    config_dict['train_version'] = 0

    # initialize a RL agent, smaple agent used for sampling training data, check agent used for evaluating 
    # and calculate used for calculating gradients
    sample_agent = PPOMujocoAgent(train_net,config_dict,is_checker=False)
    check_agent = PPOMujocoAgent(train_net,config_dict,is_checker=True)
    calculate = PPOMujocoCalculate(train_net,config_dict,0)

    # hyperparameters
    MAX_VERSION = 3000
    REPEAT_TIMES = 10

    for iteration in range(MAX_VERSION):
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
        config_dict[0] = config_dict[0] + 1
        config_dict['train_version'] = config_dict[0]
        
        # Evaluating agent
        infos = check_agent.check_env()
        for k,v in infos.items():
            writer.add_scalar(k,v,config_dict[0])
         
        print("version:",config_dict[0],"sum_rewards:",infos['sum_rewards'])

    # Saving model
    torch.save(train_net, "saved_model/sappo_mujoco_"+parser.env_name+".pkl")
    # Saving model dict
    torch.save(train_net.state_dict(), "saved_model/sappo_mujoco_"+parser.env_name+".pth")