#############################################################
# SAC agent class and helpers
#############################################################

import numpy as np
from enum import Enum
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from copy import deepcopy
from os import path, makedirs
import pickle
import pandas as pd

def load_model_weights(model, weights_file_path): 
    try:
        weights_file = torch.load(weights_file_path)
        model.load_state_dict(weights_file)
        print('Weights loaded:', weights_file_path)
    except IOError:
        print('Weights file does not exist!')
        print(weights_file_path, '\n')
        exit(1)


def create_base_dir(dir_path):
    if not path.exists(path.dirname(dir_path)):
        makedirs(path.dirname(dir_path))

def reward_shapedVVHA(bombAppeared,x_next,y_next,xc,yc,next_state):
    
    reward_VVHA = 0
    isClose = False

    # 
    distK = 0.2 # fH and fV were calculated for distK=0.2

    # Add penalty when x-distance to bomb is <0.2 and y > yc
    if bombAppeared and np.abs(x_next-xc) < distK and y_next>yc:

        # force horizontal
        k = 0.5/(3*(0.2)+0.05) # fH when x=0.2
        fH = -0.5/(3*np.abs(x_next-xc)+0.05)+k
        
        # force down
        fV = -3*(y_next-yc+distK-0.25)*(y_next-yc+distK-0.25) 
        
        # angle
        angle = np.abs(next_state[4])
        fA = -1*angle*angle
        
        # velocity
        vx = np.abs(next_state[2])
        vy = np.abs(next_state[3])
        vangle = np.abs(next_state[5])
        
        penalty_vel = -2*(vx*vx+vy*vy+vangle*vangle) 

        # add to reward
        reward_VVHA += fH
        reward_VVHA += fV
        reward_VVHA += fA
        reward_VVHA += penalty_vel

        isClose = True
    
    return reward_VVHA, isClose

class ScoreBoard_bombs:

    def __init__(self, save_path):
        
        self.max_buffer_len = 100

        # Create dataframe to store
        self.df = pd.DataFrame(columns=['TimeStepsTotal','Episode','AvgAlpha','AvgAlpha_I','AvgLoss','UpdatesTotal','Avg100Reward','Avg100EpiLen','Num_Hits','Num_Bombs','Num_Landings','I_pct','Bomb_Trial'])
        self.savename = path.join(save_path, 'df_scoreboard.csv')

    def update_data_frame(self,current_total_num_timesteps,episode,alpha,alpha_I,loss,current_num_total_updates,list_rewards,list_epilen, nhits, nbombs,nlandings, Ipct, isBomb):
        d = {'TimeStepsTotal':current_total_num_timesteps,'Episode':episode, 'AvgAlpha':alpha,'AvgAlpha_I':alpha_I,'AvgLoss':loss,'UpdatesTotal':current_num_total_updates,'Avg100Reward':sum(list_rewards)/len(list_rewards),'Avg100EpiLen':sum(list_epilen)/len(list_epilen),'Num_Hits':nhits,'Num_Bombs':nbombs,'Num_Landings':nlandings,'I_pct':Ipct, 'Bomb_Trial':isBomb}
        self.df = self.df.append(d,ignore_index=True)

    def save_df(self):
        self.df.to_csv(self.savename,index=False)
        print('- Scoreboard dataframe saved to:',self.savename)

class ScoreBoard_bipedalH2:

    def __init__(self, save_path):
        
        self.max_buffer_len = 100

        # Create dataframe to store
        self.df = pd.DataFrame(columns=['TimeStepsTotal','Episode','AvgAlpha','AvgAlpha_I','AvgLoss','UpdatesTotal','Avg100Reward','Avg100EpiLen','Num_Hits','I_pct'])
        self.savename = path.join(save_path, 'df_scoreboard.csv')

    def update_data_frame(self,current_total_num_timesteps,episode,alpha,alpha_I,loss,current_num_total_updates,list_rewards,list_epilen, nhits, Ipct):
        d = {'TimeStepsTotal':current_total_num_timesteps,'Episode':episode, 'AvgAlpha':alpha,'AvgAlpha_I':alpha_I,'AvgLoss':loss,'UpdatesTotal':current_num_total_updates,'Avg100Reward':sum(list_rewards)/len(list_rewards),'Avg100EpiLen':sum(list_epilen)/len(list_epilen),'Num_Hits':nhits, 'I_pct':Ipct}
        self.df = self.df.append(d,ignore_index=True)

    def save_df(self):
        self.df.to_csv(self.savename,index=False)
        print('- Scoreboard dataframe saved to:',self.savename)

class ScoreBoard_bipedalH3:

    def __init__(self, save_path):
        
        self.max_buffer_len = 100

        # Create dataframe to store
        self.df = pd.DataFrame(columns=['TimeStepsTotal','Episode','AvgAlpha','AvgAlpha_I','AvgLoss','UpdatesTotal','CumReward','Avg100Reward','Avg100EpiLen','Num_Hits','I_pct','trialtype'])
        self.savename = path.join(save_path, 'df_scoreboard.csv')

    def update_data_frame(self,current_total_num_timesteps,episode,alpha,alpha_I,loss,current_num_total_updates,CumReward,list_rewards,list_epilen, nhits, Ipct, trialtype):
        d = {'TimeStepsTotal':current_total_num_timesteps,'Episode':episode, 'AvgAlpha':alpha,'AvgAlpha_I':alpha_I,'AvgLoss':loss,'UpdatesTotal':current_num_total_updates,'CumReward':CumReward,'Avg100Reward':sum(list_rewards)/len(list_rewards),'Avg100EpiLen':sum(list_epilen)/len(list_epilen),'Num_Hits':nhits, 'I_pct':Ipct,'trialtype':trialtype}
        self.df = self.df.append(d,ignore_index=True)

    def save_df(self):
        self.df.to_csv(self.savename,index=False)
        print('- Scoreboard dataframe saved to:',self.savename)

class ScoreBoard_bipedalH:

    def __init__(self, save_path):
        
        self.max_buffer_len = 100

        # Create dataframe to store
        self.df = pd.DataFrame(columns=['TimeStepsTotal','Episode','AvgAlpha','AvgLoss','UpdatesTotal','Avg100Reward','Avg100EpiLen','Num_Hits'])
        self.savename = path.join(save_path, 'df_scoreboard.csv')

    def update_data_frame(self,current_total_num_timesteps,episode,alpha,loss,current_num_total_updates,list_rewards,list_epilen, nhits):
        d = {'TimeStepsTotal':current_total_num_timesteps,'Episode':episode, 'AvgAlpha':alpha,'AvgLoss':loss,'UpdatesTotal':current_num_total_updates,'Avg100Reward':sum(list_rewards)/len(list_rewards),'Avg100EpiLen':sum(list_epilen)/len(list_epilen),'Num_Hits':nhits}
        self.df = self.df.append(d,ignore_index=True)

    def save_df(self):
        self.df.to_csv(self.savename,index=False)
        print('- Scoreboard dataframe saved to:',self.savename)

class ScoreBoard:

    def __init__(self, save_path):
        
        self.max_buffer_len = 100

        # Create dataframe to store
        self.df = pd.DataFrame(columns=['TimeStepsTotal','Episode','AvgAlpha','AvgLoss','UpdatesTotal','Avg100Reward','Avg100EpiLen','Num_Hits'])
        self.savename = path.join(save_path, 'df_scoreboard.csv')

    def update_data_frame(self,current_total_num_timesteps,episode,alpha,loss,current_num_total_updates,list_rewards,list_epilen, nhits):
        d = {'TimeStepsTotal':current_total_num_timesteps,'Episode':episode, 'AvgAlpha':alpha,'AvgLoss':loss,'UpdatesTotal':current_num_total_updates,'Avg100Reward':sum(list_rewards)/len(list_rewards),'Avg100EpiLen':sum(list_epilen)/len(list_epilen),'Num_Hits':nhits}
        self.df = self.df.append(d,ignore_index=True)

    def save_df(self):
        self.df.to_csv(self.savename,index=False)
        print('- Scoreboard dataframe saved to:',self.savename)


class SAC_I():
    """ Soft Actor-Critic  """

    def __init__(self, state_dim,
                       action_dim,
                       hidden_layer_size,
                       q_lr,
                       policy_lr,
                       reward_scale,
                       num_hidden_layers=1,
                       gamma=0.99,
                       tau=1e-2,
                       use_automatic_entropy_tunning=False,
                       target_entropy=None,
                       entropy_scaling_lr=None,
                       device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                       max_nstep_return=1,
                       version=None):

        self.reward_scale = reward_scale
        self.tau = tau
        self.use_automatic_entropy_tunning = use_automatic_entropy_tunning
        self.target_entropy = target_entropy
        self.device = device

        self.max_nstep_return = max_nstep_return
        self.gammas = torch.zeros((max_nstep_return + 1, 1))
        for i_n in range(max_nstep_return + 1):
            self.gammas[i_n, :] = gamma ** i_n
            # self.gammas.append(gamma ** i_n)
        
        self.gamma = gamma # JI: adding back

        # initialize networks
        self.q_1_net = QNetwork(state_dim, action_dim, hidden_layer_size, q_lr, num_hidden_layers=num_hidden_layers).to(device)
        self.q_2_net = QNetwork(state_dim, action_dim, hidden_layer_size, q_lr, num_hidden_layers=num_hidden_layers).to(device)
        self.target_q_1_net = QNetwork(state_dim, action_dim, hidden_layer_size, q_lr, num_hidden_layers=num_hidden_layers).to(device)
        self.target_q_2_net = QNetwork(state_dim, action_dim, hidden_layer_size, q_lr, num_hidden_layers=num_hidden_layers).to(device)

        self.q_I_1_net = QNetwork(state_dim, action_dim, hidden_layer_size, q_lr, num_hidden_layers=num_hidden_layers).to(device)
        self.q_I_2_net = QNetwork(state_dim, action_dim, hidden_layer_size, q_lr, num_hidden_layers=num_hidden_layers).to(device)
        self.target_q_I_1_net = QNetwork(state_dim, action_dim, hidden_layer_size, q_lr, num_hidden_layers=num_hidden_layers).to(device)
        self.target_q_I_2_net = QNetwork(state_dim, action_dim, hidden_layer_size, q_lr, num_hidden_layers=num_hidden_layers).to(device)

        #complete copy from local to target => tau=1.0
        self.soft_update(self.q_1_net, self.target_q_1_net, 1.0) 
        self.soft_update(self.q_2_net, self.target_q_2_net, 1.0)

        self.soft_update(self.q_I_1_net, self.target_q_I_1_net, 1.0) 
        self.soft_update(self.q_I_2_net, self.target_q_I_2_net, 1.0)

        if version=='ji': # uses the original reparametrization to compute action (sample the noise)
            print('(*) Using the policy network version JI...')
            self.policy_net = PolicyNetwork_ji(state_dim, action_dim, hidden_layer_size, policy_lr, device, num_hidden_layers=num_hidden_layers).to(device)
        else:
            self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_layer_size, policy_lr, device, num_hidden_layers=num_hidden_layers).to(device)

        if self.use_automatic_entropy_tunning:
            self.entropy_scaling = torch.zeros(1, requires_grad=True, device=device)
            self.entropy_scaling_optimizer = optim.Adam([self.entropy_scaling], lr=entropy_scaling_lr)
            # Inhibitory net
            self.entropy_scaling_I = torch.zeros(1, requires_grad=True, device=device)
            self.entropy_scaling_I_optimizer = optim.Adam([self.entropy_scaling_I], lr=entropy_scaling_lr)

        ## JI: define optimizers
        self.q_1_net_optimizer = optim.Adam(self.q_1_net.parameters(), lr=q_lr)
        self.q_2_net_optimizer = optim.Adam(self.q_2_net.parameters(), lr=q_lr)
        self.q_I_1_net_optimizer = optim.Adam(self.q_I_1_net.parameters(), lr=q_lr)
        self.q_I_2_net_optimizer = optim.Adam(self.q_I_2_net.parameters(), lr=q_lr)
        self.policy_net_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)


    def act(self, state, hidden_gru_state=None):
        """Returns actions for given state as per current policy."""

        state = torch.from_numpy(state).float().to(self.device)

        self.policy_net.eval()

        with torch.no_grad():

            action, std = self.policy_net(state)
            
        self.policy_net.train()

        return action.cpu().data.numpy(), hidden_gru_state, std.cpu().data.numpy()

    # Learn regular Q-net
    def learn(self, states, actions, rewards, next_states, dones, sample_weights, nsteps):
        
        """ Update policies from batch of experience tuples. """
       
        # evaluate policy to get next actions, log probability, etc.
        new_actions, _, log_prob = self.policy_net(states, return_log_prob=True)
        
        """
        Alpha Loss
        """
        # See https://arxiv.org/pdf/1812.05905.pdf, section 5 explanation and see equation 17
        # Also https://github.com/vitchyr/rlkit/blob/2e8097f2a2c4d4d5079f12b8210f8ac8f250aee0/rlkit/torch/sac/sac.py#L99
        if self.use_automatic_entropy_tunning:
            alpha_loss = -(self.entropy_scaling * (log_prob + self.target_entropy).detach()).mean()
            self.entropy_scaling_optimizer.zero_grad()
            alpha_loss.backward()
            self.entropy_scaling_optimizer.step()
            alpha = self.entropy_scaling.exp()
        else:
            alpha = 0.1 # original
            # alpha = 0.05 # LunarLander after trained...
        
        """
        Policy Loss
        """
        predicted_new_q_values = torch.min(self.q_1_net(states, new_actions),
                                           self.q_2_net(states, new_actions))
        
        policy_loss = (alpha * log_prob - predicted_new_q_values).mean() # JI

        """
        Q-function losses
        """
        ## No grad (perhaps it is not enough to detach q_target only... doing for all the involved calculations.)
        with torch.no_grad():
            next_state_actions, _, new_log_prob = self.policy_net(next_states, return_log_prob=True)    
            target_q_value = torch.min(self.target_q_1_net(next_states, next_state_actions),
                                    self.target_q_2_net(next_states, next_state_actions)) - alpha * new_log_prob
            q_target = self.reward_scale * rewards + (1.0 - dones) * self.gamma * target_q_value
            # q_target = rewards + masks * self.gamma * target_q_value # RAFAELs version... (WORKING)

        # get losses
        q1_loss = torch.mean((self.q_1_net(states, actions) - q_target).pow(2)) # JI
        q2_loss = torch.mean((self.q_2_net(states, actions) - q_target).pow(2)) # JI
         
        """
        Update Networks
        """
        # update
        self.q_1_net_optimizer.zero_grad()
        q1_loss.backward()
        self.q_1_net_optimizer.step()  

        self.q_2_net_optimizer.zero_grad()
        q2_loss.backward()
        self.q_2_net_optimizer.step()  

        self.policy_net_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_net_optimizer.step()  

        td_loss_1, td_loss_2 = None, None

        """
        Soft Updates
        """
        self.soft_update(self.q_1_net, self.target_q_1_net, self.tau)
        self.soft_update(self.q_2_net, self.target_q_2_net, self.tau)

        return policy_loss, alpha, (td_loss_1, td_loss_2), q_target.detach().cpu().data.numpy() # JI

    def learn_I(self, states, actions, rewards_I, next_states, dones, sample_weights, nsteps):
        

        """ Update policies from batch of experience tuples. """
       
        # evaluate policy to get next actions, log probability, etc.
        new_actions, _, log_prob = self.policy_net(states, return_log_prob=True)
        
        """
        Alpha Loss
        """
        # See https://arxiv.org/pdf/1812.05905.pdf, section 5 explanation and see equation 17
        # Also https://github.com/vitchyr/rlkit/blob/2e8097f2a2c4d4d5079f12b8210f8ac8f250aee0/rlkit/torch/sac/sac.py#L99
        if self.use_automatic_entropy_tunning:
            alpha_loss = -(self.entropy_scaling * (log_prob + self.target_entropy).detach()).mean()
            self.entropy_scaling_optimizer.zero_grad()
            alpha_loss.backward()
            self.entropy_scaling_optimizer.step()
            alpha = self.entropy_scaling.exp()
        else:
            alpha = 0.1
        
        """
        Policy Loss
        """

        # JI: Include the inhibitory network
        predicted_new_q_values = torch.min(self.q_I_1_net(states, new_actions),self.q_I_2_net(states, new_actions))
    
        # policy_loss = (sample_weights * (alpha * log_prob - predicted_new_q_values)).mean()
        policy_loss = (alpha * log_prob - predicted_new_q_values).mean() # JI

        """
        Q-function losses
        """
    
        ## No grad (perhaps it is not enough to detach q_target only... doing for all the involved calculations.)
        with torch.no_grad():
            next_state_actions, _, new_log_prob = self.policy_net(next_states, return_log_prob=True)    

            target_q_I_value = torch.min(self.target_q_I_1_net(next_states, next_state_actions),
                                   self.target_q_I_2_net(next_states, next_state_actions)) - alpha * new_log_prob # JI: now including double net

            q_I_target = self.reward_scale * rewards_I + (1.0 - dones) * self.gamma * target_q_I_value

        # get losses
        q_I1_loss = torch.mean((self.q_I_1_net(states, actions) - q_I_target).pow(2)) # JI
        q_I2_loss = torch.mean((self.q_I_2_net(states, actions) - q_I_target).pow(2)) # JI
                 
        """
        Update Networks
        """
        # update
        self.q_I_1_net_optimizer.zero_grad()
        q_I1_loss.backward()
        self.q_I_1_net_optimizer.step()  

        self.q_I_2_net_optimizer.zero_grad()
        q_I2_loss.backward()
        self.q_I_2_net_optimizer.step()  

        self.policy_net_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_net_optimizer.step()  

        td_loss_1, td_loss_2 = None, None

        """
        Soft Updates
        """
        self.soft_update(self.q_I_1_net, self.target_q_I_1_net, self.tau)
        self.soft_update(self.q_I_2_net, self.target_q_I_2_net, self.tau)

        return policy_loss, alpha, (td_loss_1, td_loss_2), q_I_target.detach().cpu().data.numpy() # JI

    def learn_both_splitAlpha_fixReg(self, states, actions, rewards, rewards_I, use_I, next_states, dones, sample_weights, nsteps, alphaK=0.1):

        """ Update policies from batch of experience tuples. """
       
        # evaluate policy to get next actions, log probability, etc.
        new_actions, _, log_prob = self.policy_net(states, return_log_prob=True)
        
        """
        Alpha Loss
        """
        # See https://arxiv.org/pdf/1812.05905.pdf, section 5 explanation and see equation 17
        # Also https://github.com/vitchyr/rlkit/blob/2e8097f2a2c4d4d5079f12b8210f8ac8f250aee0/rlkit/torch/sac/sac.py#L99
        if self.use_automatic_entropy_tunning:
            # REGULAR state
            nReg = (use_I==0).sum()
            alpha = alphaK
            
            # I-state
            nI = (use_I==1).sum()
            if nI != 0:
                alpha_loss_I = -(((self.entropy_scaling_I * (log_prob + self.target_entropy).detach())*use_I).sum(dim=0)/nI) # MEAN WITHOUT ZEROS
                self.entropy_scaling_I_optimizer.zero_grad()
                alpha_loss_I.backward()
                self.entropy_scaling_I_optimizer.step()
                alpha_I = self.entropy_scaling_I.exp()
            else:
                alpha_loss_I = -((self.entropy_scaling_I * (log_prob + self.target_entropy).detach())*use_I).sum(dim=0)
                alpha_I = alphaK

        else:
            alpha = alphaK # 0.1
        
        """
        Policy Loss
        """

        # JI: Given use_I, switch between q1,q2 x q_I_1,q_I_2
        q12_min = torch.min(self.q_1_net(states, new_actions),self.q_2_net(states, new_actions))
        q_I_12_min = torch.min(self.q_I_1_net(states, new_actions),self.q_I_2_net(states, new_actions))
        predicted_new_q_values = (1.0 - use_I)*q12_min + use_I*q_I_12_min 
       
        policy_loss = ( alpha * log_prob*(1-use_I) + alpha_I * log_prob*(use_I)   - predicted_new_q_values).mean() # JI: version with split alpha

        """
        Q-function losses
        """
        ## No grad (perhaps it is not enough to detach q_target only... doing for all the involved calculations.)
        with torch.no_grad():
            next_state_actions, _, new_log_prob = self.policy_net(next_states, return_log_prob=True)    
            target_q_value = torch.min(self.target_q_1_net(next_states, next_state_actions),
                                    self.target_q_2_net(next_states, next_state_actions)) - alpha * new_log_prob
            target_q_I_value = torch.min(self.target_q_I_1_net(next_states, next_state_actions),
                                   self.target_q_I_2_net(next_states, next_state_actions)) - alpha * new_log_prob # JI: now including double net

            q_target = self.reward_scale * rewards + (1.0 - dones) * self.gamma * target_q_value
            q_I_target = self.reward_scale * rewards_I + (1.0 - dones) * self.gamma * target_q_I_value # FIXED

        # get losses
        q1_loss = torch.mean((self.q_1_net(states, actions) - q_target).pow(2)) # JI
        q2_loss = torch.mean((self.q_2_net(states, actions) - q_target).pow(2)) # JI
        q_I1_loss = torch.mean((self.q_I_1_net(states, actions) - q_I_target).pow(2)) # JI
        q_I2_loss = torch.mean((self.q_I_2_net(states, actions) - q_I_target).pow(2)) # JI
         
        """
        Update Networks
        """
        # update
        self.q_1_net_optimizer.zero_grad()
        q1_loss.backward()
        self.q_1_net_optimizer.step()  

        self.q_2_net_optimizer.zero_grad()
        q2_loss.backward()
        self.q_2_net_optimizer.step()  

        self.q_I_1_net_optimizer.zero_grad()
        q_I1_loss.backward()
        self.q_I_1_net_optimizer.step()  

        self.q_I_2_net_optimizer.zero_grad()
        q_I2_loss.backward()
        self.q_I_2_net_optimizer.step()  

        self.policy_net_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_net_optimizer.step()  

        td_loss_1, td_loss_2 = None, None

        """
        Soft Updates
        """
        self.soft_update(self.q_1_net, self.target_q_1_net, self.tau)
        self.soft_update(self.q_2_net, self.target_q_2_net, self.tau)
        self.soft_update(self.q_I_1_net, self.target_q_I_1_net, self.tau)
        self.soft_update(self.q_I_2_net, self.target_q_I_2_net, self.tau)

        return policy_loss, alpha, alpha_I, (td_loss_1, td_loss_2), q_target.detach().cpu().data.numpy() # JI
    
    ## Used in SAC-I agent VVHA
    def learn_both_splitAlpha_splitQloss_limitA(self, states, actions, rewards, rewards_I, use_I, next_states, dones, sample_weights, nsteps, last_alpha=None, last_alpha_I=None, max_alpha_I = 0.03):

        """ Update policies from batch of experience tuples. """
       
        # evaluate policy to get next actions, log probability, etc.
        new_actions, _, log_prob = self.policy_net(states, return_log_prob=True)
        
        """
        Alpha Loss
        """
        # See https://arxiv.org/pdf/1812.05905.pdf, section 5 explanation and see equation 17
        # Also https://github.com/vitchyr/rlkit/blob/2e8097f2a2c4d4d5079f12b8210f8ac8f250aee0/rlkit/torch/sac/sac.py#L99
        if self.use_automatic_entropy_tunning:
            nReg = (use_I==0).sum()
            if nReg!=0:
                alpha_loss = -(((self.entropy_scaling * (log_prob + self.target_entropy).detach())*(1-use_I)).sum(dim=0)/nReg) # MEAN WITHOUT ZEROS
                self.entropy_scaling_optimizer.zero_grad()
                alpha_loss.backward()
                self.entropy_scaling_optimizer.step()
                alpha = self.entropy_scaling.exp()
                alpha = np.min([alpha.cpu().data.numpy()[0],max_alpha_I])
            else:
                alpha_loss = -((self.entropy_scaling * (log_prob + self.target_entropy).detach())*(1-use_I)).sum(dim=0)
                alpha = last_alpha

            # I-state
            nI = (use_I==1).sum()
            if nI != 0:
                alpha_loss_I = -(((self.entropy_scaling_I * (log_prob + self.target_entropy).detach())*use_I).sum(dim=0)/nI) # MEAN WITHOUT ZEROS
                self.entropy_scaling_I_optimizer.zero_grad()
                alpha_loss_I.backward()
                self.entropy_scaling_I_optimizer.step()
                alpha_I = self.entropy_scaling_I.exp()
                alpha_I = np.min([alpha_I.cpu().data.numpy()[0],max_alpha_I])
            else:
                alpha_loss_I = -((self.entropy_scaling_I * (log_prob + self.target_entropy).detach())*use_I).sum(dim=0)
                alpha_I = last_alpha_I

        else:
            alpha = last_alpha # 
        
        """
        Policy Loss
        """
        # JI: Given use_I, switch between q1,q2 x q_I_1,q_I_2
        q12_min = torch.min(self.q_1_net(states, new_actions),self.q_2_net(states, new_actions))
        q_I_12_min = torch.min(self.q_I_1_net(states, new_actions),self.q_I_2_net(states, new_actions))
        predicted_new_q_values = (1.0 - use_I)*q12_min + use_I*q_I_12_min 
        policy_loss = ( alpha * log_prob*(1-use_I) + alpha_I * log_prob*(use_I)   - predicted_new_q_values).mean() # JI: version with split alpha

        """
        Q-function losses (split losses for Q1 and Q_I)
        """
        ## No grad (perhaps it is not enough to detach q_target only... doing for all the involved calculations.)
        with torch.no_grad():
            next_state_actions, _, new_log_prob = self.policy_net(next_states, return_log_prob=True)    
            target_q_value = torch.min(self.target_q_1_net(next_states, next_state_actions),
                                    self.target_q_2_net(next_states, next_state_actions)) - alpha * new_log_prob
            target_q_I_value = torch.min(self.target_q_I_1_net(next_states, next_state_actions),
                                   self.target_q_I_2_net(next_states, next_state_actions)) - alpha * new_log_prob # JI: now including double net

            q_target = self.reward_scale * rewards + (1.0 - dones) * self.gamma * target_q_value
            q_I_target = self.reward_scale * rewards_I + (1.0 - dones) * self.gamma * target_q_I_value # FIXED

        # Split losses by step type
        if nReg!=0:
            q1_loss = ( (self.q_1_net(states, actions) - q_target).pow(2)*(1-use_I) ).sum(dim=0)/nReg # mean without the zeros
            q2_loss = ( (self.q_2_net(states, actions) - q_target).pow(2)*(1-use_I) ).sum(dim=0)/nReg # mean without the zeros
        if nI!=0:
            q_I1_loss = ( (self.q_I_1_net(states, actions) - q_I_target).pow(2)*(use_I) ).sum(dim=0)/nI # mean without the zeros
            q_I2_loss = ( (self.q_I_2_net(states, actions) - q_I_target).pow(2)*(use_I) ).sum(dim=0)/nI # mean without the zeros

        """
        Update Networks
        """
        # update
        if nReg!=0: # Do not update if there were not regular Q1 samples
            self.q_1_net_optimizer.zero_grad()
            q1_loss.backward()
            self.q_1_net_optimizer.step()  

            self.q_2_net_optimizer.zero_grad()
            q2_loss.backward()
            self.q_2_net_optimizer.step()  
        if nI!=0: # Do not update with there were no I-samples
            self.q_I_1_net_optimizer.zero_grad()
            q_I1_loss.backward()
            self.q_I_1_net_optimizer.step()  

            self.q_I_2_net_optimizer.zero_grad()
            q_I2_loss.backward()
            self.q_I_2_net_optimizer.step()  

        self.policy_net_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_net_optimizer.step()  

        td_loss_1, td_loss_2 = None, None

        """
        Soft Updates
        """
        self.soft_update(self.q_1_net, self.target_q_1_net, self.tau)
        self.soft_update(self.q_2_net, self.target_q_2_net, self.tau)
        self.soft_update(self.q_I_1_net, self.target_q_I_1_net, self.tau)
        self.soft_update(self.q_I_2_net, self.target_q_I_2_net, self.tau)

        return policy_loss, alpha, alpha_I, (td_loss_1, td_loss_2), q_target.detach().cpu().data.numpy() # JI


    def learn_both_splitAlpha_splitQloss(self, states, actions, rewards, rewards_I, use_I, next_states, dones, sample_weights, nsteps, last_alpha=None, last_alpha_I=None):
        """ Update policies from batch of experience tuples. """
       
        # evaluate policy to get next actions, log probability, etc.
        new_actions, _, log_prob = self.policy_net(states, return_log_prob=True)
        
        """
        Alpha Loss
        """
        # See https://arxiv.org/pdf/1812.05905.pdf, section 5 explanation and see equation 17
        # Also https://github.com/vitchyr/rlkit/blob/2e8097f2a2c4d4d5079f12b8210f8ac8f250aee0/rlkit/torch/sac/sac.py#L99
        if self.use_automatic_entropy_tunning:
            nReg = (use_I==0).sum()
            if nReg!=0:
                alpha_loss = -(((self.entropy_scaling * (log_prob + self.target_entropy).detach())*(1-use_I)).sum(dim=0)/nReg) # MEAN WITHOUT ZEROS
                self.entropy_scaling_optimizer.zero_grad()
                alpha_loss.backward()
                self.entropy_scaling_optimizer.step()
                alpha = self.entropy_scaling.exp()
            else:
                alpha_loss = -((self.entropy_scaling * (log_prob + self.target_entropy).detach())*(1-use_I)).sum(dim=0)
                alpha = last_alpha

            # I-state
            nI = (use_I==1).sum()
            if nI != 0:
                alpha_loss_I = -(((self.entropy_scaling_I * (log_prob + self.target_entropy).detach())*use_I).sum(dim=0)/nI) # MEAN WITHOUT ZEROS
                self.entropy_scaling_I_optimizer.zero_grad()
                alpha_loss_I.backward()
                self.entropy_scaling_I_optimizer.step()
                alpha_I = self.entropy_scaling_I.exp()
            else:
                alpha_loss_I = -((self.entropy_scaling_I * (log_prob + self.target_entropy).detach())*use_I).sum(dim=0)
                alpha_I = last_alpha_I
        else:
            alpha = alphaK # 0.1
        
        """
        Policy Loss
        """
        # JI: Given use_I, switch between q1,q2 x q_I_1,q_I_2
        q12_min = torch.min(self.q_1_net(states, new_actions),self.q_2_net(states, new_actions))
        q_I_12_min = torch.min(self.q_I_1_net(states, new_actions),self.q_I_2_net(states, new_actions))
        predicted_new_q_values = (1.0 - use_I)*q12_min + use_I*q_I_12_min 
        policy_loss = ( alpha * log_prob*(1-use_I) + alpha_I * log_prob*(use_I)   - predicted_new_q_values).mean() # JI: version with split alpha

        """
        Q-function losses (split losses for Q1 and Q_I)
        """
        ## No grad (perhaps it is not enough to detach q_target only... doing for all the involved calculations.)
        with torch.no_grad():
            next_state_actions, _, new_log_prob = self.policy_net(next_states, return_log_prob=True)    
            target_q_value = torch.min(self.target_q_1_net(next_states, next_state_actions),
                                    self.target_q_2_net(next_states, next_state_actions)) - alpha * new_log_prob
            target_q_I_value = torch.min(self.target_q_I_1_net(next_states, next_state_actions),
                                   self.target_q_I_2_net(next_states, next_state_actions)) - alpha * new_log_prob # JI: now including double net

            q_target = self.reward_scale * rewards + (1.0 - dones) * self.gamma * target_q_value
            q_I_target = self.reward_scale * rewards_I + (1.0 - dones) * self.gamma * target_q_I_value # FIXED

        # Split losses by step type
        if nReg!=0:
            q1_loss = ( (self.q_1_net(states, actions) - q_target).pow(2)*(1-use_I) ).sum(dim=0)/nReg # mean without the zeros
            q2_loss = ( (self.q_2_net(states, actions) - q_target).pow(2)*(1-use_I) ).sum(dim=0)/nReg # mean without the zeros
        if nI!=0:
            q_I1_loss = ( (self.q_I_1_net(states, actions) - q_I_target).pow(2)*(use_I) ).sum(dim=0)/nI # mean without the zeros
            q_I2_loss = ( (self.q_I_2_net(states, actions) - q_I_target).pow(2)*(use_I) ).sum(dim=0)/nI # mean without the zeros

        """
        Update Networks
        """
        # update

        if nReg!=0: # Do not update if there were not regular Q1 samples
            self.q_1_net_optimizer.zero_grad()
            q1_loss.backward()
            self.q_1_net_optimizer.step()  

            self.q_2_net_optimizer.zero_grad()
            q2_loss.backward()
            self.q_2_net_optimizer.step()  
        if nI!=0: # Do not update with there were no I-samples
            self.q_I_1_net_optimizer.zero_grad()
            q_I1_loss.backward()
            self.q_I_1_net_optimizer.step()  

            self.q_I_2_net_optimizer.zero_grad()
            q_I2_loss.backward()
            self.q_I_2_net_optimizer.step()  

        self.policy_net_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_net_optimizer.step()  

        td_loss_1, td_loss_2 = None, None

        """
        Soft Updates
        """
        self.soft_update(self.q_1_net, self.target_q_1_net, self.tau)
        self.soft_update(self.q_2_net, self.target_q_2_net, self.tau)
        self.soft_update(self.q_I_1_net, self.target_q_I_1_net, self.tau)
        self.soft_update(self.q_I_2_net, self.target_q_I_2_net, self.tau)

        return policy_loss, alpha, alpha_I, (td_loss_1, td_loss_2), q_target.detach().cpu().data.numpy() # JI


    def learn_both(self, states, actions, rewards, rewards_I, use_I, next_states, dones, sample_weights, nsteps, alphaK=0.1):

        """ Update policies from batch of experience tuples. """
       
        # evaluate policy to get next actions, log probability, etc.
        new_actions, _, log_prob = self.policy_net(states, return_log_prob=True)
        
        """
        Alpha Loss
        """
        # See https://arxiv.org/pdf/1812.05905.pdf, section 5 explanation and see equation 17
        # Also https://github.com/vitchyr/rlkit/blob/2e8097f2a2c4d4d5079f12b8210f8ac8f250aee0/rlkit/torch/sac/sac.py#L99
        if self.use_automatic_entropy_tunning:
            alpha_loss = -(self.entropy_scaling * (log_prob + self.target_entropy).detach()).mean()
            self.entropy_scaling_optimizer.zero_grad()
            alpha_loss.backward()
            self.entropy_scaling_optimizer.step()
            alpha = self.entropy_scaling.exp()
        else:
            alpha = alphaK # 0.1
        
        """
        Policy Loss
        """
        # Inhibitory rule: Given use_I, switch between q1,q2 x q_I_1,q_I_2
        q12_min = torch.min(self.q_1_net(states, new_actions),self.q_2_net(states, new_actions))
        q_I_12_min = torch.min(self.q_I_1_net(states, new_actions),self.q_I_2_net(states, new_actions))
        predicted_new_q_values = (1.0 - use_I)*q12_min + use_I*q_I_12_min 
        
        policy_loss = (alpha * log_prob - predicted_new_q_values).mean() 

        """
        Q-function losses
        """
        with torch.no_grad():
            next_state_actions, _, new_log_prob = self.policy_net(next_states, return_log_prob=True)    
            target_q_value = torch.min(self.target_q_1_net(next_states, next_state_actions),
                                    self.target_q_2_net(next_states, next_state_actions)) - alpha * new_log_prob
            target_q_I_value = torch.min(self.target_q_I_1_net(next_states, next_state_actions),
                                   self.target_q_I_2_net(next_states, next_state_actions)) - alpha * new_log_prob # JI: now including double net

            q_target = self.reward_scale * rewards + (1.0 - dones) * self.gamma * target_q_value
            q_I_target = self.reward_scale * rewards_I + (1.0 - dones) * self.gamma * target_q_I_value # FIXED

        # get losses
        q1_loss = torch.mean((self.q_1_net(states, actions) - q_target).pow(2)) # JI
        q2_loss = torch.mean((self.q_2_net(states, actions) - q_target).pow(2)) # JI
        q_I1_loss = torch.mean((self.q_I_1_net(states, actions) - q_I_target).pow(2)) # JI
        q_I2_loss = torch.mean((self.q_I_2_net(states, actions) - q_I_target).pow(2)) # JI
        
        
        """
        Update Networks
        """
        # update
        self.q_1_net_optimizer.zero_grad()
        q1_loss.backward()
        self.q_1_net_optimizer.step()  

        self.q_2_net_optimizer.zero_grad()
        q2_loss.backward()
        self.q_2_net_optimizer.step()  

        self.q_I_1_net_optimizer.zero_grad()
        q_I1_loss.backward()
        self.q_I_1_net_optimizer.step()  

        self.q_I_2_net_optimizer.zero_grad()
        q_I2_loss.backward()
        self.q_I_2_net_optimizer.step()  

        self.policy_net_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_net_optimizer.step()  

        td_loss_1, td_loss_2 = None, None

        """
        Soft Updates
        """
        self.soft_update(self.q_1_net, self.target_q_1_net, self.tau)
        self.soft_update(self.q_2_net, self.target_q_2_net, self.tau)
        self.soft_update(self.q_I_1_net, self.target_q_I_1_net, self.tau)
        self.soft_update(self.q_I_2_net, self.target_q_I_2_net, self.tau)

        return policy_loss, alpha, (td_loss_1, td_loss_2), q_target.detach().cpu().data.numpy() # JI


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class SAC():
    """ Soft Actor-Critic  """

    def __init__(self, state_dim,
                       action_dim,
                       hidden_layer_size,
                       q_lr,
                       policy_lr,
                       reward_scale,
                       num_hidden_layers=1,
                       gamma=0.99,
                       tau=1e-2,
                       use_automatic_entropy_tunning=False,
                       target_entropy=None,
                       entropy_scaling_lr=None,
                       device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                       max_nstep_return=1):

        self.reward_scale = reward_scale
        self.tau = tau
        self.use_automatic_entropy_tunning = use_automatic_entropy_tunning
        self.target_entropy = target_entropy
        self.device = device

        self.max_nstep_return = max_nstep_return
        self.gammas = torch.zeros((max_nstep_return + 1, 1))
        for i_n in range(max_nstep_return + 1):
            self.gammas[i_n, :] = gamma ** i_n
        
        self.gamma = gamma # 

        # initialize networks
        self.q_1_net = QNetwork(state_dim, action_dim, hidden_layer_size, q_lr, num_hidden_layers=num_hidden_layers).to(device)
        self.q_2_net = QNetwork(state_dim, action_dim, hidden_layer_size, q_lr, num_hidden_layers=num_hidden_layers).to(device)
        self.target_q_1_net = QNetwork(state_dim, action_dim, hidden_layer_size, q_lr, num_hidden_layers=num_hidden_layers).to(device)
        self.target_q_2_net = QNetwork(state_dim, action_dim, hidden_layer_size, q_lr, num_hidden_layers=num_hidden_layers).to(device)

        #complete copy from local to target => tau=1.0
        self.soft_update(self.q_1_net, self.target_q_1_net, 1.0) 
        self.soft_update(self.q_2_net, self.target_q_2_net, 1.0)

        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_layer_size, policy_lr, device, num_hidden_layers=num_hidden_layers).to(device)

        if self.use_automatic_entropy_tunning:
            self.entropy_scaling = torch.zeros(1, requires_grad=True, device=device)
            self.entropy_scaling_optimizer = optim.Adam([self.entropy_scaling], lr=entropy_scaling_lr)

        ## define optimizers
        self.q_1_net_optimizer = optim.Adam(self.q_1_net.parameters(), lr=q_lr)
        self.q_2_net_optimizer = optim.Adam(self.q_2_net.parameters(), lr=q_lr)
        self.policy_net_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)


    def act(self, state, hidden_gru_state=None):
        """Returns actions for given state as per current policy."""

        state = torch.from_numpy(state).float().to(self.device)

        self.policy_net.eval()

        with torch.no_grad():

            action, std = self.policy_net(state)
            
        self.policy_net.train()

        return action.cpu().data.numpy(), hidden_gru_state, std.cpu().data.numpy()


    def learn(self, states, actions, rewards, next_states, dones, sample_weights, nsteps):
        
        """ Update policies from batch of experience tuples. """
       
        # evaluate policy to get next actions, log probability, etc.
        new_actions, _, log_prob = self.policy_net(states, return_log_prob=True)
        
        """
        Alpha Loss
        """
        # See https://arxiv.org/pdf/1812.05905.pdf, section 5 explanation and see equation 17
        # Also https://github.com/vitchyr/rlkit/blob/2e8097f2a2c4d4d5079f12b8210f8ac8f250aee0/rlkit/torch/sac/sac.py#L99
        if self.use_automatic_entropy_tunning:
            alpha_loss = -(self.entropy_scaling * (log_prob + self.target_entropy).detach()).mean()
            self.entropy_scaling_optimizer.zero_grad()
            alpha_loss.backward()
            self.entropy_scaling_optimizer.step()
            alpha = self.entropy_scaling.exp()
        else:
            alpha = 0.1
        
        """
        Policy Loss
        """
        predicted_new_q_values = torch.min(self.q_1_net(states, new_actions),
                                           self.q_2_net(states, new_actions))        
        policy_loss = (alpha * log_prob - predicted_new_q_values).mean() 

        """
        Q-function losses
        """
        with torch.no_grad():
            next_state_actions, _, new_log_prob = self.policy_net(next_states, return_log_prob=True)    
            target_q_value = torch.min(self.target_q_1_net(next_states, next_state_actions),
                                    self.target_q_2_net(next_states, next_state_actions)) - alpha * new_log_prob
            q_target = self.reward_scale * rewards + (1.0 - dones) * self.gamma * target_q_value

        # get losses
        q1_loss = torch.mean((self.q_1_net(states, actions) - q_target).pow(2)) #
        q2_loss = torch.mean((self.q_2_net(states, actions) - q_target).pow(2)) # 
        
        """
        Update Networks
        """
        # update
        self.q_1_net_optimizer.zero_grad()
        q1_loss.backward()
        self.q_1_net_optimizer.step()  

        self.q_2_net_optimizer.zero_grad()
        q2_loss.backward()
        self.q_2_net_optimizer.step()  

        self.policy_net_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_net_optimizer.step()  

        td_loss_1, td_loss_2 = None, None

        """
        Soft Updates
        """
        self.soft_update(self.q_1_net, self.target_q_1_net, self.tau)
        self.soft_update(self.q_2_net, self.target_q_2_net, self.tau)

        return policy_loss, alpha, (td_loss_1, td_loss_2), q_target.detach().cpu().data.numpy() # JI


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class QNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size, lr, num_hidden_layers=1, init_w=3e-3):
        super(QNetwork, self).__init__()
        
        self.num_hidden_layers = num_hidden_layers

        self.linear0 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear0.weight.data.uniform_(-init_w, init_w)
        self.linear0.bias.data.uniform_(-init_w, init_w)

        self.hidden_linear_layers = nn.ModuleList()
        for i_layer in range(self.num_hidden_layers):
            self.hidden_linear_layers.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_linear_layers[i_layer].weight.data.uniform_(-init_w, init_w)
            self.hidden_linear_layers[i_layer].bias.data.uniform_(-init_w, init_w)    

        self.linear2 = nn.Linear(hidden_size, 1)
        self.linear2.weight.data.uniform_(-init_w, init_w)
        self.linear2.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear0(x))
        for i_layer in range(self.num_hidden_layers):
            x = F.relu(self.hidden_linear_layers[i_layer](x))
        x = self.linear2(x)
        return x

class PolicyNetwork(nn.Module):

    def __init__(self, num_states, num_actions, hidden_size, lr, device, num_hidden_layers=1,
                 init_w=3e-3, log_std_min=-20, log_std_max=2, epsilon=1e-6):
        super(PolicyNetwork, self).__init__()
        
        self.device = device

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.epsilon = epsilon

        self.num_hidden_layers = num_hidden_layers
        
        self.linear0 = nn.Linear(num_states, hidden_size)
        self.linear0.weight.data.uniform_(-init_w, init_w)
        self.linear0.bias.data.uniform_(-init_w, init_w)
        
        self.hidden_linear_layers = nn.ModuleList()
        for i_layer in range(self.num_hidden_layers):
            self.hidden_linear_layers.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_linear_layers[i_layer].weight.data.uniform_(-init_w, init_w)
            self.hidden_linear_layers[i_layer].bias.data.uniform_(-init_w, init_w)    

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)


    def forward(self, state, deterministic=False, return_log_prob=False):

        self.state = state

        x = F.relu(self.linear0(state))
        for i_layer in range(self.num_hidden_layers):
            x = F.relu(self.hidden_linear_layers[i_layer](x))
        mean = self.mean_linear(x)

        if deterministic:
            action = torch.tanh(mean) 
            return action
        else:
            log_std = self.log_std_linear(x)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            std = log_std.exp()

            normal = Normal(mean, std)
            # Reparametrization trick (to allow backprop of derivatives with distribution) is implemented through .rsample()
            # REF: https://pytorch.org/docs/stable/distributions.html
            x_t = normal.rsample() 
            action = torch.tanh(x_t)

            if return_log_prob:
                log_prob = normal.log_prob(x_t)
                # Enforcing Action Bound
                log_prob -= torch.log(1 * (1 - action.pow(2)) + self.epsilon)
                log_prob = log_prob.sum(1, keepdim=True)
                
                return action, std, log_prob
            else:
                return action, std

