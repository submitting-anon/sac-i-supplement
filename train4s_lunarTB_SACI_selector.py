##################################################################################
# 4s) Script to train SAC-I agent (without shaping and with automated inhibitory networks)
# - Env: LunarLanderContinuous-v2
# - Reward: 
#     r_R = r_time + r_bomb + r_env
#     r_I = r_time + r_bomb + r_env 
# - State: Add bomb coordinates if it appears
# - Inhibition rule: use a learned/automated selector (of Q-networks)
# - Q-I weights: trained from scratch
# - Options:
#   - Use STOP_TRIAL_PCT to set the stop trial percentage. For example STOP_TRIAL_PCT=0.5 is the default
# - Obs:
#    - NO shaping (reward_I = reward + r_bomb) -->  almost like SAC without shaping but we keep the original Q and train the Q-I
#    - load Q-I from Q-R
##################################################################################

import cProfile
import numpy as np
import pickle
import torch
import sys
import os
import gym
import time

from collections import deque
from copy import deepcopy
from os import path
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

# selector specific
from replaybuffer import ReplayBuffer_I_S 
from sac_vanilla import SAC_I_S, PolicyNetwork, create_base_dir, load_model_weights, ScoreBoard_bombs_S, reward_shapedVVHA_only
      
from utils import uniform_value_from_range, dist_euclidean #
from lunar_lander_withBomb import LunarLanderContinuous

####################################################
# Set global variables
#
STOP_TRIAL_PCT = None  # (Set in command line) Valid values: 0.25, 0.5, 0.75
####################################################

class PolicyNetworkType():
    STANDARD = 0
    GRU = 1

class MultiAgentSACTrainer:

    def __init__(self, env_name=None, num_hidden_layers=None, hidden_layer_size=None, batch_size=None, update_n_steps=None, update_n_times=None,
                       q_lr=None, policy_lr=None, soft_tau=None, reward_scale=None, entropy_scaling_lr=None, 
                       use_automatic_entropy_tunning=True, replay_buffer_size=None, num_episodes_before_training=None,
                       num_episodes_random_actions=None, max_num_episodes=None,
                       load_weights=None, load_weights_Q_I=None, save_path=None, 
                       log_interval_num_episodes=None, use_tensorboard=None, run_tag=None, policy_network_type=None, 
                       nback=None, seed=None, render=None):

        ## Prepare environment
        self.seed = seed
        self.env_name = env_name
        
        ######################################################################
        self.env0 = gym.make(self.env_name)
        self.env = LunarLanderContinuous() # USE LOCAL ENV
        self.env._max_episode_steps = self.env0._max_episode_steps # get info from the original env
        ######################################################################

        self.state_dim = self.env.observation_space.shape[0] + 4 # ADD BOMB COORDINATE STATES
        self.action_dim = self.env.action_space.shape[0] # Continuous (scoreboard)
        
        self.action_dim_S = 1  # selector action dimension: regular or inhibitory (after 809b)

        self.max_num_steps_per_episode = self.env._max_episode_steps
        self.render = render
        
        # Hyperparameters
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.batch_size = batch_size
        self.update_n_steps = update_n_steps
        self.update_n_times = update_n_times
        self.q_lr = q_lr
        self.policy_lr = policy_lr
        self.soft_tau = soft_tau
        self.entropy_scaling_lr = entropy_scaling_lr
        self.reward_scale = reward_scale
        self.use_automatic_entropy_tunning = use_automatic_entropy_tunning
        self.replay_buffer_size = replay_buffer_size
        self.num_episodes_before_training = num_episodes_before_training
        self.num_episodes_random_actions = num_episodes_random_actions
        self.policy_network_type = policy_network_type
        self.nback = nback

        # Training script parameters        
        self.max_num_episodes = max_num_episodes
        self.load_weights = load_weights
        self.log_interval_num_episodes = log_interval_num_episodes
        
        # Paths and others
        self.use_tensorboard = use_tensorboard
        self.run_tag = run_tag
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.save_path = save_path
        self.save_weights_path = path.join(self.save_path,self.env_name,'0_weights', self.run_tag+'_seed%d'%args.seed)
        self.save_path_tb = path.join(self.save_path,self.env_name,'0_runs', self.run_tag+'_seed%d'%args.seed)

        # Set seeds (it works for the env. but not for torch actor...)
        # Ref: https://discuss.pytorch.org/t/random-seed-initialization/7854/20 
        self.env.seed(seed)
        self.seed = seed
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.random.manual_seed(seed)
        import random
        random.seed(seed)

        ################### (balanced number of stop trials) ##########################
        # Once seed is set, create an unique sequence of GO/STOP trials (50%/50%): shuffle the array [0..0, 1..1]
        if STOP_TRIAL_PCT == 0.5:
            a2000 = np.concatenate((np.zeros(1000),np.ones(1000)))
            a3000 = np.concatenate((np.zeros(1500),np.ones(1500)))
        elif STOP_TRIAL_PCT == 0.25:
            a2000 = np.concatenate((np.zeros(1500),np.ones(500)))
        elif STOP_TRIAL_PCT == 0.75:
            a2000 = np.concatenate((np.zeros(500),np.ones(1500)))
            
        # random.shuffle(a2000)
        # self.vstop = a2000
        random.shuffle(a3000)
        self.vstop = a3000
        print('- OBS: stop trials are shuffled equally across 2000 episodes.')
        ##############################################################################
        
        self.algo_name = 'sac_hidden_{:d}_rscale_{:d}_s{:d}_u{:d}'.format(self.hidden_layer_size, 
                                                                                    self.reward_scale,
                                                                                    self.update_n_steps,
                                                                                    self.update_n_times)

        # Used in SAC.learn()
        self.gpu_nsteps = torch.zeros((self.batch_size, 1), requires_grad=False)#.to(self.learner_device)
        self.gpu_sample_importance_weights = torch.zeros((self.batch_size, 1), requires_grad=False).to(self.device)

        # init dictionaries
        self.replay_buffers = {}
        self.sac = {}
        self.q_1_nets_save_weights_path = {}
        self.q_2_nets_save_weights_path = {}
        self.target_q_1_nets_save_weights_path = {}
        self.target_q_2_nets_save_weights_path = {}
        self.q_I_1_nets_save_weights_path = {}
        self.q_I_2_nets_save_weights_path = {}
        self.target_q_I_1_nets_save_weights_path = {}
        self.target_q_I_2_nets_save_weights_path = {}
        self.policy_nets_save_weights_path = {}

        # Selector
        self.q_S_1_nets_save_weights_path = {}
        self.target_q_S_1_nets_save_weights_path = {}
        self.q_S_2_nets_save_weights_path = {}
        self.target_q_S_2_nets_save_weights_path = {}
        self.selector_nets_save_weights_path = {}

        if self.use_automatic_entropy_tunning:
            self.entropy_scaling_save_weights_path = {}
    
        # initialize replay buffers
        self.replay_buffers = ReplayBuffer_I_S(self.replay_buffer_size) # selector

        if use_automatic_entropy_tunning:
            # Heuristic value for target entropy lower bound used in entropy scaling                
            self.target_entropy = -np.prod(self.action_dim).item()
        else:
            self.target_entropy = -np.prod(self.action_dim).item()

        # initialize networks
        self.sac = SAC_I_S(self.state_dim, 
                            self.action_dim,
                            self.action_dim_S, # selector action dimension
                            self.hidden_layer_size,
                            self.q_lr,
                            self.policy_lr,
                            self.reward_scale,
                            num_hidden_layers=self.num_hidden_layers,
                            tau=self.soft_tau,
                            use_automatic_entropy_tunning=self.use_automatic_entropy_tunning,
                            target_entropy=self.target_entropy,
                            entropy_scaling_lr=self.entropy_scaling_lr,
                            device=self.device,
                            max_nstep_return=1)  
    
        # weights paths
        self.q_1_nets_save_weights_path = path.join(self.save_weights_path, '__sac_q_1_weights_%s.weights' % (self.algo_name))
        self.q_2_nets_save_weights_path = path.join(self.save_weights_path, '__sac_q_2_weights_%s.weights' % (self.algo_name))
        self.target_q_1_nets_save_weights_path = path.join(self.save_weights_path, '__sac_target_q_1_weights_%s.weights' % (self.algo_name))
        self.target_q_2_nets_save_weights_path = path.join(self.save_weights_path, '__sac_target_q_2_weights_%s.weights' % (self.algo_name))
        
        self.q_I_1_nets_save_weights_path = path.join(self.save_weights_path, '__sac_q_I_1_weights_%s.weights' % (self.algo_name))
        self.q_I_2_nets_save_weights_path = path.join(self.save_weights_path, '__sac_q_I_2_weights_%s.weights' % (self.algo_name))
        self.target_q_I_1_nets_save_weights_path = path.join(self.save_weights_path, '__sac_target_q_I_1_weights_%s.weights' % (self.algo_name))
        self.target_q_I_2_nets_save_weights_path = path.join(self.save_weights_path, '__sac_target_q_I_2_weights_%s.weights' % (self.algo_name))

        # selector weight paths
        self.q_S_1_nets_save_weights_path = path.join(self.save_weights_path, '__sac_q_S_1_weights_%s.weights' % (self.algo_name))
        self.target_q_S_1_nets_save_weights_path = path.join(self.save_weights_path, '__sac_target_q_S_1_weights_%s.weights' % (self.algo_name))
        self.q_S_2_nets_save_weights_path = path.join(self.save_weights_path, '__sac_q_S_2_weights_%s.weights' % (self.algo_name))
        self.target_q_S_2_nets_save_weights_path = path.join(self.save_weights_path, '__sac_target_q_S_2_weights_%s.weights' % (self.algo_name))
        self.selector_nets_save_weights_path = path.join(self.save_weights_path, '__sac_selector_weights_%s.weights' % (self.algo_name))

        self.policy_nets_save_weights_path = path.join(self.save_weights_path, '__sac_policy_weights_%s.weights' % (self.algo_name))
        if self.use_automatic_entropy_tunning:
            self.entropy_scaling_save_weights_path = path.join(self.save_weights_path, '__sac_entropy_scaling_weights_%s.weights' % (self.algo_name))

        create_base_dir(self.policy_nets_save_weights_path)
        if self.load_weights:
            load_model_weights(self.sac.q_1_net, self.q_1_nets_save_weights_path)
            load_model_weights(self.sac.target_q_1_net, self.target_q_1_nets_save_weights_path)
            load_model_weights(self.sac.q_2_net, self.q_2_nets_save_weights_path)
            load_model_weights(self.sac.target_q_2_net, self.target_q_2_nets_save_weights_path)

            if load_weights_Q_I:
                print('*** Q_I networks loaded! ***')
                load_model_weights(self.sac.q_I_1_net, self.q_I_1_nets_save_weights_path)
                load_model_weights(self.sac.target_q_I_1_net, self.target_q_I_1_nets_save_weights_path)
                load_model_weights(self.sac.q_I_2_net, self.q_I_2_nets_save_weights_path)
                load_model_weights(self.sac.target_q_I_2_net, self.target_q_I_2_nets_save_weights_path)
            
            load_model_weights(self.sac.policy_net, self.policy_nets_save_weights_path)
            
            if self.use_automatic_entropy_tunning:
                # Loading entropy scaling here since it is a scalar
                if path.exists(self.entropy_scaling_save_weights_path):
                    self.sac.entropy_scaling = torch.load(self.entropy_scaling_save_weights_path)
                    self.sac.entropy_scaling_optimizer = torch.optim.Adam([self.sac.entropy_scaling], lr=self.entropy_scaling_lr)
                    print('Weights Loaded:', self.entropy_scaling_save_weights_path)
                else:
                    print('No weights file found!')
                    print(self.entropy_scaling_save_weights_path)
        else:
            print('Weights Reset!')
            
        if self.use_tensorboard:
            self.writer = SummaryWriter(self.save_path_tb)

        # Scoreboard
        self.scoreboard = ScoreBoard_bombs_S(self.save_path_tb)

        # Store last alpha
        self.last_alpha =   0.007 # initialize with default
        self.last_alpha_I = 0.1

    def update(self, batch_size, update_selector = True):

        # grab batch of experiences
        if self.policy_network_type == PolicyNetworkType.STANDARD:
            states, actions, actions_S, rewards, rewards_I, use_I, next_states, dones = self.replay_buffers.sample(self.batch_size) # selector action from replay
            state_trajs = None
        
        elif self.policy_network_type == PolicyNetworkType.GRU:
            states, actions, rewards, rewards_I, next_states, use_I, dones, state_trajs, _ = self.replay_buffers.sample(self.batch_size)
    
        alphaK = None
        if update_selector:
            policy_losses, alphas, alphas_I, alpha_S, td_errors, q_targets = self.sac.learn_R_I_S_splitAlpha_splitQloss_limitA_x(
                                                            torch.FloatTensor(states).to(self.device),
                                                            torch.FloatTensor(actions).to(self.device),
                                                            torch.FloatTensor(actions_S).to(self.device), # selector actions
                                                            torch.FloatTensor(rewards).to(self.device).unsqueeze(1),
                                                            torch.FloatTensor(rewards_I).to(self.device).unsqueeze(1),
                                                            torch.FloatTensor(use_I).to(self.device).unsqueeze(1),
                                                            torch.FloatTensor(next_states).to(self.device),
                                                            torch.FloatTensor(dones).to(self.device).unsqueeze(1),
                                                            self.gpu_sample_importance_weights,
                                                            self.gpu_nsteps,
                                                            self.last_alpha,
                                                            self.last_alpha_I)
        else:
            policy_losses, alphas, alphas_I, td_errors, q_targets = self.sac.learn_I_splitAlpha_splitQloss_limitA_x(
                                                            torch.FloatTensor(states).to(self.device),
                                                            torch.FloatTensor(actions).to(self.device),
                                                            torch.FloatTensor(rewards).to(self.device).unsqueeze(1),
                                                            torch.FloatTensor(rewards_I).to(self.device).unsqueeze(1),
                                                            torch.FloatTensor(use_I).to(self.device).unsqueeze(1),
                                                            torch.FloatTensor(next_states).to(self.device),
                                                            torch.FloatTensor(dones).to(self.device).unsqueeze(1),
                                                            self.gpu_sample_importance_weights,
                                                            self.gpu_nsteps,
                                                            self.last_alpha,
                                                            self.last_alpha_I)
            alpha_S = 0

        return policy_losses, alphas, alphas_I, alpha_S, q_targets


    def train(self):
        t0 = time.time()

        nUpdates = 0
        ready_to_train = False
        use_policy_action = False
        nhits = 0
        nbombs = 0
        nlandings = 0

        average_alpha = 0 # bomb
        average_loss = 0  # bomb
        count_stop_trial = 0

        num_steps_per_episode_list = deque(maxlen=self.log_interval_num_episodes)
        list_rewards = deque(maxlen=100)
        list_epilen = deque(maxlen=100) # scoreboard

        # last rewards
        last5_rewards = deque(maxlen=5)

        alpha_list = []
        alpha_list_I = []
        alpha_list_S = [] # selector alphas
        loss_list = []

        epi_min = 100 # burn-in phase before training the selector
        
        try:
            current_total_num_timesteps = 0
            episode = 0
            while current_total_num_timesteps <= 6e5: # update 807

                ## 50% chance to bomb appear (shuffled balanced vector) #######################
                if self.vstop[episode]==1:
                    stop_trial = True
                    isBomb = 1
                else:
                    stop_trial = False
                    isBomb = 0
                    
                if stop_trial:
                    # Initialize bomb coordinates (use a fixed box for now)
                    x0 = uniform_value_from_range(-0.2, 0.2)
                    y0 = uniform_value_from_range(0.1, 0.5)
                    x_bomb = [x0-0.1,x0+0.1] # same size as the fixed box
                    y_bomb = [y0-0.05,y0+0.05]
                    count_stop_trial += 1
                else:
                    x_bomb = [-2,-2]
                    y_bomb = [-2,-2]
                # bomb center coordinate
                xc2appear = (x_bomb[0]+x_bomb[1])/2
                yc2appear = (y_bomb[0]+y_bomb[1])/2
                #####################################################################

                rewards_list_episode = []
                
                if episode >= self.num_episodes_random_actions and not use_policy_action:
                    print("Now using policy for actions...\n")
                    use_policy_action = True

                if episode >= self.num_episodes_before_training and not ready_to_train:
                    print("Now training...\n\n", end='')
                    ready_to_train = True

                print('--- Episode {:5d} (seed: {:2d})-----------------------'.format(episode,self.seed))
                print('  Bomb coordinates x: [%2.2f,%2.2f]'%(x_bomb[0],x_bomb[1]))
                print('  Bomb coordinates y: [%2.2f,%2.2f]'%(y_bomb[0],y_bomb[1]))

                # Initialize
                state = self.env.reset()
                # ADD DUMMY BOMB COORDINATES IN THE STATES
                state = np.concatenate((state,-2*np.ones(4))) # DONT TELL AT BEGINNING...
                
                state_original = state.copy() # 

                if self.render:
                    self.env.render()

                cum_return = 0
                cum_return_I = 0
                cum_return_raw = 0
                count_I = 0
                alpha = 0
                alpha_I = 0
                alpha_S = 0 # selector alpha initialize
                actor_loss = 0
                avg_q_target = 0
                cum_a_std = 0
                epi_num_updates = 0
                success = False # LunarLander
                k_dist = 0
                use_I = False
                max_angle = 0

            
                ''' step loop '''
                aux = ['/','-','\\','|']
                bombAppeared = False
                isClose = False
                r_too_close = 0

                # Auxiliary vars to keep track of the current raw comulative reward
                last_use_I = 0
                last_nonI_cum_reward = 0
                
                # 809b
                action_S = [-0.5] # selector initialize (negative is for Regular) 

                for step_num in range(self.max_num_steps_per_episode):
                    # Bomb appears after y<1+noise if STOP trial
                    xc = -2 # default (GO trial)
                    yc = -2
                    if stop_trial and ((state[1] < 1+uniform_value_from_range(-0.1,0.1)) or bombAppeared):
                        xc = xc2appear
                        yc = yc2appear
                        bombAppeared = True

                    # selector info
                    if use_I:
                        whichQ = 'I'
                    else:
                        whichQ = 'R'
                    print('  Step {:4d}  {:s}  {:s} \r'.format(step_num, whichQ, aux[step_num%4]),end="",flush=True)

                    if use_policy_action:
                        action, a_std = self.sac.policy_net(torch.FloatTensor(state).to(self.device))
                        action = action.cpu().data.numpy()
                        a_std = a_std.cpu().data.numpy()
                        
                    else:
                        action = np.random.uniform(low=-1.0, high=1.0, size=self.action_dim)
                        a_std = 0
                    
                    # STEP #
                    next_state, reward, done, _ = self.env.step(action)
    
                    ## BOMB update ###############################################
                    # Check if it hit the bomb
                    x_next = next_state[0]
                    y_next = next_state[1]

                    if (x_next>x_bomb[0]) and (x_next<x_bomb[1]) and (y_next>y_bomb[0]) and (y_next<y_bomb[1]):
                        print('  /\/\/\/\/\/\/\/\/\/')
                        print('< Boooom!! Exploded!! >')
                        print('  \/\/\/\/\/\/\/\/\/\/')
                        reward_bomb = -150 # bomb-bomb3 
                        nbombs += 1
                        done = True
                    else:
                        reward_bomb = 0.0 # same as the Nov13


                    # Reward Shaping ###############################################
                    
                    # original reward
                    reward0 = reward # keep the original for check done status                    
                    reward += -0.1 #  time penalty: in 1000 steps will get -100

                    # Q_I: inhibitory reward
                    reward_I = reward + reward_bomb # original + sparse bomb reward

                    ## Use selector to decide to use inhibitory net ################### 
                    step2update = 10
                    if episode > epi_min: # burn-in phase
                        # update every n-step if min episode was achieved
                        if ((step_num+1) %  step2update == 0) and (episode > epi_min):
                            action_S, a_std_S = self.sac.selector_net(torch.FloatTensor(state).to(self.device))
                            action_S = action_S.cpu().data.numpy()
                    else:
                        action_S = [-0.5]  # 
                        
                    if action_S[0] > 0:
                        option = 1 # inhibitory
                    else:
                        option = 0 # regular

                    if option==1:
                        use_I = 1
                    else:
                        use_I = 0
                    
                    if bombAppeared:
                        next_state = np.concatenate((next_state,x_bomb))
                        next_state = np.concatenate((next_state,y_bomb))
                    else:
                        next_state = np.concatenate((next_state,-2*np.ones(4)))

                    current_total_num_timesteps += 1
                    cum_return += reward
                    cum_return_I += reward_I
                    cum_return_raw += reward0
                    count_I += use_I
                    cum_a_std += a_std
                    last5_rewards.append(reward)
                    if self.render:
                        self.env.render_withBomb(xc,yc) # appear bomb
                    ###########################################
                    
                    # push to replay buffer and append to lists for stats
                    done01 = 0.0 if step_num+1 == self.max_num_steps_per_episode else float(done)
                    if self.policy_network_type == PolicyNetworkType.STANDARD:
                        nfactor = 10
                        self.replay_buffers.push(state, action, action_S, reward/nfactor, reward_I/nfactor, use_I, next_state, done01) # selector action included
                        
                    
                    # update networks
                    if ready_to_train:
                        for _ in range(self.update_n_times):
                            ## LEARN(UPDATE)
                            if episode > 0: # 
                                loss_policy, alpha_updated, alpha_updated_I, alpha_updated_S, q_target = self.update(self.batch_size, True) # selector alpha
                            else:
                                loss_policy, alpha_updated, alpha_updated_I, alpha_updated_S, q_target = self.update(self.batch_size, False) # do not update selector


                            if self.use_automatic_entropy_tunning:
                                alpha = alpha_updated
                                
                                if torch.is_tensor(alpha_updated_I):
                                    alpha_I = alpha_updated_I.cpu().detach().numpy()
                                    alpha_I = alpha_I[0]
                                else:
                                    alpha_I = alpha_updated_I    
                                
                                # selector alpha
                                if torch.is_tensor(alpha_updated_S):
                                    alpha_S = alpha_updated_S.cpu().detach().numpy()
                                    alpha_S = alpha_S[0]
                                else:
                                    alpha_S = alpha_updated_S    

                            else:
                                alpha = alpha_updated
                                alpha_I = alpha_updated_I
                                alpha_S = alpha_updated_S # selector

                            loss_list.append(loss_policy.cpu().detach().numpy() / self.reward_scale)
                            alpha_list.append(alpha)
                            alpha_list_I.append(alpha_I)
                            alpha_list_S.append(alpha_S) # selector alpha to the list
                            actor_loss = loss_policy
                            nUpdates += 1
                            epi_num_updates += 1

                            # Nov25: store last alphas
                            self.last_alpha = alpha
                            self.last_alpha_I = alpha_I
                            self.last_alpha_S = alpha_S # selector

                    # Episode averages
                    avg_a_std = cum_a_std/(step_num+1)
                    if done:
                        
                        if reward0 == -100:
                            print('  /\/\/\/\/')
                            print('< Deck hit!! >')
                            print('  \/\/\/\/\/')
                            nhits += 1
                        if reward0 == 100:
                            print('*********************************')
                            print('** Safely landed: Well done!! ***')
                            print('*********************************')
                            nlandings += 1

                        if ready_to_train:
                            end_loss = actor_loss # /epi_num_updates
                            avg_q_target = q_target.mean() 
                        else:
                            end_loss = -0.0
                            avg_q_target = -0.0
                        break

                    # Prepare next step
                    state = next_state

            

                # save number of steps of episode
                num_steps_per_episode_list.append(step_num)
                list_rewards.append(cum_return)
                list_epilen.append(step_num) # scoreboard

                log_interval = (episode - self.num_episodes_before_training) % self.log_interval_num_episodes
                tt = int(time.time()-t0)
                # selector alpha added
                print("--- Ep {} (ep.len= {}, Timesteps={:7d}, alpha={:.3f}, alpha_I={:.3f}, alpha_S={:.3f}, loss={:.3f}):  Ep.Reward:{:4.2f}  Ep.Reward_I:{:4.2f} Avg100.Reward:{:4.2f} Time: {:02}:{:02}:{:02}".format(episode, step_num,current_total_num_timesteps,alpha, alpha_I, alpha_S, end_loss,cum_return,cum_return_I, sum(list_rewards)/len(list_rewards),tt//3600,tt%3600//60,tt%60))
                print('Last action:', action)
                print('Avg action_std:', avg_a_std)
                print('Buffer size current:',len(self.replay_buffers))
                print('Last batch.avg.q_target:', avg_q_target)
                print('Total I-steps Percent: %2.3f'%(count_I/step_num))
                print('Too close penalty (k_dist=%1.2f, max_angle=%d): %2.3f'%(k_dist,max_angle,r_too_close))
                print('Stop trials: %d'%count_stop_trial)
                print('Num_Landings:', nlandings,'\t Num Ground Hits:',nhits,'\t Num Bomb Hits:',nbombs) # scoreboard

                if ready_to_train and log_interval == 0:

                    average_num_steps_per_episode = int(np.mean(num_steps_per_episode_list))
                    num_steps_per_episode_list = []

                    total_reward = cum_return
                    average_alpha = np.mean(alpha_list)
                    average_alpha_I = np.mean(alpha_list_I)
                    average_alpha_S = np.mean(alpha_list_S) # selector
                    average_loss = np.mean(loss_list) / self.reward_scale
                    
                    if episode % 2 == 0:
                        suffix = '_backup'
                    else:
                        suffix = ''

                    create_base_dir(self.q_1_nets_save_weights_path)
                    torch.save(self.sac.q_1_net.state_dict(), path.join(self.q_1_nets_save_weights_path + suffix))
                    create_base_dir(self.q_2_nets_save_weights_path)
                    torch.save(self.sac.q_2_net.state_dict(), path.join(self.q_2_nets_save_weights_path + suffix))
                    create_base_dir(self.target_q_1_nets_save_weights_path)
                    torch.save(self.sac.target_q_1_net.state_dict(), path.join(self.target_q_1_nets_save_weights_path + suffix))
                    create_base_dir(self.target_q_2_nets_save_weights_path)
                    torch.save(self.sac.target_q_2_net.state_dict(), path.join(self.target_q_2_nets_save_weights_path + suffix))

                    create_base_dir(self.q_I_1_nets_save_weights_path)
                    torch.save(self.sac.q_I_1_net.state_dict(), path.join(self.q_I_1_nets_save_weights_path + suffix))
                    create_base_dir(self.q_I_2_nets_save_weights_path)
                    torch.save(self.sac.q_I_2_net.state_dict(), path.join(self.q_I_2_nets_save_weights_path + suffix))
                    create_base_dir(self.target_q_I_1_nets_save_weights_path)
                    torch.save(self.sac.target_q_I_1_net.state_dict(), path.join(self.target_q_I_1_nets_save_weights_path + suffix))
                    create_base_dir(self.target_q_I_2_nets_save_weights_path)
                    torch.save(self.sac.target_q_I_2_net.state_dict(), path.join(self.target_q_I_2_nets_save_weights_path + suffix))

                    # SELECTOR save weights
                    create_base_dir(self.q_S_1_nets_save_weights_path)
                    torch.save(self.sac.q_S_1_net.state_dict(), path.join(self.q_S_1_nets_save_weights_path + suffix))
                    create_base_dir(self.target_q_S_1_nets_save_weights_path)
                    torch.save(self.sac.target_q_S_1_net.state_dict(), path.join(self.target_q_S_1_nets_save_weights_path + suffix))
                    create_base_dir(self.q_S_2_nets_save_weights_path)
                    torch.save(self.sac.q_S_2_net.state_dict(), path.join(self.q_S_2_nets_save_weights_path + suffix))
                    create_base_dir(self.target_q_S_2_nets_save_weights_path)
                    torch.save(self.sac.target_q_S_2_net.state_dict(), path.join(self.target_q_S_2_nets_save_weights_path + suffix))
                    create_base_dir(self.selector_nets_save_weights_path)
                    torch.save(self.sac.selector_net.state_dict(), path.join(self.selector_nets_save_weights_path + suffix))

                    create_base_dir(self.policy_nets_save_weights_path)
                    torch.save(self.sac.policy_net.state_dict(), path.join(self.policy_nets_save_weights_path + suffix))

                    if self.use_automatic_entropy_tunning:
                        create_base_dir(self.entropy_scaling_save_weights_path)
                        torch.save(self.sac.entropy_scaling, path.join(self.entropy_scaling_save_weights_path + suffix))

                    if self.use_tensorboard:
                        ts = current_total_num_timesteps
                        self.writer.add_scalars('0_TotalReward_x_Timesteps', {'SAC_%s' % (self.algo_name): total_reward}, ts)
                        self.writer.add_scalars('0_Num.Hits_x_Timesteps', {'SAC_%s' % (self.algo_name): nhits}, ts)
                        self.writer.add_scalars('0_Num.Bomb.Hits_x_Timesteps', {'SAC_%s' % (self.algo_name): nbombs}, ts)
                        self.writer.add_scalars('0_Num.Landings_x_Timesteps', {'SAC_%s' % (self.algo_name): nlandings}, ts)
                        self.writer.add_scalars('0_PctInhibitoryUse_x_Timesteps', {'SAC_%s' % (self.algo_name): (count_I/step_num)}, ts)
                        self.writer.add_scalars('1_AvgActorLoss_x_Timesteps', {'SAC_%s' % (self.algo_name): average_loss}, ts)
                        self.writer.add_scalars('2_AvgAlpha_x_Timesteps', {'SAC_%s' % (self.algo_name): average_alpha}, ts)
                        self.writer.add_scalars('2_AvgAlpha_I_x_Timesteps', {'SAC_%s' % (self.algo_name): average_alpha_I}, ts)
                        self.writer.add_scalars('2_AvgAlpha_S_x_Timesteps', {'SAC_%s' % (self.algo_name): average_alpha_S}, ts) # selector
                        self.writer.add_scalars('3_Ep.Length_x_Timesteps', {'SAC_%s' % (self.algo_name): step_num}, ts)
                        self.writer.add_scalars('4_Episodes_x_Timesteps', {'SAC_%s' % (self.algo_name): episode}, ts)
                        self.writer.add_scalars('5_Updates_x_Timesteps', {'SAC_%s' % (self.algo_name): nUpdates}, ts)

                    # Update scoreboard
                    if episode % 1 ==0:
                        # selector: add alpha_S
                        self.scoreboard.update_data_frame(current_total_num_timesteps,episode,average_alpha,average_alpha_I,average_alpha_S,average_loss,nUpdates,list_rewards,list_epilen,nhits,nbombs,nlandings,(count_I/step_num),isBomb)
                    # save scoreboard
                    if episode % 10 ==0:
                        self.scoreboard.save_df()
                        
                alpha_list = []
                alpha_list_I = []
                alpha_list_S = []
                loss_list = []
                episode+= 1 # update 807
            
            # selector: add alpha_S
            self.scoreboard.update_data_frame(current_total_num_timesteps,episode,average_alpha,average_alpha_I,average_alpha_S,average_loss,nUpdates,list_rewards,list_epilen,nhits,nbombs,nlandings,(count_I/step_num),isBomb)
            self.scoreboard.save_df()
                
        except (KeyboardInterrupt, SystemExit):
            print("Exiting...")
            # selector: add alpha_S
            self.scoreboard.update_data_frame(current_total_num_timesteps,episode,average_alpha,average_alpha_I,average_alpha_S,average_loss,nUpdates,list_rewards,list_epilen,nhits,nbombs,nlandings,(count_I/step_num),isBomb)
            self.scoreboard.save_df()
            self.env.close()
            return

if __name__ == '__main__':

    import argparse
    import os
    import shutil
    
    #########################################################################################################
    # inputs
    #########################################################################################################

    config = {'render' : False,
              'render_mode' : 'dis'}

    num_hidden_layers = 1
    hidden_layer_size = 256
    
    policy_lr = 5e-4
    q_lr = 5e-4
    entropy_scaling_lr = 5e-4
    
    soft_tau = 1e-3 #5e-3

    use_automatic_entropy_tunning =  True
    
    reward_scale = 1

    batch_size = 64 #

    update_n_steps = 1
    update_n_times = 1

    replay_buffer_size = 1000000

    num_episodes_before_training = 50
    num_episodes_random_actions = 5

    max_num_episodes = int(2e3)

    load_weights = True
    load_weights_Q_I = True

    render = True #False
    save_path = 'models'
    trained_model_path = 'models_trained'

    
    use_tensorboard = True

    log_interval_num_episodes = 1

    ######################
    # SAC Policy Specific
    ######################
    policy_network_type = 0
    nback = None
    if policy_network_type == 1: # GRU
        nback = 20
    #########################################################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', dest='run_tag', type=str, default='no_tag', help="Tag name for run")
    parser.add_argument('-i', dest='max_timesteps', type=int, default=1000000, help="Maximum number of time steps")
    parser.add_argument('--env', dest='env_name', type=str, default='LunarLanderContinuous-v2', help="Environment name")
    parser.add_argument('--seed', dest='seed', type=int, default=7, help="Seed for the env initialization")
    parser.add_argument('--quick', dest='test_mode', action='store_true', help="Quick test mode")
    parser.add_argument('--stopPct', dest='stop_pct', type=float, default=0.5, help="Use stop pct: 0.25, 0.5, 0.75")
    args = parser.parse_args()
    # UPDATE if given
    STOP_TRIAL_PCT = args.stop_pct 

  
    # Get state and action information from the selected gym environment
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] # Continuous
    env.close()

    ## COPY WEIGHTS (Bomb case) ###########################
    tag2copy = 'sac_lunarT_baseline_plus4_seed0' 

    if tag2copy and load_weights:
        src = os.path.join(trained_model_path,args.env_name,'0_weights',tag2copy)
        dest = os.path.join(save_path,args.env_name,'0_weights',args.run_tag+'_seed%d'%args.seed)
        shutil.copytree(src,dest)
        if load_weights_Q_I:
            shutil.copy(os.path.join(src,'__sac_q_1_weights_sac_hidden_256_rscale_1_s1_u1.weights'),os.path.join(dest,'__sac_q_I_1_weights_sac_hidden_256_rscale_1_s1_u1.weights'))
            shutil.copy(os.path.join(src,'__sac_q_2_weights_sac_hidden_256_rscale_1_s1_u1.weights'),os.path.join(dest,'__sac_q_I_2_weights_sac_hidden_256_rscale_1_s1_u1.weights'))
            shutil.copy(os.path.join(src,'__sac_target_q_1_weights_sac_hidden_256_rscale_1_s1_u1.weights'),os.path.join(dest,'__sac_target_q_I_1_weights_sac_hidden_256_rscale_1_s1_u1.weights'))
            shutil.copy(os.path.join(src,'__sac_target_q_2_weights_sac_hidden_256_rscale_1_s1_u1.weights'),os.path.join(dest,'__sac_target_q_I_2_weights_sac_hidden_256_rscale_1_s1_u1.weights'))
            print(' (*) Original Q1 and Q2 are copied as Q_I_1 and Q_I_2 (since they learned to land)...')
        else:
            print(' (*) Original Q1 and Q2 are copied. Q_I_1 and Q_I_2 start from Zero.')
        print('Copy weights... \n From: %s \n To: %s'%(src,dest))
        time.sleep(10)
        
    # Save run information
    auxpath=os.path.join(save_path,args.env_name,'0_runs',args.run_tag+'_seed%d'%args.seed)
    if not os.path.exists(auxpath):    
        os.makedirs(auxpath)
    with open(os.path.join(save_path,args.env_name,'0_runs',args.run_tag+'_seed%d'%args.seed,'info_run.txt'),'w') as f:
        f.write('Training method: SAC-Vanilla \n')
        f.write('Environment: %s \n'%(args.env_name))
        f.write('Number of states: %d \n'%(state_dim))
        f.write('Number of actions: %d \n'%(action_dim))
        f.write('Max timesteps: %d \n'%(args.max_timesteps))
        f.write('Env seed: %d \n'%(args.seed))
        f.write('hidden_layer_size: %d \n'%(hidden_layer_size))
        f.write('num_hidden_layers: %d \n'%(num_hidden_layers))
        f.write('batch_size: %d \n'%(batch_size))
        f.write('update_ntimes: %d \n'%(update_n_times))
        f.write('num_episodes_before_training: %d \n'%(num_episodes_before_training))
        f.write('num_episodes_random_actions: %d \n'%(num_episodes_random_actions))
        f.write('replay_buffer_size: %d \n'%(replay_buffer_size))
        f.write('policy_lr: %1.2e \n'%(policy_lr))
        f.write('q_lr: %1.2e \n'%(q_lr))
        f.write('entropy_scaling_lr: %1.2e \n'%(entropy_scaling_lr))
        f.write('soft_tau: %1.2e \n'%(soft_tau))
        f.write('reward_scale: %d \n'%(reward_scale))
        f.write('use_automatic_entropy_tunning: %s \n'%(use_automatic_entropy_tunning))
        f.write('load_weights: %s \n'%(load_weights))
        f.write('load_weights_Q_I: %s \n'%(load_weights_Q_I))
         

    print('\n==============================================')
    print('=====TRAINING: SAC-I (automated inhibition)  ====')
    print('==============================================')
    print(' --------------------------------------------')
    print('| Environment:   {:s}'.format(args.env_name))
    print('| run_tag:       {:s}'.format(args.run_tag))
    print('| run info saved in: {:s}'.format(auxpath))
    print('| Number of states:  {:5d}'.format(state_dim))
    print('| Number of actions: {:5d}'.format(action_dim))
    print('| Max timesteps:     {:5d}'.format(args.max_timesteps))
    print('| Env seed:          {:5d}'.format(args.seed))
    print(' ---------------------------------------------')
    time.sleep(3)

    multiagent_sac_trainer = MultiAgentSACTrainer(env_name = args.env_name,
                                                  num_hidden_layers=num_hidden_layers,
                                                  hidden_layer_size=hidden_layer_size,
                                                  batch_size=batch_size,
                                                  update_n_steps=update_n_steps,
                                                  update_n_times=update_n_times,
                                                  q_lr=q_lr,
                                                  policy_lr=policy_lr,
                                                  soft_tau=soft_tau,
                                                  reward_scale=reward_scale,
                                                  entropy_scaling_lr=entropy_scaling_lr,
                                                  use_automatic_entropy_tunning=use_automatic_entropy_tunning,
                                                  replay_buffer_size=replay_buffer_size,
                                                  num_episodes_before_training=num_episodes_before_training,
                                                  num_episodes_random_actions=num_episodes_random_actions,
                                                  max_num_episodes=max_num_episodes,
                                                  load_weights=load_weights,
                                                  load_weights_Q_I = load_weights_Q_I,
                                                  save_path=save_path,
                                                  log_interval_num_episodes=log_interval_num_episodes,
                                                  use_tensorboard=use_tensorboard,
                                                  run_tag=args.run_tag,
                                                  policy_network_type=policy_network_type,
                                                  nback=nback,
                                                  seed=args.seed,
                                                  render = render,
                                                  )


    multiagent_sac_trainer.train()
