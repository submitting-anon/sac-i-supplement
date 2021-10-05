##################################################################################
# 5) Script to train SAC agent
# - Env: BipedalWalkerHardcore-v3
# - Reward: 
#     - stuck penalty: add last 5 steps cumulative reward
#     - remove the original fall penalty
# - Inhibition rule: whenever stuck position (cumulative last 5 rewards < 0)
# - Q-I weights: retrained
# - Options:
#   - Use STOP_TRIAL_PCT: from 0.0 to 1.0 (default is 1.0)
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
from replaybuffer import ReplayBuffer
from sac_vanilla import SAC, PolicyNetwork, create_base_dir, load_model_weights, ScoreBoard_bipedalH3

from utils import uniform_value_from_range

####################################################
# Set global variables
#
STOP_TRIAL_PCT = 1.0  # from 0.0 to 1.0 (default is 1.0, i.e. Hardcore version)
####################################################

class PolicyNetworkType():
    STANDARD = 0
    GRU = 1

class MultiAgentSACTrainer:

    def __init__(self, env_name=None, num_hidden_layers=None, hidden_layer_size=None, batch_size=None, update_n_steps=None, update_n_times=None,
                       q_lr=None, policy_lr=None, soft_tau=None, reward_scale=None, entropy_scaling_lr=None, 
                       use_automatic_entropy_tunning=True, replay_buffer_size=None, num_episodes_before_training=None,
                       num_episodes_random_actions=None, max_num_episodes=None,
                       load_weights=None, save_path=None, 
                       log_interval_num_episodes=None, use_tensorboard=None, run_tag=None, policy_network_type=None, 
                       nback=None, seed=None, render=None):

        ## Prepare environment
        self.seed = seed
        self.env_name = env_name
        
        # ######################################################################
        self.env_stop = gym.make(self.env_name)    # April19
        self.env_go = gym.make('BipedalWalker-v3') #
        self.env_go.seed(seed)                     #
        self.env = self.env_stop                   #

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0] # Continuous (scoreboard)
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
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.random.manual_seed(seed)
        import random
        random.seed(seed)

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
        self.policy_nets_save_weights_path = {}

        if self.use_automatic_entropy_tunning:
            self.entropy_scaling_save_weights_path = {}
    
        # initialize replay buffers
        self.replay_buffers = ReplayBuffer(self.replay_buffer_size)
        if self.policy_network_type == PolicyNetworkType.STANDARD:
            # self.replay_buffers = PriorityReplayBuffer(self.replay_buffer_size)
            self.replay_buffers = ReplayBuffer(self.replay_buffer_size)
        
        elif self.policy_network_type == PolicyNetworkType.GRU:
            self.replay_buffers = ReplayBufferGRUs(self.replay_buffer_size)

        if use_automatic_entropy_tunning:
            # Heuristic value for target entropy lower bound used in entropy scaling                
            self.target_entropy = -np.prod(self.action_dim).item()
        else:
            self.target_entropy = -np.prod(self.action_dim).item()

        # initialize networks
        self.sac = SAC(self.state_dim, 
                            self.action_dim, 
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
        self.policy_nets_save_weights_path = path.join(self.save_weights_path, '__sac_policy_weights_%s.weights' % (self.algo_name))
        if self.use_automatic_entropy_tunning:
            self.entropy_scaling_save_weights_path = path.join(self.save_weights_path, '__sac_entropy_scaling_weights_%s.weights' % (self.algo_name))

        create_base_dir(self.policy_nets_save_weights_path)
        if self.load_weights:
            load_model_weights(self.sac.q_1_net, self.q_1_nets_save_weights_path)
            load_model_weights(self.sac.target_q_1_net, self.target_q_1_nets_save_weights_path)
            load_model_weights(self.sac.q_2_net, self.q_2_nets_save_weights_path)
            load_model_weights(self.sac.target_q_2_net, self.target_q_2_nets_save_weights_path)
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
        self.scoreboard = ScoreBoard_bipedalH3(self.save_path_tb)

    def update(self, batch_size):

        # grab batch of experiences
        if self.policy_network_type == PolicyNetworkType.STANDARD:
            states, actions, rewards, next_states, dones = self.replay_buffers.sample(self.batch_size)
            state_trajs = None
        
        elif self.policy_network_type == PolicyNetworkType.GRU:
            states, actions, rewards, next_states, dones, state_trajs, _ = self.replay_buffers.sample(self.batch_size)
    
        policy_losses, alphas, td_errors, q_targets = self.sac.learn(
                                                            torch.FloatTensor(states).to(self.device),
                                                            torch.FloatTensor(actions).to(self.device),
                                                            torch.FloatTensor(rewards).to(self.device).unsqueeze(1),
                                                            torch.FloatTensor(next_states).to(self.device),
                                                            torch.FloatTensor(dones).to(self.device).unsqueeze(1),
                                                            self.gpu_sample_importance_weights,
                                                            self.gpu_nsteps)

        return policy_losses, alphas, q_targets


    def train(self):
        t0 = time.time()

        nUpdates = 0
        ready_to_train = False
        use_policy_action = False
        nhits = 0

        average_alpha = 0 # bomb
        average_loss = 0  # bomb
        count_stop_trial = 0


        num_steps_per_episode_list = deque(maxlen=self.log_interval_num_episodes)
        list_rewards = deque(maxlen=100)
        list_epilen = deque(maxlen=100) # scoreboard

        # last rewards
        last_rewards = deque(maxlen=5)

        alpha_list = []
        loss_list = []

        # April19
        contGo = 0
        contStop = 0
                
        try:
            current_total_num_timesteps = 0
            for episode in range(self.max_num_episodes):
                if np.random.rand() > STOP_TRIAL_PCT:
                    self.env = self.env_go
                    trialtype = 'Go'
                    contGo += 1
                else:
                    self.env = self.env_stop
                    trialtype = 'Stop'
                    contStop += 1

                rewards_list_episode = []
                
                if self.policy_network_type == PolicyNetworkType.GRU:
                    hidden_gru_states = None
                    states_list = deque(maxlen=self.nback)
                    actions_list = deque(maxlen=self.nback)
                    rewards_list = deque(maxlen=self.nback)
                    next_states_list = deque(maxlen=self.nback)
                    dones_list = deque(maxlen=self.nback)

                if episode >= self.num_episodes_random_actions and not use_policy_action:
                    print("Now using policy for actions...\n")
                    use_policy_action = True

                if episode >= self.num_episodes_before_training and not ready_to_train:
                    print("Now training...\n\n", end='')
                    ready_to_train = True

                print('--- Episode {:5d} ({:2s} trial, Total Go:{:4d}| Stop:{:4d}) -----------------------'.format(episode,trialtype, contGo,contStop))
                

                # Initialize
                state = self.env.reset()
                
                if self.render:
                    self.env.render()
                
                cum_return = 0
                alpha = 0
                actor_loss = 0
                avg_q_target = 0
                cum_a_std = 0
                epi_num_updates = 0
                success = False #

                # BipedalWalker
                previous_lidar = state[14:] 
                count_stop = 0
                use_I = 0 # use extra reward when stuck

                ''' step loop '''
                aux = ['/','-','\\','|']
                
                for step_num in range(self.max_num_steps_per_episode):

                    print('  Step {:4d}  {:s} \r'.format(step_num,aux[step_num%4]),end="",flush=True)

                    if use_policy_action:
                        action, a_std = self.sac.policy_net(torch.FloatTensor(state).to(self.device))
                        action = action.cpu().data.numpy()
                        a_std = a_std.cpu().data.numpy()
                        
                    else:
                        action = np.random.uniform(low=-1.0, high=1.0, size=self.action_dim)
                        a_std = 0

                    # STEP #
                    next_state, reward, done, _ = self.env.step(action)

                    # Reward Shaping ###############################################
                    
                    # original reward
                    reward0 = reward # keep the original for check done status                    
                    
                    #### REWARD SHAPING ##################
                    
                    # Check if stuck every 10 step
                    if (step_num+1) % 10 == 0:
                        
                        # # Threshold based on Lidar change
                        if sum(last_rewards) < 0: # May11 (seeds 1-4)
                            print('--- Step %d: STUCK PENALTY! Not getting reward... sum(last_rewards):'%(step_num),sum(last_rewards))
                            count_stop+=1
                            use_I = 1
                        else:
                            use_I = 0

                    # if stuck, add extra penalty
                    if use_I:
                        if reward == -100: # when stuck, disconsider hit in Q to motivate exploring more
                            reward = sum(last_rewards) # add last 5 steps cum reward (if stuck, usually it is negative...) the idea is to motivate to get out of stuck position
                        else:
                            reward += sum(last_rewards) # add last 5 steps cum reward (if stuck, usually it is negative...) the idea is to motivate to get out of stuck position

                    current_total_num_timesteps += 1
                    cum_return += reward0
                    last_rewards.append(reward0)

                    cum_a_std += a_std
                    if self.render:
                        self.env.render() #
                    ###########################################
                    
                    # push to replay buffer and append to lists for stats
                    done01 = 0.0 if step_num+1 == self.max_num_steps_per_episode else float(done)
                    if self.policy_network_type == PolicyNetworkType.STANDARD:
                        self.replay_buffers.push(state, action, reward/10, next_state, done01) # JI: binary done    
                    
                    elif self.policy_network_type == PolicyNetworkType.GRU:
                        states_list.append(state)
                        actions_list.append(action)
                        rewards_list.append(reward)
                        next_states_list.append(next_state)
                        dones_list.append(False)
        
                        if step_num > self.nback: # push to memory if it has the minumum number of steps
                            self.replay_buffers.push(states_list,
                                                    actions_list,
                                                    rewards_list,
                                                    next_states_list,
                                                    dones_list)
                    
                    # update networks
                    if ready_to_train:
                        for _ in range(self.update_n_times):
                            ## LEARN(UPDATE)
                            loss_policy, alpha_updated,q_target = self.update(self.batch_size)

                            if self.use_automatic_entropy_tunning:
                                alpha = alpha_updated.cpu().detach().numpy()
                                alpha = alpha[0]
                            else:
                                alpha = alpha_updated

                            loss_list.append(loss_policy.cpu().detach().numpy() / self.reward_scale)
                            alpha_list.append(alpha)
                            actor_loss = loss_policy
                            nUpdates += 1
                            epi_num_updates += 1
                    
                    # Episode averages
                    avg_a_std = cum_a_std/(step_num+1)
                    if done:
                        
                        if reward0 == -100:
                            print('  /\/\/\/\/')
                            print('< Deck hit!! >')
                            print('  \/\/\/\/\/')
                            nhits += 1
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
                print("--- Ep {} (ep.len= {}, Timesteps={:7d}, alpha={:.3f}, loss={:.3f}):   Ep.Reward:{:4.2f}   Avg100.Reward:{:4.2f}   Time: {:02}:{:02}:{:02}".format(episode, step_num,current_total_num_timesteps,alpha,end_loss,cum_return, sum(list_rewards)/len(list_rewards),tt//3600,tt%3600//60,tt%60))
                print('Last action:', action)
                print('Avg action_std:', avg_a_std)
                print('Last batch.avg.q_target:', avg_q_target)
                print('Stop trial (percent): %2.2f'%(count_stop/step_num))
    
                if ready_to_train and log_interval == 0:

                    average_num_steps_per_episode = int(np.mean(num_steps_per_episode_list))
                    num_steps_per_episode_list = []

                    total_reward = cum_return
                    average_alpha = np.mean(alpha_list)
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

                    create_base_dir(self.policy_nets_save_weights_path)
                    torch.save(self.sac.policy_net.state_dict(), path.join(self.policy_nets_save_weights_path + suffix))

                    if self.use_automatic_entropy_tunning:
                        create_base_dir(self.entropy_scaling_save_weights_path)
                        torch.save(self.sac.entropy_scaling, path.join(self.entropy_scaling_save_weights_path + suffix))

                    if self.use_tensorboard:
                        ts = current_total_num_timesteps
                        self.writer.add_scalars('0_TotalReward_x_Timesteps', {'SAC_%s' % (self.algo_name): total_reward}, ts)
                        self.writer.add_scalars('1_AvgActorLoss_x_Timesteps', {'SAC_%s' % (self.algo_name): average_loss}, ts)
                        self.writer.add_scalars('2_AvgAlpha_x_Timesteps', {'SAC_%s' % (self.algo_name): average_alpha}, ts)
                        self.writer.add_scalars('3_Ep.Length_x_Timesteps', {'SAC_%s' % (self.algo_name): step_num}, ts)
                        self.writer.add_scalars('4_Episodes_x_Timesteps', {'SAC_%s' % (self.algo_name): episode}, ts)
                        self.writer.add_scalars('5_Updates_x_Timesteps', {'SAC_%s' % (self.algo_name): nUpdates}, ts)

                    # Update scoreboard
                    if episode % 1 ==0:
                        self.scoreboard.update_data_frame(current_total_num_timesteps,episode,average_alpha,-99,average_loss,nUpdates,cum_return,list_rewards,list_epilen,nhits,0, trialtype)
                        self.scoreboard.save_df()
                        

                alpha_list = []
                loss_list = []
            
            self.scoreboard.update_data_frame(current_total_num_timesteps,episode,average_alpha,-99,average_loss,nUpdates,cum_return,list_rewards,list_epilen,nhits,0, trialtype)
            self.scoreboard.save_df()

        except (KeyboardInterrupt, SystemExit):
            print("Exiting...")
            self.scoreboard.update_data_frame(current_total_num_timesteps,episode,average_alpha,-99,average_loss,nUpdates,cum_return,list_rewards,list_epilen,nhits,0, trialtype)
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

    config_policy_pretrained = {
        'num_neurons_per_layer':256,
        'num_hidden_layers':2,
        'weights_path':'models_trained/BipedalWalker-v3/0_weights/sac_bipedal_baseline_seed1/',
    }

    num_hidden_layers = 2
    hidden_layer_size = 256
    
    policy_lr = 5e-4
    q_lr = 5e-4
    entropy_scaling_lr = 5e-4
    
    soft_tau = 1e-3 # 5e-3

    use_automatic_entropy_tunning = True
    
    reward_scale = 1

    batch_size = 64 #

    update_n_steps = 1
    update_n_times = 1

    replay_buffer_size = 1000000

    num_episodes_before_training = 50
    num_episodes_random_actions = 5

    max_num_episodes = int(4e3)

    # load_weights = True
    render = True
    save_path = 'models'
    
    use_tensorboard = True

    log_interval_num_episodes = 1

    ######################
    # SAC Policy Specific
    ######################
    # policy_network_type = PolicyNetworkType.GRU
    policy_network_type = 0
    nback = None
    if policy_network_type == 1: # GRU
        nback = 20

    #########################################################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', dest='run_tag', type=str, default='no_tag', help="Tag name for run")
    parser.add_argument('-i', dest='max_timesteps', type=int, default=1000000, help="Maximum number of time steps")
    parser.add_argument('--env', dest='env_name', type=str, default='BipedalWalkerHardcore-v3', help="Environment name")
    parser.add_argument('--seed', dest='seed', type=int, default=7, help="Seed for the env initialization")
    parser.add_argument('--quick', dest='test_mode', action='store_true', help="Quick test mode")
    parser.add_argument('--render', dest='render', action='store_true', help="Render on")
    parser.add_argument('--from_scratch', dest='from_scratch', action='store_false', help="Train from scratch")
    parser.add_argument('--stop_pct', dest='stop_pct', type=float, default=1.0, help="Stop trial pct. Try 0.70-1.0")
    args = parser.parse_args()
    # UPDATE if given
    STOP_TRIAL_PCT = args.stop_pct 

    # Update configuration
    render = args.render
    load_weights = args.from_scratch  
  
    # Get state and action information from the selected gym environment
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] # Continuous (scoreboard)
    env.close()

    ## COPY WEIGHTS ###########################

    # Option 2: give the whole path
    if config_policy_pretrained['weights_path'] and load_weights:
        src = os.path.join(config_policy_pretrained['weights_path'])
        dest = os.path.join(save_path,args.env_name,'0_weights',args.run_tag+'_seed%d'%args.seed)
        shutil.copytree(src,dest)
        print('Copy weights... \n From: %s \n To: %s'%(src,dest))
        time.sleep(10)

    if args.env_name=='BipedalWalkerHardcore-v3':
        
        # Load B_seed7 from BipedalWalker-v3
        num_hidden_layers = 2
        hidden_layer_size = 256
        batch_size = 64 # 128 # seems to work better for SAC agents
        render = True
        use_automatic_entropy_tunning = True
        num_episodes_before_training = 5
        num_episodes_random_actions = 0



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

    print('\n==============================================')
    print('==========   TRAINING: Vanilla SAC   =========')
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
