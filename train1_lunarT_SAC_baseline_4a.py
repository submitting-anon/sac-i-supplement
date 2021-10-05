##################################################################################
# 1) Script to train baseline agent
# - Env: LunarLanderContinuous-v2
# - Reward: It includes the time penalty
# - State: dummy states for the bomb coordinates
##################################################################################

# import cProfile
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
# from collections import defaultdict
from replaybuffer import ReplayBuffer
from sac_vanilla import SAC, PolicyNetwork, create_base_dir, load_model_weights, ScoreBoard


n_dummy_states = 4 
    
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
        self.env = gym.make(self.env_name)
        self.state_dim = self.env.observation_space.shape[0] + n_dummy_states
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
        self.scoreboard = ScoreBoard(self.save_path_tb)

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

        num_steps_per_episode_list = deque(maxlen=self.log_interval_num_episodes)
        list_rewards = deque(maxlen=100)
        list_epilen = deque(maxlen=100) # scoreboard

        alpha_list = []
        loss_list = []
                
        try:
            current_total_num_timesteps = 0
            for episode in range(self.max_num_episodes):

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

                print('Episode:%6d \r' % (episode), end='')

                # Initialize
                state = self.env.reset()
                # ADD DUMMY STATES
                state = np.concatenate((state,-2*np.ones(n_dummy_states)))


                if self.render:
                    self.env.render()
                cum_return = 0
                alpha = 0
                actor_loss = 0
                avg_q_target = 0
                cum_a_std = 0
                epi_num_updates = 0
                success = False # LunarLander

                ''' step loop '''
                aux = ['/','-','\\','|']
                for step_num in range(self.max_num_steps_per_episode):
                    print('--- Episode {:5d}  Step {:4d}  {:s} \r'.format(episode,step_num,aux[step_num%4]),end="",flush=True)

                    if use_policy_action:
                        action, a_std = self.sac.policy_net(torch.FloatTensor(state).to(self.device))
                        action = action.cpu().data.numpy()
                        a_std = a_std.cpu().data.numpy()
                        
                    else:
                        action = np.random.uniform(low=-1.0, high=1.0, size=self.action_dim)
                        a_std = 0
                    
                    # STEP #
                    next_state, reward, done, _ = self.env.step(action)
                    # ADD DUMMY STATES
                    next_state = np.concatenate((next_state,-2*np.ones(n_dummy_states)))

                    # Reward Shaping ###############################################
                    reward0 = reward # ORIGINAL

                    ## ADD penalty per step (TIMED)
                    reward += -0.1 # in 1000 steps will get -100
                    ################################################################

                    current_total_num_timesteps += 1
                    cum_return += reward
                    cum_a_std += a_std
                    if self.render:
                        self.env.render()
                    ###########################################
                    
                    # push to replay buffer and append to lists for stats
                    done01 = 0.0 if step_num+1 == self.max_num_steps_per_episode else float(done)
                    if self.policy_network_type == PolicyNetworkType.STANDARD:
                        # RESCALE REWARD (normalize)
                        self.replay_buffers.push(state, action, reward/10, next_state, done01) # binary done    
                    
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

                            # print('q_target(batch avg):',q_target.mean())
                            # _ = input('PRESS...')

                            if self.use_automatic_entropy_tunning:
                                alpha = alpha_updated.cpu().detach().numpy()
                                alpha = alpha[0]
                            else:
                                alpha = alpha_updated

                            loss_list.append(loss_policy.cpu().detach().numpy() / self.reward_scale)
                            alpha_list.append(alpha)
                            # print(step_num)
                            actor_loss = loss_policy
                            nUpdates += 1
                            epi_num_updates += 1
                    
                    # Episode averages
                    avg_a_std = cum_a_std/(step_num+1)
                    if done:
                        
                        # LunarLander 
                        if reward0==100:
                            success = True
                        if reward0==-100:
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
                # print('\n ALPHA(%s):'%type(alpha), alpha)
                print("--- Ep {} (ep.len= {}, Timesteps={:7d}, alpha={:.3f}, loss={:.3f}):   Ep.Reward:{:4.2f}   Avg100.Reward:{:4.2f}   Time: {:02}:{:02}:{:02}".format(episode, step_num,current_total_num_timesteps,alpha,end_loss,cum_return, sum(list_rewards)/len(list_rewards),tt//3600,tt%3600//60,tt%60))
                # print("--- Episode {:5d} (len={:3d}, timestep={:7d}) \t TOTAL:{:4.2f} ".format(episode, step_num,current_total_num_timesteps,cum_return))
                print('Last action:', action)
                print('Avg action_std:', avg_a_std)
                print('Last batch.avg.q_target:', avg_q_target)
                print('Success:', success,'\t Num Hits:',nhits) # scoreboard


                # Stop if average greater than 200 in the last 100 episodes (LunarLander)
                if (min(list_rewards)>=200) and (self.env_name=='LunarLanderContinuous-v2'):
                    self.scoreboard.update_data_frame(current_total_num_timesteps,episode,average_alpha,average_loss,nUpdates,list_rewards,list_epilen,nhits)
                    print('--------------------------------------------------------')
                    print('--- Goal Achieved: Avg reward >200 over 100 episodes ---')
                    print('--------------------------------------------------------')
                    break
                    # self.env.close()
                
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
                        self.writer.add_scalars('0_Num.Hits_x_Timesteps', {'SAC_%s' % (self.algo_name): nhits}, ts)
                        self.writer.add_scalars('1_AvgActorLoss_x_Timesteps', {'SAC_%s' % (self.algo_name): average_loss}, ts)
                        self.writer.add_scalars('2_AvgAlpha_x_Timesteps', {'SAC_%s' % (self.algo_name): average_alpha}, ts)
                        self.writer.add_scalars('3_Ep.Length_x_Timesteps', {'SAC_%s' % (self.algo_name): step_num}, ts)
                        self.writer.add_scalars('4_Episodes_x_Timesteps', {'SAC_%s' % (self.algo_name): episode}, ts)
                        self.writer.add_scalars('5_Updates_x_Timesteps', {'SAC_%s' % (self.algo_name): nUpdates}, ts)

                    # Update scoreboard
                    if episode % 10 ==0:
                        self.scoreboard.update_data_frame(current_total_num_timesteps,episode,average_alpha,average_loss,nUpdates,list_rewards,list_epilen,nhits)
                    

                alpha_list = []
                loss_list = []
            
            self.scoreboard.update_data_frame(current_total_num_timesteps,episode,average_alpha,average_loss,nUpdates,list_rewards,list_epilen,nhits)
            self.scoreboard.save_df()

        except (KeyboardInterrupt, SystemExit):
            print("Exiting...")
            self.scoreboard.update_data_frame(current_total_num_timesteps,episode,average_alpha,average_loss,nUpdates,list_rewards,list_epilen,nhits)
            self.scoreboard.save_df()
            self.env.close()
            return

if __name__ == '__main__':

    import argparse
    import os
    
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
    
    soft_tau = 5e-3

    use_automatic_entropy_tunning = True
    
    reward_scale = 1

    batch_size = 256 # 128

    update_n_steps = 1
    update_n_times = 1

    replay_buffer_size = 1000000

    num_episodes_before_training = 5 # 50
    num_episodes_random_actions = 5

    max_num_episodes = int(2e3)

    load_weights = False
    render = True
    save_path = 'models'

    
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
    args = parser.parse_args()

    # Get state and action information from the selected gym environment
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] # Continuous (scoreboard)
    env.close()

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
