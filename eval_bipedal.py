import gym
import time
import torch
import os
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np

from sac_vanilla import PolicyNetwork, load_model_weights
from gym import wrappers
from datetime import datetime

from utils import uniform_value_from_range # random bomb coordinates
import random
        

def run_eval(config_dict=None, seed=7):
    
    try:
        # Auxiliar variables
        t0 = time.time()
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        df = pd.DataFrame(columns=['episode','length','reward','nlandings','nhits'])
        aux = ['/','-','\\','|']

        ''' Environment and network information '''
        if config_dict:
            method = config_dict['Training method']
            env_name = config_dict['Environment'] 
            run_tag = config_dict['run_tag']
            num_states = int(config_dict['Number of states'])
            num_actions = int(config_dict['Number of actions'])
            num_neurons_per_layer =  int(config_dict['hidden_layer_size'])
            num_hidden_layers = int(config_dict['num_hidden_layers'])
            reward_scale = int(config_dict['reward_scale'])
            save_movies = config_dict['save_movies']
            save_df = config_dict['save_df']
            include_date = config_dict['include_date']
            model_dir = config_dict['model_dir']
            if method.lower() == 'sac-vanilla':
                print('Is vanila!')
                update_ntimes = int(config_dict['update_ntimes'])
            if 'randBomb' in  run_tag:
                num_states+=4 # add bomb coordinates
            if 'plus4' in  run_tag:
                num_states+=4 # add bomb coordinates
            # Switch to Go trial if requested
            if env_name=='BipedalWalkerHardcore-v3' and config_dict['gotrial']:
                use_bipedal = True
            else:
                use_bipedal = False

        else:
            method = 'SAC-Vanilla' 
            env_name = 'LunarLanderContinuous-v2' 
            num_states = 8 
            num_actions = 2 
            run_tag = 'exp6_0'
            num_neurons_per_layer =  256 
            num_hidden_layers = 1 
            reward_scale = 2
            update_every_n_memories = 5
            save_movies = True
            save_df = False
            include_date = False
            model_dir = 'models_vanilla'

        print('--------------------------------------------------------')
        print('---------------------- EVALUATION ----------------------')
        print('- Environment:', env_name)
        print('- run_tag:', run_tag)
        print('- training method: %s'%method)
        print('--------------------------------------------------------')

        # Option to save with date and time
        now = datetime.now()
        if include_date:
            dt_string = now.strftime("_%m-%d-%Y_%Hh%Mm")
        else:
            dt_string = ''

        ''' load main policy '''
        policy = PolicyNetwork(num_states,
                                    num_actions,
                                    num_neurons_per_layer,
                                    0,
                                    device,
                                    num_hidden_layers)

        if method.lower() == 'sac-vanilla':
            fname = '__sac_policy_weights_sac_hidden_%d_rscale_%d_s1_u%d.weights'%(num_neurons_per_layer,reward_scale,update_ntimes)
            weights_path = os.path.join(model_dir,env_name,'0_weights',run_tag,fname)
        else:
            fname = '__sac_policy_weights_sac_hidden_%d_rscale_%d_s1.weights'%(num_neurons_per_layer,reward_scale)
            weights_path = os.path.join(model_dir,env_name,'sac','0_weights',run_tag,fname)
            

        load_model_weights(policy, weights_path)
        policy.to(device)
        policy.eval()

        """ Main loop with the evaluation (default 100 episodes)"""

        # Random bomb special case
        if ('randBomb' in run_tag) or ('lunarT' in run_tag) :
            env = gym.make('LunarLanderContinuous-v2')
            add_bomb_coord = True
        elif use_bipedal:
            env = gym.make('BipedalWalker-v3')
            add_bomb_coord = False
        else:
            env = gym.make(env_name)
            add_bomb_coord = False
        
        # set seed
        env.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.random.manual_seed(seed)
        random.seed(seed)

        # Option to save movies
        if save_movies:
            env = wrappers.Monitor(env, './eval_out/movies/%s/%s%s'%(env_name,run_tag,dt_string), force=True)       # Output MP4
        
        # counters
        nhits = 0
        nlandings = 0

        # Run
        for e in range(100):
            # Simulate
            cum_r = 0
            state = env.reset()

            print('--- Episode {:5d} ----------------------- (seed {:2d})'.format(e,seed))
            if add_bomb_coord:
                # Fixed version:
                # x_bomb = [-0.1,0.1]
                # y_bomb = [0.3,0.4]
                x0 = uniform_value_from_range(-0.2, 0.2)
                y0 = uniform_value_from_range(0.1, 0.5)
                x_bomb = [x0-0.1,x0+0.1] # same size as the fixed box
                y_bomb = [y0-0.05,y0+0.05]
                # ADD BOMB COORDINATES IN THE STATES
                state = np.concatenate((state,x_bomb))
                state = np.concatenate((state,y_bomb))
                print('  Bomb coordinates x: [%2.2f,%2.2f]'%(x_bomb[0],x_bomb[1]))
                print('  Bomb coordinates y: [%2.2f,%2.2f]'%(y_bomb[0],y_bomb[1]))
            elif 'plus4' in run_tag:
                    # add dummy states
                    state = np.concatenate((state,-2*np.ones(4)))

            for s in range(3000):
                env.render()
                statex = state
                # s_next, reward, done, _ = env.step(env.action_space.sample()) # take a random action
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = policy(state, deterministic=True)
                action = action[0].detach().cpu().numpy()
                # STEP
                next_state, reward, done, _ = env.step(action)
                cum_r += reward
                
                # print('Episode {:2d}  Step {:4d}  {:s} \r'.format(e,s,aux[s%4]),end="",flush=True)
                print('  Step {:4d}  {:s} \r'.format(s,aux[s%4]),end="",flush=True)

                ## BOMB update ###############################################
                if add_bomb_coord:
                    # ADD BOMB COORDINATES IN THE STATES 
                    next_state = np.concatenate((next_state,x_bomb))
                    next_state = np.concatenate((next_state,y_bomb))
                    # Check if it hit the bomb
                    # state[0]: x   
                    # state[1]: y   
                    x_next = next_state[0]
                    y_next = next_state[1]
                    if (x_next>x_bomb[0]) and (x_next<x_bomb[1]) and (y_next>y_bomb[0]) and (y_next<y_bomb[1]):
                        print('  /\/\/\/\/\/\/\/\/\/')
                        print('< Boooom!! Exploded!! >')
                        print('  \/\/\/\/\/\/\/\/\/\/')
                        # _ = input('PAUSE...') 
                        reward_bomb = -150 # bomb-bomb3 
                        # nbombs += 1
                        done = True
                    else:
                        reward_bomb = 0
                    # ########## LunarLanderContinous-v2 ###########
                    # print()
                    # print('(x,y,angle):   (%2.2f,%2.2f,%2.2f)'%(statex[0],statex[1],statex[4]))
                    # print('(vx,vy,vangle):(%2.2f,%2.2f,%2.2f)'%(statex[2],statex[3],statex[5]))
                    # print('(Left.Leg,Right.Leg):(%d,%d)'%(statex[7],statex[6]))
                    # print('action:', action)
                    # print('reward:', reward)
                    # print('done:',done)
                    # _ = input('PRESS TO NEXT STEP...')
                    # ##############################################
                    time.sleep(0.01)
                
                elif 'plus4' in run_tag:
                    # add dummy states
                    next_state = np.concatenate((next_state,-2*np.ones(4)))


                # End of episode
                if done:
                    if reward == -100:
                        print('  /\/\/\/\/')
                        print('< Deck hit!! >')
                        print('  \/\/\/\/\/')
                        nhits += 1
                    if reward == 100:
                        print('*********************************')
                        print('** Safely landed: Well done!! ***')
                        print('*********************************')
                        nlandings += 1
                        
                    print('- End of episode %d (done=%s,len=%d): total reward %d      '%(e,done,s,cum_r))
                    print('- Total number of deck hits:',nhits)
                    # _ = input('(PRESS TO START NEW EPISODE...)')
                    # Store in the dataframe
                    d = {'episode':e,'length':s,'reward':cum_r, 'nhits':nhits}
                    df = df.append(d,ignore_index=True)
                    break
                # Prepare next step
                state = next_state
                if save_movies:
                    time.sleep(0.01)
            
        env.close()

        # Save run information
        savename = './eval_out/res_%s_%s%s'%(env_name,run_tag,dt_string)
        if save_df:
            df.to_csv(savename, index=False)
        print('--------------------------------------------------------')
        print('- Average Reward (n=%d): %2.2f(+-%2.2f), Min=%2.2f, Max=%2.2f'%(e,df.reward.mean(),df.reward.std(),df.reward.min(),df.reward.max()))
        print('- Stats: n_success: %d, n_hits: %d'%(nlandings,nhits))
        print('- Results saved in: %s'%(savename))
        print('- Computation time (minutes): %4.1f'%((time.time()-t0)/60))
        print('-------------------------- END -------------------------')
    
    except (KeyboardInterrupt, SystemExit):
        print("- Exiting Evaluation earlier...")
        # Save run information
        savename = './eval_out/res_%s_%s%s'%(env_name,run_tag,dt_string)
        if save_df:
            df.to_csv(savename, index=False)
        print('--------------------------------------------------------')
        print('- Average Reward (n=%d): %2.2f(+-%2.2f), Min=%2.2f, Max=%2.2f'%(e,df.reward.mean(),df.reward.std(),df.reward.min(),df.reward.max()))
        print('- Stats: n_success: %d, n_hits: %d'%(nlandings,nhits))
        print('- Results saved in: %s'%(savename))
        print('- Computation time (minutes): %4.1f'%((time.time()-t0)/60))
        print('-------------------------- END -------------------------')
        env.close()
        return

if __name__ == '__main__':

    """ Input argments """
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', dest='run_tag', type=str, default='no_tag', help="Tag name for run")
    parser.add_argument('--dir', dest='model_dir', type=str, default='models_trained', help="Model directory")
    parser.add_argument('--env', dest='env_name', type=str, default='BipedalWalkerHardcore-v3', help="Environment name")
    parser.add_argument('--movie', dest='save_movies', action='store_true', help="Save movies as well")
    parser.add_argument('--date', dest='include_date', action='store_true', help="Save files with date tag")
    parser.add_argument('--save', dest='save_df', action='store_true', help="Save results to csv files")
    parser.add_argument('--seed', dest='seed', type=int, default=7, help="Seed for the env initialization and the shuffling")
    parser.add_argument('--go', dest='gotrial', action='store_true', help="Go trial")
    args = parser.parse_args()

    """ Load experiment information (if not available, just set the dict manually) """
    try:
        f = open(os.path.join(args.model_dir,args.env_name,'sac','0_runs',args.run_tag,'info_run.txt'))
    except:
        f = open(os.path.join(args.model_dir,args.env_name,'0_runs',args.run_tag,'info_run.txt'))

    lines = list(f)
    f.close()
    config_dict = {}
    for l in lines:
        key = l.split(': ')[0]
        value_str = l.split(': ')[1].split(" ")[0]
        config_dict[key] = value_str
    # Set others
    if 'Training method' not in config_dict.keys():
        config_dict['Training method'] = 'N/A'    

    config_dict['run_tag'] = args.run_tag
    config_dict['save_movies'] = args.save_movies
    config_dict['save_df'] = args.save_df
    config_dict['include_date'] = args.include_date
    config_dict['model_dir'] = args.model_dir
    config_dict['gotrial'] = args.gotrial
    print('- CONFIGURATION: ',config_dict)
    
    # Run evaluation
    run_eval(config_dict, args.seed)

   

