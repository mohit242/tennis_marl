from tennis_maddpg.maddpg_agent import MADDPGAgent
from tennis_maddpg.maddpg_agent import experiment
from unityagents import UnityEnvironment
from collections import deque
import numpy as np
import argparse
import torch as torch
import json


if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-c", "--config", help="File path of config json file", type=str, default="config.json")
    argparser.add_argument("--play", help="sets mode to play instead of train", action="store_true")
    print("Loading params from config.json .....")
    args = argparser.parse_args()
    with open(args.config, 'r') as f:
        params = json.load(f)
    env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86", no_graphics=params['no_graphics'])
    # agent = MADDPGAgent(env, train_after_every=20, gamma=0.995, minibatch_size=128, steps_per_epoch=10,
    #                     start_steps=1000, polyak=0.001, gradient_clip=1.0, device='cuda')
    agent = MADDPGAgent(env, train_after_every=params['train_after_every'], gamma=params['gamma'],
                        minibatch_size=params['minibatch_size'], steps_per_epoch=params['steps_per_epoch'],
                        start_steps=params['start_steps'], polyak=params['polyak'], gradient_clip=params['gradient_clip'],
                        device=params['device'])
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    scores = []
    scores_window = deque(maxlen=100)
    i = 1
    while True:
        score = agent.learn_step()
        scores.append(score)
        scores_window.append(score)
        print("\rEpisode- {:8d} \t Score- {:+8f} \t Mean Score- {:+8f}".format(i, score, np.mean(scores_window)), end="")
        if i%100 == 0:
            print("\rEpisode- {:8d} \t Score- {:+8f} \t Mean Score- {:+8f}".format(i, score, np.mean(scores_window)))
        if np.mean(scores_window) >= 0.5:
            break
        i += 1
    torch.save(agent.actor.state_dict(), 'maddpg_actor.pth')
    torch.save(agent.critic.state_dict(), 'maddpg_critic.pth')
    experiment.log_asset('maddpg_actor.pth')
    experiment.log_asset('maddpg_critic.pth')
