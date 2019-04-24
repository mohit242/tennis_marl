from tennis_maddpg.maddpg_agent import MADDPGAgent
from unityagents import UnityEnvironment
from collections import deque
import numpy as np
import pickle
import torch as torch


if __name__=="__main__":
    env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86")
    agent = MADDPGAgent(env, train_after_every=100, gamma=0.95, minibatch_size=128, steps_per_epoch=20,
                        start_steps=10000)
    torch.manual_seed(0)
    np.random.seed(0)
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
        if i == 2000:
            break
        i += 1
