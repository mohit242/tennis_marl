from comet_ml import Experiment
import torch
import torch.nn.functional as F
import numpy as np
# from .network import *
from .ac import Actor as DDPGActor
from .ac import Critic as DDPGCritic
from .utils import *
import sys
import logging
# import wandb
experiment = Experiment(api_key="FhiGGed6g73CWKq7YS2AEDSaL",
                        project_name="tennis_marl", workspace="drl")
class MADDPGAgent:

    def __init__(self, env, start_steps=1000, train_after_every=20, steps_per_epoch=10, gradient_clip=2, gamma=0.95,
                 device='cpu', minibatch_size=256, buffer_size=10e5, polyak=0.01):
        super().__init__()
        self.device = device
        self.minibatch_size = minibatch_size
        self.env = env
        self.gradient_clip = gradient_clip
        self.steps_per_epoch = steps_per_epoch
        self.train_after_every = train_after_every
        self.gamma = gamma
        self.polyak = polyak
        self.start_steps = start_steps
        self.brain_name = env.brain_names[0]
        self.brain = env.brains[self.brain_name]
        env_info = env.reset(train_mode=True)[self.brain_name]
        self.num_agents = len(env_info.agents)
        self.action_dim = self.brain.vector_action_space_size
        self.state_dim = env_info.vector_observations.shape[1]
        # For this particular scenario we are using the same actor for both agents
        self.actor = DDPGActor(self.state_dim, self.action_dim).to(device)
        self.critic = DDPGCritic(self.state_dim, self.action_dim).to(device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4, weight_decay=0.0001)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.replay_buffer = ReplayBuffer(self.action_dim, int(buffer_size), minibatch_size)

        #wandb.watch((self.critic, self.actor))
        hyperparameters = {"start_steps": start_steps, "train_after_every": train_after_every,
                           "steps_per_epoch": steps_per_epoch, "gradient_clip": gradient_clip, "gamma": gamma,
                           "minibatch_size": minibatch_size, "polyak": polyak}
        experiment.log_multiple_params(hyperparameters)
        self.actor_target = DDPGActor(self.state_dim, self.action_dim).to(device)
        self.critic_target = DDPGCritic(self.state_dim, self.action_dim).to(device)
        self.step_counter = 0
        # for target_param, local_param in zip(self.actor_target.parameters(), self.actor.parameters()):
        #     target_param.data.copy_(local_param.data)
        #
        # for target_param, local_param in zip(self.critic_target.parameters(), self.critic.parameters()):
        #     target_param.data.copy_(local_param.data)

    def learn_step(self):
        if len(self.replay_buffer) < self.start_steps:
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            state = env_info.vector_observations
            for _ in range(self.start_steps):
                action = np.random.uniform(-1, 1, (self.num_agents, self.action_dim))
                # action = np.clip(action, -1, 1)
                env_info = self.env.step(action)[self.brain_name]
                next_state = env_info.vector_observations
                reward = np.array(env_info.rewards)
                done = np.array(env_info.local_done)
                # for s, a, r, ns, d in zip(state, action, reward, next_state, done):
                #     self.replay_buffer.add(s, a, r, ns, d)
                self.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                if np.any(done):
                    env_info = self.env.reset(train_mode=True)[self.brain_name]
                    state = env_info.vector_observations

        env_info = self.env.reset(train_mode=True)[self.brain_name]
        state = env_info.vector_observations
        score = np.zeros(self.num_agents)
        while True:
            action = self.actor_target(torch.Tensor(state).to(self.device))
            action = add_noise(action.cpu()).to(self.device)
            env_info = self.env.step(action.detach().cpu().numpy())[self.brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            score += reward
            done = env_info.local_done
            # for s, a, r, ns, d in zip(state, action.detach().numpy(), reward, next_state, done):
            #     self.replay_buffer.add(s, a, r, ns, d)
            self.replay_buffer.add(state, action.detach().cpu().numpy(), reward, next_state, done)
            state = next_state
            self.step_counter += 1
            if self.step_counter % self.train_after_every == 0:
                for epc in range(self.steps_per_epoch):
                    states, actions, rewards, next_states, dones = self.replay_buffer.sample()
                    states = states.to(self.device)
                    actions = actions.to(self.device)
                    rewards = rewards.to(self.device)
#                     rewards = (rewards - torch.mean(rewards))/(torch.std(rewards) + 1.0e-10)
                    next_states = next_states.to(self.device)
                    dones = dones.to(self.device)
                    q_fut = self.critic_target(next_states, self.actor_target(next_states)).squeeze(-1)
                    targets = rewards + self.gamma * (1 - dones) * q_fut
                    critic_loss = F.mse_loss(self.critic(states.float(), actions.float()).squeeze(-1), targets)

                    self.critic_opt.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
                    self.critic_opt.step()
                    if epc % 2 == 0:
#                     if True:
                        actor_loss = -torch.mean(self.critic(states.detach(), self.actor(states)))

                        self.actor_opt.zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
                        self.actor_opt.step()
                        experiment.log_metric("critic_loss", critic_loss)
                        experiment.log_metric("actor_loss", actor_loss)
                        #wandb.log({"critic_loss": critic_loss, "actor_loss": actor_loss})
                    self.soft_update()

            if np.any(done):
                break
        experiment.log_metric("score", np.mean(score))
        return np.mean(score)

    def soft_update(self):
        for target_param, local_param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.polyak*local_param.data + (1.0 - self.polyak)*target_param.data)

        for target_param, local_param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.polyak*local_param.data + (1.0 - self.polyak)*target_param.data)

    def eval_step(self):
        pass

