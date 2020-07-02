from argparse import Namespace
import random
from collections import deque
import torch
from models.actor_model import Actor
from models.critic_model import Critic


class Exerminator:

    def __init__(self, env_side_len, actor_hparams, critic_hparams, agent_hparams):
        if isinstance(agent_hparams, Namespace):
            self.hparams = agent_hparams
        else:
            self.hparams = Namespace(**agent_hparams)
        self.device = self.hparams.device  # easier to access from outside the class
        actor_hparams['device'] = self.device
        critic_hparams['device'] = self.device
        self.env_side_len = env_side_len
        self.actor = Actor(actor_hparams)
        self.actor.to(self.hparams.device)
        self.critic = Critic(critic_hparams)
        self.critic.to(self.hparams.device)
        self.memory = deque(maxlen=self.hparams.mem_size)
        self.current_state = None,
        self.current_action = torch.zeros(
            (1, 1, self.env_side_len, self.env_side_len),
            dtype=torch.float32,
            device=self.hparams.device
        )

    def get_sample(self):
        sample = random.sample(self.memory, self.hparams.batch_sz)
        states, actions, rewards, done, next_states = map(list, zip(*sample))
        states = torch.stack(states, dim=0).to(self.hparams.device)
        actions = torch.stack(actions, dim=0).to(self.hparams.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.hparams.device)
        next_states = torch.stack(next_states, dim=0).to(self.hparams.device)
        return states, actions, rewards, done, next_states

    def observe(self, next_state, reward, done):
        next_state = torch.tensor(next_state, device=self.hparams.device).reshape(
            (-1, 1, self.env_side_len, self.env_side_len))
        self.memory.append((self.current_state, self.current_action, reward, done, next_state))
        self.current_state = next_state
        if len(self.memory) > self.hparams.mem_size:
            self.memory.popleft()
        self.train()

    def pick_action(self):
        with torch.no_grad():
            action_prob = self.actor(self.current_state, self.current_action)
            assert action_prob.size() == (1, self.env_side_len * self.env_side_len)
            self.current_action = torch.round(action_prob).reshape((1, 1, self.env_side_len, self.env_side_len))
        return self.current_action.cpu().numpy()

    def train(self):
        if len(self.memory) > self.hparams.batch_sz:
            states, actions, rewards, done, next_states = self.get_sample()
            next_actions = self.actor(next_states)
            next_q_values = self.critic(next_states, next_actions)
            td_targets = rewards + self.hparams.gamma * next_q_values
            if td_targets.dim() < 2:
                td_targets.reshape_((-1, 1))

            # Train critic
            value_loss = self.critic.training_step((states, td_targets))

            # Train actor
            with torch.no_grad:
                current_qs = self.critic(states)
                td_errors = td_targets - current_qs
            policy_loss = self.actor.training_step((states, actions, td_errors))


"""
Notes:
agent_hparams:
    mem_size: Size of replay buffer (default = 10000 ?)
    gamma: discount rate (default = 0.99 ?)
    steps: num. steps in an episode (default = 300 ?)
    device: 'cpu' or 'gpu'
    batch_sz: size of batches from replay buffer
    
"""


