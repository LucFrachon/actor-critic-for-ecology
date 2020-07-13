from argparse import Namespace
import random
from collections import deque
import torch
import torch.distributions.binomial as binomial
from models.actor_model import Actor
from models.critic_model import Critic

torch.autograd.set_detect_anomaly(True
                                  )
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
        self.current_state = None
        self.current_action = None

    def get_sample(self):
        sample = random.sample(self.memory, min(len(self.memory), self.hparams.batch_sz))
        states, actions, rewards, done, next_states = map(list, zip(*sample))
        states = torch.cat(states, dim=0).to(self.hparams.device)
        actions = torch.cat(actions, dim=0).to(self.hparams.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.hparams.device).reshape(-1, 1)
        next_states = torch.cat(next_states, dim=0).to(self.hparams.device)
        return states, actions, rewards, done, next_states

    def observe(self, next_state, reward, done):
        next_state = torch.tensor(next_state, device=self.hparams.device).reshape(
            (-1, 1, self.env_side_len, self.env_side_len))
        if self.hparams.normalise_states:
            next_state = self.normalise_state(next_state)
        self.memory.append((self.current_state, self.current_action, reward, done, next_state))
        self.current_state = next_state
        if len(self.memory) > self.hparams.mem_size:
            self.memory.popleft()

    def pick_action(self):
        # with torch.no_grad():
        action_prob = self.actor(self.current_state).detach()  # , self.current_action)
        epsilon = self.actor.hparams.epsilon
        # Enforce some exploration
        action_prob = torch.clamp(action_prob, epsilon, 1 - epsilon)
        assert action_prob.size() == (1, self.env_side_len * self.env_side_len)
        binom_dist = binomial.Binomial(1, probs = action_prob)
        self.current_action = binom_dist.sample().reshape((1, 1, self.env_side_len, self.env_side_len))
        # self.current_action = torch.round(action_prob).reshape((1, 1, self.env_side_len, self.env_side_len))
        return self.current_action.cpu().numpy()

    def train(self):
        # if len(self.memory) > self.hparams.batch_sz:
        if len(self.memory) >= 1:
            states, actions, rewards, done, next_states = self.get_sample()
            next_values = self.critic(next_states).detach()
            td_targets = rewards + self.hparams.gamma * next_values
            if td_targets.dim() < 2:
                td_targets.reshape_((-1, 1))

            # Train critic
            value_loss = self.critic.training_step((states, td_targets))
            print(f"\rUpdated Critic, value loss = {value_loss.detach().item():.2f}")

            # Train actor
            # with torch.no_grad():
            current_vs = self.critic(states)
            td_errors = td_targets - current_vs
            actions_ = actions.flatten(start_dim=1)
            policy_loss = self.actor.training_step((states, td_errors, actions_))
            print(f"\rUpdated Actor, policy loss = {policy_loss.detach().item():.2f}")
            return value_loss.detach().item(), policy_loss.detach().item()
        return None, None

    def normalise_state(self, state):
        if (state > 0.).any():
            return state / state.max()
        else:
            return state

"""
Notes:
agent_hparams:
    mem_size: Size of replay buffer (default = 10000 ?)
    gamma: discount rate (default = 0.99 ?)
    device: 'cpu' or 'gpu'
    batch_sz: size of batches from replay buffer
    
"""


