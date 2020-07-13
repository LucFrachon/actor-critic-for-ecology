from argparse import Namespace
import torch
import torch.nn as nn
import torch.optim as optim


class Actor(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, Namespace):
            self.hparams = hparams
        else:
            self.hparams = Namespace(**hparams)
        self._build_model()
        self.optim = self._configure_optimiser()

    @staticmethod
    def _conv_block(in_ch, out_ch, ksz, st):
        return nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, ksz, st, ksz // 2),
            nn.ReLU(inplace = True)
        )

    def _build_model(self):
        channels = self.hparams.channels
        k_sizes = self.hparams.kernel_sizes
        strides = self.hparams.strides

        state_features = []
        for i in range(len(k_sizes)):
            state_features.append(self._conv_block(channels[i], channels[i + 1], k_sizes[i], strides[i]))
        self.state_features = nn.Sequential(
            *state_features,
            nn.AdaptiveAvgPool2d(1)
        )
        # action_features = []
        # for i in range(len(k_sizes)):
        #     action_features.append(self._conv_block(channels[i], channels[i + 1], k_sizes[i], strides[i]))
        # self.action_features = nn.Sequential(
        #     *action_features,
        #     nn.AdaptiveAvgPool2d(1)
        # )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], self.hparams.dense_units, bias=True),
            nn.Sigmoid()
        )

    def _configure_optimiser(self):
        return optim.AdamW(self.parameters(), lr = self.hparams.lr, weight_decay = self.hparams.wd)

    def forward(self, states):
        h_states = self.state_features(states)
        # h_actions = self.action_features(actions)
        return self.classifier(h_states)
                               # ).clamp(min=self.hparams.epsilon)

    def loss_fn(self, td_errors, action_probs, actions):
        # Scalar actions: loss = - log(pi(action|state) * td_target
        # Vector-valued actions, assuming independence of dimensions:
        #     loss = -log(pi(action|state) * td_error)
        #          = -log(pi(a_1|state) * pi(a_2|state) * ... * pi(a_n|state)) * td_error
        #          = -td_error * sum(log(pi(a_i|state)))  ?? hopefully
        # but clamp pi(*) values above some small constant to avoid zero everywhere?
        # Add regulariser to prevent all probs to go to 1 (trivial minimal)
        action_probs = action_probs * actions  # mask action_probs with actual action locations
        action_probs = torch.clamp(action_probs, self.hparams.epsilon, 1 - self.hparams.epsilon)
        loss_val = -(td_errors * torch.sum(torch.log(action_probs), dim=1)).mean() \
            + self.hparams.regul_rate * torch.norm(action_probs, p=1, dim=1).mean()
        return loss_val

    def training_step(self, batch):
        states, td_errors, actions = batch
        # actions.reshape_(states.size())
        # if states.dim() == 2: states = states.unsqueeze(0)
        # if td_errors.dim() == 1: td_errors = td_errors.unsqueeze(0)
        self.optim.zero_grad()
        # with torch.enable_grad():
        action_probs = self(states)  # , actions)
        loss_value = self.loss_fn(td_errors, action_probs, actions)
        loss_value.backward()
        self.optim.step()
        return loss_value


"""
Notes:
hparams:
    lr: learning rate
    wd: weight decay
    kernel_sizes: list of length C (C = number of conv layers), 
    channels: list of length C + 1 (first value == input == 1)
    strides: list of length C (C = number of conv layers), 
    dense_units: int, needs to be equal to the number of cells in the environment,
    epsilon: small constant, lower bound for action probs
    device: 'gpu' or 'cpu'
"""