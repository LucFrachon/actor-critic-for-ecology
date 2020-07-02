from argparse import Namespace
import torch
import torch.nn as nn
import torch.optim as optim


class Critic(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, Namespace): self.hparams = hparams
        else: self.hparams = Namespace(**hparams)
        self._build_model()
        self.optim = self._configure_optimiser()

    @staticmethod
    def _conv_block(in_ch, out_ch, ksz, st):
        return nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, ksz, st, ksz // 2),
            nn.ReLU(inplace=True)
        )

    def _build_model(self):
        features = []
        channels = self.hparams.channels
        k_sizes = self.hparams.kernel_sizes
        strides = self.hparams.strides
        for i in range(len(k_sizes)):
            features.append(self._conv_block(channels[i], channels[i + 1], k_sizes[i], strides[i]))
        self.features = nn.Sequential(
            *features,
            nn.AdaptiveAvgPool2d(1)
        )
        self.regressor = nn.Sequential(nn.Flatten(),
                                       nn.Linear(channels[-1], 1),
                                       nn.Sigmoid())

    def _configure_optimiser(self):
        return optim.AdamW(self.parameters(), lr = self.hparams.lr, weight_decay = self.hparams.wd)

    def forward(self, states):
        h = self.features(states)
        return self.regressor(h)

    def loss_fn(self, td_targets, value_estimates):
        return (td_targets - value_estimates) ** 2

    def training_step(self, batch):
        states, td_targets = batch
        # if states.dim() == 2: states.unsqueeze_(0)
        # if td_targets.dim() == 1: td_targets.unsqueeze_(0)
        self.optim.zero_grad()
        with torch.enable_grad():
            value_estimates = self(states)
            loss_value = self.loss_fn(td_targets, value_estimates)
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
    strides,
    device: 'gpu' or 'cpu'

"""