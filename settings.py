from _collections import OrderedDict

env_hparams = OrderedDict(
    side_len          = 11,
    death_rate        = 0.01,
    disp_sigma        = 0.5,
    erad_alpha        = 4,
    erad_beta         = 3,
    k                 = 10.,
    mgmt_cost         = 1.,
    eradication_bonus = 1000.,
    n_pop_ini         = 10,
    reward_method     = 'count',
)
agent_hparams = OrderedDict(
    mem_size = 1000,
    gamma    = 0.99,
    device   = 'cpu',
    batch_sz = 16,  # must be < mem_size
)
actor_hparams = OrderedDict(
    lr           = 1e-3,
    wd           = 1e-5,
    kernel_sizes = [5, 3, 3],
    channels     = [1, 16, 32, 64],  # always prepend channel widths with 1 (for the input, which has 1 channel).
    strides      = [1, 2, 1],
    dense_units  = env_hparams['side_len'] ** 2,
    epsilon      = 0.01,
    device       = agent_hparams['device'],
)
critic_hparams = OrderedDict(
    lr           = 1e-3,
    wd           = 1e-5,
    kernel_sizes = [5, 3, 3],
    channels     = [1, 16, 32, 64],  # always prepend channel widths with 1 (for the input, which has 1 channel).
    strides      = [1, 2, 1],
    device       = agent_hparams['device']
)