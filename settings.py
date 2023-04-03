from _collections import OrderedDict

env_hparams = OrderedDict(
    side_len          = 35,
    death_rate        = 0.1,
    disp_sigma        = 0.2,
    erad_alpha        = 6,
    erad_beta         = 2,
    k                 = 10.,
    mgmt_cost         = 10.,
    eradication_bonus = 300.,
    n_pop_ini         = 20,  # max per occupied cell
    reward_method     = 'sum',
    normalise_reward  = True
)
agent_hparams = OrderedDict(
    mem_size         = 2000,
    gamma            = 0.95,
    device           = 'cuda',
    batch_sz         = 64,  # must be < mem_size
    normalise_states = True,
)
actor_hparams = OrderedDict(
    lr           = 1e-5,
    wd           = 0.,
    regul_rate   = 1e-4,  # sparse activity regulariser
    kernel_sizes = [5, 5, 3, 3, 3],
    channels     = [1, 16, 32, 64, 128, 256],  # always prepend channel widths with 1 (for the input, which has 1 channel).
    strides      = [1, 2, 1, 2, 2],
    dense_units  = env_hparams['side_len'] ** 2,
    epsilon      = 1e-3,
    device       = agent_hparams['device'],
)
critic_hparams = OrderedDict(
    lr           = 1e-5,
    wd           = 0.,
    kernel_sizes = [5, 5, 3, 3, 3],
    channels     = [1, 16, 32, 64, 128, 256],  # always prepend channel widths with 1 (for the input, which has 1 channel).
    strides      = [1, 2, 1, 2, 2],
    device       = agent_hparams['device']
)
