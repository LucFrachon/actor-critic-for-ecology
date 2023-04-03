from _collections import OrderedDict

env_hparams = OrderedDict(
    side_len              = 31,
    death_rate            = 0.1,
    disp_sigma            = 0.2,
    erad_alpha            = 4,
    erad_beta             = 3,
    k                     = 10.,
    mgmt_cost             = 50.,
    eradication_bonus     = 100.,
    proliferation_penalty = 1000.,
    proportion_occupied   = 0.25,
    n_pop_ini             = 20,  # max per occupied cell
    reward_method         = 'sum',
    normalise_reward      = True,
    normalise_cost        = True,
)
agent_hparams = OrderedDict(
    mem_size         = 2048,
    gamma            = 0.95,
    device           = 'cuda',
    batch_sz         = 2048,  # must be < mem_size
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
