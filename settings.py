from _collections import OrderedDict

env_hparams = OrderedDict(
    side_len              = 21,
    death_rate            = 0.1,
    disp_sigma            = 0.2,
    erad_alpha            = 6,
    erad_beta             = 2,
    k                     = 10.,
    mgmt_cost             = 1.,
    eradication_bonus     = 100.,
    proliferation_penalty = 100000.,
    proportion_occupied   = 0.25,
    n_pop_ini             = 100,  # max per occupied cell
    reward_method         = 'count',
    normalise_reward      = True,  # scale penalty from population by grid size
    normalise_cost        = True,  # scale management cost by grid size
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
    strides      = [1, 1, 1, 2, 2],
    dense_units  = env_hparams['side_len'] ** 2,
    epsilon      = 5e-2,
    device       = agent_hparams['device'],
)
critic_hparams = OrderedDict(
    lr           = 1e-5,
    wd           = 0.,
    kernel_sizes = [5, 5, 3, 3, 3],
    channels     = [1, 16, 32, 64, 128, 256],  # always prepend channel widths with 1 (for the input, which has 1 channel).
    strides      = [1, 1, 1, 2, 2],
    device       = agent_hparams['device']
)
run_hparams = OrderedDict(
    n_episodes         = 50,
    n_steps_per_ep     = 200,
    save_plots         = True,
    snapshot_interval  = 20,
)
