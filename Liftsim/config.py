
config = {

    #========== algorithms =============
    'algorithm':'a2c', # default alg.
    #==========  remote config ==========
    'master_address': 'localhost:6006',

    #==========  actor config ==========
    'actor_num':5, 
    'env_num': 5,
    'sample_batch_steps': 5,

    #==========  learner config ==========
    'max_sample_steps': int(2e8),
    'gamma': 0.99,
    'lambda': 0.97,  # GAE

    # start learning rate
    'start_lr':1e-5,
    'l2_reg':5e-5,

    # coefficient of policy entropy adjustment schedule: (train_step, coefficient)
    'entropy_coeff_scheduler': [(0, -0.01)],
    'vf_loss_coeff': 0.5,
    'get_remote_metrics_interval': 10,
    'log_metrics_interval_s': 10,
}
