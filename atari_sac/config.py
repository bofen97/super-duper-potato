hyperparam = {}
hyperparam['lr'] = 3e-4
hyperparam['max_sample_steps']= int(1e7)
hyperparam['gamma'] = .99
hyperparam['temperature'] = .2
hyperparam['env_name'] = "LunarLanderContinuous-v2"
hyperparam['env_num'] = 1
hyperparam['batch_size'] = 256
hyperparam['smooth_weight'] = 0.005
hyperparam['replay_buffer_size'] = 1000000

