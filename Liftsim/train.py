from model import MLP
from algorithms import ppo,a2c
from agent import Agent
from actor import Actor
from config import config
from wrapper import Wrapper,ActionWrapper,ObservationWrapper
from rlschool import LiftSim
import queue
from parl.utils.window_stat import WindowStat
from parl.utils.time_stat import TimeStat
from tensorboardX import SummaryWriter
import parl
import threading
import time
import numpy as np
from collections import defaultdict
class Learner(object):
    def __init__(self,config):
        self.config = config
        # get action space & observation space
        mansion_env = LiftSim()
        mansion_env = Wrapper(mansion_env)
        mansion_env = ActionWrapper(mansion_env)
        mansion_env = ObservationWrapper(mansion_env)
        self.config['act_dim'] = mansion_env.action_space
        self.config['obs_shape'] = (mansion_env.observation_space,)
        model=MLP(self.config['act_dim'])
        if self.config['algorithm'] =='a2c':
            print("algorithm is a2c .  ") 
            algorithm= a2c(model,self.config)
        elif self.config['algorithm'] =='ppo':
            print("algorithm is ppo . ")
            algorithm= ppo(model,self.config)
        else:
            algorithm = None

        self.agent = Agent(algorithm,self.config)
        self.total_loss_stat = WindowStat(100)
        self.pi_loss_stat = WindowStat(100)
        self.vf_loss_stat = WindowStat(100)
        self.entropy_stat = WindowStat(100)
        self.lr = None
        self.rewards_sum_stat = WindowStat(100)

        self.writer = SummaryWriter()
        self.learn_time_stat = TimeStat(100)
        self.start_time = None

        self.remote_count = 0
        self.sample_data_queue = queue.Queue()
        self.sample_total_steps = 0
        self.params_queues = []


        self.create_actors()

    def create_actors(self):
        parl.connect(self.config['master_address'])
        for _ in range(self.config['actor_num']):
            params_queue = queue.Queue()
            self.params_queues.append(params_queue)

            self.remote_count += 1

            remote_thread = threading.Thread(
                target=self.run_remote_sample,args=(params_queue,)
            )

            remote_thread.setDaemon(True)
            remote_thread.start()
        self.start_time = time.time()
    def run_remote_sample(self,params_queue):
        remote_actor = Actor(self.config)
        while True:
            latest_params = params_queue.get()
            remote_actor.set_weights(latest_params)
            batch = remote_actor.sample()
            self.sample_data_queue.put(batch)
    def step(self):
        latest_params = self.agent.get_weights()
        for params_queue in self.params_queues:
            params_queue.put(latest_params)
        
        train_batch = defaultdict(list)

        for i in range(self.config['actor_num']):
            sample_data = self.sample_data_queue.get()
            for key,value in sample_data.items():
                train_batch[key].append(value)
            self.sample_total_steps += sample_data['obs'].shape[0]
        for key,value in train_batch.items():
            train_batch[key] = np.concatenate(value)
        
        with self.learn_time_stat:
            total_loss,pi_loss,vf_loss,entropy,lr,_ = self.agent.learn(
                train_batch['obs'],train_batch['act'],train_batch['adv'],
                train_batch['vtag'])
        self.rewards_sum_stat.add(np.sum(train_batch['rews'])/(self.config['actor_num']*
                                                                                                                                self.config['env_num']*
                                                                                                                                self.config['sample_batch_steps']*4.0 ))
        self.total_loss_stat.add(total_loss)
        self.pi_loss_stat.add(pi_loss)
        self.vf_loss_stat.add(vf_loss)
        self.entropy_stat.add(entropy)
        self.lr = lr
    def log_metrics(self):
        if self.start_time is None:
            return
        metric = {
            'sample_reward_per_step_mean':self.rewards_sum_stat.mean,
            'pi_loss':self.pi_loss_stat.mean,
            'total_loss':self.total_loss_stat.mean,
            'vf_loss':self.vf_loss_stat.mean,
            'entropy':self.entropy_stat.mean,
            'lr':self.lr,
        }
        for key,value in metric.items():
            self.writer.add_scalar(key,value,self.sample_total_steps)
    def should_stop(self):
        return self.sample_total_steps >= self.config['max_sample_steps']

