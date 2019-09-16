
import gym
import parl
from parl.env.atari_wrappers import wrap_deepmind
from acer import ACER_unTRoptimize
from model import AtariModel
from agent import Agent
from actor import Actor
from parl.utils.time_stat import TimeStat
from parl.utils.window_stat import WindowStat
import queue
import threading
import numpy as np
from collections import defaultdict
import time
from tensorboardX import SummaryWriter
class Learner(object):

    def __init__(self,config):
        self.config = config
        env = gym.make(config['env_name'])
        env= wrap_deepmind(env,dim=config['env_dim'],obs_format='NCHW')
        self.config['obs_shape'] = env.observation_space.shape
        self.config['act_dim'] = env.action_space.n
        model = AtariModel(self.config['act_dim'])
        alg =ACER_unTRoptimize(model,self.config)
        self.agent = Agent(alg,self.config)


        self.total_loss_stat = WindowStat(100)
        self.pi_loss_stat = WindowStat(100)
        self.vf_loss_stat = WindowStat(100)
        self.entropy_stat = WindowStat(100)
        self.lr = None
        self.entropy_coeff = None
        self.writer = SummaryWriter()

        self.learn_time_stat = TimeStat(100)
        self.start_time = None


        self.remote_count = 0
        self.sample_data_queue = queue.Queue()

        self.remote_metrics_queue = queue.Queue()
        self.sample_total_steps = 0

        self.params_queues = []
        self.create_actors()

    def create_actors(self):

        parl.connect(self.config['master_address'])

        for i in range(self.config['actor_num']):
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
        cnt=0
        while True:
            latest_params  = params_queue.get()
            remote_actor.set_weights(latest_params)

            batch = remote_actor.sample()

            self.sample_data_queue.put(batch)

            cnt += 1

            if cnt % self.config['get_remote_metrics_interval'] == 0:
                metrics = remote_actor.get_metrics()
                if metrics:
                    self.remote_metrics_queue.put(metrics)
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
            
        obs = train_batch['obs']
        act = train_batch['act']
        v = train_batch['v']
        qi = train_batch['qi']
        rhoi = train_batch['rhoi']
        f = train_batch['f']
        rho = train_batch['rho']
        q = train_batch['q']
        qret =train_batch['qret']
        c = train_batch['c']

        with self.learn_time_stat:
            total_loss,pi_loss,entropy,qloss,lr,entropy_coeff = \
                self.agent.learn(obs,act,v,qi,rhoi,f,rho,q,qret,c)
                
        self.total_loss_stat.add(total_loss)
        self.pi_loss_stat.add(pi_loss)
        self.vf_loss_stat.add(qloss)
        self.entropy_stat.add(entropy)
        self.lr=lr
        self.entropy_coeff = entropy_coeff
    def log_metrics(self):

        if self.start_time is None:
            return
        
        metrics=[]

        while True:
            try:
                metric = self.remote_metrics_queue.get_nowait()
                metrics.append(metric)
            except queue.Empty:
                break
        episode_rewards, episode_steps = [], []
        for x in metrics:
            episode_rewards.extend(x['episode_rewards'])
            episode_steps.extend(x['episode_steps'])
        max_episode_rewards, mean_episode_rewards, min_episode_rewards, \
                max_episode_steps, mean_episode_steps, min_episode_steps =\
                None, None, None, None, None, None
        if episode_rewards:
            mean_episode_rewards = np.mean(np.array(episode_rewards).flatten())
            max_episode_rewards = np.max(np.array(episode_rewards).flatten())
            min_episode_rewards = np.min(np.array(episode_rewards).flatten())

            mean_episode_steps = np.mean(np.array(episode_steps).flatten())
            max_episode_steps = np.max(np.array(episode_steps).flatten())
            min_episode_steps = np.min(np.array(episode_steps).flatten())
            print(self.sample_total_steps)
        metric = {
            'Sample steps': self.sample_total_steps,
            'max_episode_rewards': max_episode_rewards,
            'mean_episode_rewards': mean_episode_rewards,
            'min_episode_rewards': min_episode_rewards,
            'max_episode_steps': max_episode_steps,
            'mean_episode_steps': mean_episode_steps,
            'min_episode_steps': min_episode_steps,
            'total_loss': self.total_loss_stat.mean,
            'pi_loss': self.pi_loss_stat.mean,
            'vf_loss': self.vf_loss_stat.mean,
            'entropy': self.entropy_stat.mean,
            'learn_time_s': self.learn_time_stat.mean,
            'elapsed_time_s': int(time.time() - self.start_time),
            'lr': self.lr,
            'entropy_coeff': self.entropy_coeff,
        }

        for key, value in metric.items():
            if value is not None:
                self.writer.add_scalar(key, value, self.sample_total_steps)
    def should_stop(self):
        return self.sample_total_steps >= self.config['max_sample_steps']
    

    


        

if __name__ == '__main__':
    from config import config
    
    learner = Learner(config)
    assert config['log_metrics_interval_s'] > 0

    while not learner.should_stop():
        start = time.time()
        while time.time() - start < config['log_metrics_interval_s']:
            learner.step()
        learner.log_metrics()










