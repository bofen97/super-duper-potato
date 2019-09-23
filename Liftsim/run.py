import time
from config import config
from Learner import Learner
if __name__ == "__main__":
    learner = Learner(config)
    learner.agent.restore('./model.ckpt')
    print("==========restored==========")
    assert config['log_metrics_interval_s']>0
    while not learner.should_stop():
        start = time.time()
        while time.time() - start < config['log_metrics_interval_s']:
            learner.step()
        learner.log_metrics()
        learner.agent.save('./model.ckpt')
