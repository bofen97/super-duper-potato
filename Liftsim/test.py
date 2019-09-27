from config import config
from Learner import Learner
from wrapper import Wrapper,ActionWrapper,ObservationWrapper
from rlschool import LiftSim
if __name__ == "__main__":
    learner = Learner(config)
    learner.agent.restore('./model.ckpt')
    mansion_env = LiftSim()
    mansion_env = Wrapper(mansion_env)
    mansion_env = ActionWrapper(mansion_env)
    mansion_env = ObservationWrapper(mansion_env)
    obs=mansion_env.reset()
    rs =0.
    for i in range(28800*6):
        mansion_env.render()
        obs = obs.reshape(1,-1)
        acts= learner.agent.predict(obs)
        acts = [int(a) for a in acts]
        obs,r,_,_ = mansion_env.step(acts)
        rs+=r
    print(rs)