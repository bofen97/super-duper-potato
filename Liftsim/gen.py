from wrapper import Wrapper,ActionWrapper,ObservationWrapper
from rlschool import LiftSim
if __name__ == "__main__":
    mansion_env = LiftSim()
    mansion_env = Wrapper(mansion_env)
    mansion_env = ActionWrapper(mansion_env)
    mansion_env = ObservationWrapper(mansion_env)

    mansion_env.reset()
    rs =0.
    for i in range(28800*6):
        mansion_env.render()
        acts = [1,2,3,4]
        _,r,_,_ = mansion_env.step(acts)