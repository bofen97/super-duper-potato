import parl
from paddle import fluid
class MLP(parl.Model):
    
    
    def __init__(self,actdim):
        
        self.fc1 = parl.layers.fc(size=256,act='tanh')
        self.fc2 = parl.layers.fc(size=256,act='tanh')
        self.policy_fc = parl.layers.fc(size=actdim,act=None)
        self.value_fc  = parl.layers.fc(size=1,act=None)
        
    def policy(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        policy_logits = self.policy_fc(x)
        return policy_logits
    def value(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        values = self.value_fc(x)
        values = fluid.layers.squeeze(values,axes=[1])
        return values
    def policy_and_value(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        policy_logits = self.policy_fc(x)   
        values = self.value_fc(x)
        values = fluid.layers.squeeze(values,axes=[1])

        return policy_logits,values