#model.py
import parl
class LiftSimModel(parl.Model):
    
    def __init__(self,act_dim):
        self.__fc1 = parl.layers.fc(size=200,act='tanh')
        self.__fc2 = parl.layers.fc(size=200,act='tanh')
        self.__head_logit = parl.layers.fc(size=act_dim,act=None)
        self.__head_value    =parl.layers.fc(size=1,act=None)
        
    def forward(self,x):
        x = self.__fc1(x)
        x = self.__fc2(x)
        logit = self.__head_logit(x)
        v     = self.__head_value(x)
        return logit,v
    def __call__(self,x):
        return self.forward(x)