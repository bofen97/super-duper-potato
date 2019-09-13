import parl
class  PolicyModel(parl.Model):
    def __init__(self,act_dim):
        self.act_dim = act_dim
        self.__fc1 = parl.layers.fc(size=256,act='tanh')  
        self.__fc2 = parl.layers.fc(size=256,act='tanh')
        self.__fc3 = parl.layers.fc(size=128,act='tanh')
        
        self.__head_logit = parl.layers.fc(size=self.act_dim,act=None)
        self.__head_q    =parl.layers.fc(size=self.act_dim,act=None)
        
    def forward(self,x):
        x = self.__fc1(x)
        x = self.__fc2(x)
        logit = self.__head_logit(x)
        q     = self.__head_q(x)
        return logit,q
    

    def __call__(self,x):
        return self.forward(x)
