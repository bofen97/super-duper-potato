import parl
from paddle import fluid
class AtariModel(parl.Model):
    """
    only use for atari
    
    
    """
    
    def __init__(self,actdim):
        
        self.conv1 = parl.layers.conv2d(num_filters=32,filter_size=8,stride=4,padding=1,act='relu')
        self.conv2 = parl.layers.conv2d(num_filters=64,filter_size=4,stride=2,padding=2,act='relu')
        self.conv3 = parl.layers.conv2d(num_filters=64,filter_size=3,stride=1,padding=0,act='relu')
        self.fc = parl.layers.fc(size=512,act='relu')
        
        self.policy_fc = parl.layers.fc(size=actdim,act=None)
        self.value_fc  = parl.layers.fc(size=1,act=None)
        
    def policy(self,obs):
        obs = obs / 255.0
        
        conv1 = self.conv1(obs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        
        flatten = fluid.layers.flatten(conv3,axis=1)
        
        fc_output = self.fc(flatten)
        
        policy_logits = self.policy_fc(fc_output)
        return policy_logits
    def value(self,obs):
        obs = obs / 255.0
        
        conv1 = self.conv1(obs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        
        flatten = fluid.layers.flatten(conv3,axis=1)
        
        fc_output = self.fc(flatten)
        
        values = self.value_fc(fc_output)
        values = fluid.layers.squeeze(values,axes=[1])
        return values
    def policy_and_value(self,obs):
        
        """
        alg sample use it .
        
        INPUT : [ B,OBSERVATION_SPACE]
        OUTPUT: [BATCHSIZE ACTDIM], [BATCHSIZE]
        """
        obs = obs / 255.0
        
        conv1 = self.conv1(obs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        
        flatten = fluid.layers.flatten(conv3,axis=1)
        
        fc_output = self.fc(flatten)
        
        policy_logits = self.policy_fc(fc_output)
        values = self.value_fc(fc_output)
        # squeeze ..
        values = fluid.layers.squeeze(values,axes=[1])
        return policy_logits,values