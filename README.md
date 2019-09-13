super-duper-potato
The dependency package is as follows (this is not complete),maybe you still need matplotlib.

algorithm:
         S AMPLE E FFICIENT A CTOR -C RITIC WITH EXPERIENCE REPLAY [1] . 

requirements: 
	rlschool, 
	paddlepaddle-gpu==1.5.1.post97 ,
	parl 

I think using conda will be more "safe" , view setup.sh.


Using xparl[2][3] parallel tools .
Xparl is an efficient parallel tool that is very easy to use :) 



xparl start --port 6006 
cd ACER/ 
python learner.py 




[1]arxiv:1611.01224v2 .
[2]https://parl.readthedocs.io/en/latest/parallel_training/overview.html  .
[3]https://github.com/paddlepaddle/parl   .
