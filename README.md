super-duper-potato
The dependency package is as follows (this is not complete),maybe you still need matplotlib.

requirements: 
	rlschool, 
	paddlepaddle-gpu==1.5.1.post97 ,
	parl 

I think using conda will be more "safe" , view setup.sh.


Using xparl[1][2] parallel tools 
Xparl is an efficient parallel tool that is very easy to use :) 


xparl start --port 6006 
cd ACER/ 
python learner.py 





[1]https://parl.readthedocs.io/en/latest/parallel_training/overview.html 
[2]https://github.com/paddlepaddle/parl 
