super-duper-potato \n
The dependency package is as follows (this is not complete),maybe you still need matplotlib. \n

requirements: \n
	rlschool \n
	paddlepaddle-gpu==1.5.1.post97 \n
	parl \n

I think using conda will be more "safe". \n

conda create liftsim python=3.6 \n
conda activate liftsim \n
pip install -r requirements.txt \n



Using xparl[1][2] parallel tools \n
Xparl is an efficient parallel tool that is very easy to use :) \n

xparl start --port 6006 \n
cd ACER/ \n
python learner.py \n





[1]https://parl.readthedocs.io/en/latest/parallel_training/overview.html \n
[2]https://github.com/paddlepaddle/parl \n
