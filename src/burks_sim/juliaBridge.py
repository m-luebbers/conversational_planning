"""
***********************************************************
File: juliaBridge.py
Author: Luke Burks
Date: May 2018

Allows for passing information between a python process and 
a julia process. Passes via standard in and out. 

***********************************************************
"""

__author__ = "Luke Burks"
__copyright__ = "Copyright 2018"
__credits__ = ["Luke Burks"]
__license__ = "GPL"
__version__ = "0.1.0"
__maintainer__ = "Luke Burks"
__email__ = "luke.burks@colorado.edu"
__status__ = "Development"

import julia

import time



def watch(j,solver,pomdp,planner):

	getPOMCPAction = j.eval("getPOMCPAction"); 
	makeDist = j.eval("makeDist"); 
	makeMixtureDist = j.eval("makeMixtureDist")


	while(True):
		try:

			cp0 = []; 
			cp1 = []; 
			mmean0 = []; 
			mmean1 = []; 
			cov0 = []; 
			cov1 = []; 
			weights = []; 

			num_mix = int(raw_input()); 
			for i in range(0,num_mix):
				cp0.append(float(raw_input()));
				cp1.append(float(raw_input())); 
				mmean0.append(float(raw_input())); 
				mmean1.append(float(raw_input())); 
				cov0.append(float(raw_input())); 
				cov1.append(float(raw_input()));
				weights.append(float(raw_input())); 


			#initDist = makeDist(cp0,cp1,mmean0,mmean1,cov0,cov1); 
			initDist = makeMixtureDist(cp0,cp1,mmean0,mmean1,cov0,cov1,weights); 
			act = getPOMCPAction(planner,initDist); 

			print(act); 
		except IOError as e:
			if(e.errno == 11):
				time.sleep(0.1); 


		

def main():
	j = julia.Julia(); 
	j.include("julia_POMCP_Controller.jl");
	makeSolver = j.eval("makeSolver"); 

	[solver,pomdp,planner] = makeSolver();

	watch(j,solver,pomdp,planner); 


if __name__ == '__main__':
    main()