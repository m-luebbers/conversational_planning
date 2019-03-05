"""
***********************************************************
File: juliaController.py
Author: Luke Burks
Date: May 2018

Implements a subclasses Controller which calls the 
JuliaPOMDP packages implementation for online planning
and control.
Also implements a recreation function which accounts for 
model changes

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


import julia; 
from robotControllers import Controller


class JuliaController(Controller):


	def __init__(self,model=None):
		self.model = model; 
		# self.julia = julia.Julia(); 
		# self.julia.include("julia_POMCP_Controller.jl");  
		# self.makeSolver = self.julia.eval("makeSolver"); 
		# self.getPOMCPAction = self.julia.eval("getPOMCPAction"); 
		# self.initDist = self.julia.eval('initial_state_distribution'); 
		# [solver,pomdp,planner] = self.makeSolver();
		# self.solver = solver; 
		# self.pomdp = pomdp; 
		# self.planner = planner; 
		# self.b = self.initDist(pomdp); 


	def getActionKey(self):
		# act = self.getPOMCPAction(self.planner,self.b); 
		return act; 

if __name__ == '__main__':
	c = JuliaController(); 
	print(c.getActionKey()); 