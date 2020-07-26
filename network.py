import math
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim

class perceptron(nn.Module):
	def __init__(self):
		super(perceptron,self).__init__()

		self.layer1 = nn.Linear(60, 4)
		self.layer2 = nn.Linear(4, 2)
		self.layer3 = nn.Linear(2, 1)
		#self.layer4 = nn.Linear(16,1)
		self.activation = nn.Tanh()
		self.activationF = nn.Sigmoid()
		
	def forward(self,x):
		x = self.layer1(x)
		x = self.activation(x)
		
		x = self.layer2(x)
		x = self.activation(x)
		
		x = self.layer3(x)
	
		output = self.activationF(x)
		return (output)
		
		
class resultado():
	def __init__(self,Y_r,Y_p):
		
		self.Y_r = Y_r
		self.Y_p = Y_p
	
		self.TP = 0
		self.TN = 0 
		self.FP = 0 
		self.FN = 0
		#print ("dentro de resultado",self.Y_p)
		self.longitud = len(Y_r)
		#print ("entrada real:",Y_r,"entrada predicha:",Y_p,self.longitud)
		
	def clasificacion(self):
		Y_p_entera = self.define_TF()
		#print ("entrada real:",self.Y_r,"entrada predicha entera:",Y_p_entera,self.longitud)
		for i in range(self.longitud):
			if self.Y_r[i]==Y_p_entera[i]: 
				if self.Y_r[i]==1:
					self.TP +=1
				else:
					self.TN +=1
			else:
				if self.Y_r[i]==1:
					self.FN +=1
				else:
					self.FP +=1
		print("TP:",self.TP,"\nTN:",self.TN,"\nFP:",self.FP,"\nFN:",self.FN)
				
	def define_TF(self):
		for i in range(self.longitud):
			
			if self.Y_p[i]>= 0.5:
				self.Y_p[i] = 1
			else:
				self.Y_p[i] = 0
		return (self.Y_p)
		
	def exactitud(self):
		self.e = (self.TP+self.TN)/(self.longitud)
		print ("Exactitud:",self.e)
		return (self.e)
		
	def precision(self):
		self.p = self.TP/(self.TP+self.FP)
		print ("Precision",self.p)
		return (self.p)
		
	def pf1(self):
		self.f1 = (2*self.precision()*self.exactitud())/(self.precision()+self.exactitud())	
		print ("Puntuación F1",self.f1)
		return(self.f1)			
	
	
			
			
	
			

	
		
