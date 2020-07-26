from dataset import*
from network import*
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler

def graficar(x,y,ejex,ejey):
	plt.style.use('seaborn-whitegrid')	
	plt.plot(x,y)
	plt.ylabel(ejey)
	plt.xlabel(ejex)
	plt.show()	


if __name__ == "__main__":
	
	
	x,y = data()
	validation_test_split = 0.4
	train_split = 0.6
	longitud = 208
	indices = list(range(longitud))
	
	split = int(np.floor(validation_test_split * longitud))
	split2 = int(np.floor(train_split * longitud))

	#train_indices, val_indices, test_indices = indices[2*split:], indices[:split], indices [(split+1):(2*(split-1)]
	train_indices, val_indices = indices[split:], indices[:split]
	
	train_sampler = SubsetRandomSampler(train_indices)
	valid_sampler = SubsetRandomSampler(val_indices)
	
	
	
	modelo = perceptron()
	loss = nn.MSELoss()
	
	data = Dataset_my(x,y)
	#print ("training set",data[0])
	dataloader = DataLoader(dataset=data, batch_size = 62	,sampler = train_sampler, shuffle = False, num_workers = 2)




	optimizador = optim.Adam(modelo.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, amsgrad=False)
	
	perdidas = np.array([])

	menor_cost = 100000
	epoca = 3000
	for z in range(epoca):
		running_loss = 0.0
		
		for i, data1 in enumerate(dataloader):
			inputs,labels = data1
		
			
			modelo.zero_grad()
			output = modelo.forward(inputs.float())
			e = loss(output, labels.float())
			e.backward()
			optimizador.step()
			running_loss +=e 

			
			if (i+1)%2 == 0:
	
			    if menor_cost > (running_loss/2):
				    menor_cost = running_loss/2
				    torch.save(modelo,'modelo.pth')
				    mejor_epoca = z
					
					
			    perdidas = np.append(perdidas, (running_loss/2).detach().numpy())
			    print ("costo:",(running_loss/2))
				
				
			    print ("epoca:",z+1)
			    running_loss = 0.0
				
	
			
	graficar(np.arange(epoca),perdidas,"Epocas","Costo")

	dataloader_aux = DataLoader(dataset=data, batch_size =124,sampler = train_sampler, shuffle = False, num_workers = 2)
	

	for data2 in  dataloader_aux:
		inputs1,labels1 = data2
		print ("menor_cost:",menor_cost,"mejor_epoca:",mejor_epoca)
		modelo = torch.load('modelo.pth')
		output = modelo.forward(inputs1.float())
		
	
		matriz = resultado(labels1,output)
		matriz.clasificacion()
		break
