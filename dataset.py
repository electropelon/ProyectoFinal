import csv
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np

import torch
from torch.utils import data


data = []
"""with open("sonar.csv") as csvfile:
	reader = csv.reader(csvfile)
	for row in reader: 
		data.append(row)
	

random.shuffle(data)



with open('sonarNew.csv','w',newline="") as file:
	writer = csv.writer(file)
	writer.writerows(data) 

"""	
def data():
	data_sonar = []
	data_resultado = []
	data = []
		
	with open('sonarNew.csv') as file:
		reader = csv.reader(file)
		for row in reader: 
			data.append(row)
	data = np.array(data)		
	
		
							
	data_sonar = data[:,0:60]
	data_sonar = data_sonar.astype(np.float)
	print ("dato de un elemento",type(data_sonar[1,1]))
	data_resultado = data[:,60]
	
	
	for i in range(0,len(data_resultado)):
		if data_resultado[i] == "R": 
				
			data_resultado[i] = 0  
		else: 
			data_resultado[i] = 1
	data_resultado = data_resultado.astype(np.float)
	data_resultado.resize((208,1))
	#print (data_resultado,type)		
	return (data_sonar,data_resultado)


class Dataset_my(Dataset):
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        y = self.labels[index]
        X = self.list_IDs[index]

        return X, y