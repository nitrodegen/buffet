import os,io,sys
import torch.nn as nn 
import torch
import numpy as np
import cv2
from math import * 

# *** constants *** 
EPOCHS= 120
LR=0.0001
MAX_IMG_PER_FOLDER = int(os.getenv("MAXIMG")) # we need to limit this ,cause M1 is not that powerful.
BSIZE = int(os.getenv("BSIZE"))
NUM_OF_CLASSES = 25
IMG_SIZE=224
DATA_FOLDER = os.getenv("DIR_PATH")

def load_and_scale_image(xpath):
    img = cv2.imread(xpath)
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    return np.array(img,dtype=np.float32)


class Dataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()

        self.precision  = 0.5
        self.x = [] 
        self.y = [] 
        
        dir = os.listdir(DATA_FOLDER)
    
        omk = 0 


        for folder in dir:
            folder_dir = os.listdir(DATA_FOLDER+folder)
            
            for i in range(MAX_IMG_PER_FOLDER):
                img = folder_dir[i]
                img = load_and_scale_image(DATA_FOLDER+folder+"/"+img)
             
                self.x.append(img)
                self.y.append(np.array([omk]))
                
            omk+=1

        self.embedding_vector  = [[ ] for i in range(omk)]
        self.x = torch.tensor(np.array(self.x)).float()
        self.y = torch.tensor(np.array(self.y)).float()

        for i in range(len(self.x)):
            x,y = self.__getitem__(i)
            y= int(y.detach().numpy().tolist()[0])
            self.embedding_vector[y].append(x.numpy())

        self.embedding_vector = torch.tensor(np.array(self.embedding_vector))  
        

    def get_embedding_item(self,img,img_class):
        
        img_mean_fin = torch.mean(img)
        out = None

        oud = self.embedding_vector[img_class]
        
        means = [] 

        for img in oud:
            img_mean = torch.mean(img)
            means.append(torch.abs(img_mean-img_mean_fin))

        mind = means[0]
        indx = 0 
        for i in range(len(means)):
            if(means[i] < mind):
                mind = means[i]
                indx = i

        return self.embedding_vector[img_class][indx]
    



    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self,d):
        return self.x[d],self.y[d]