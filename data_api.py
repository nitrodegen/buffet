import os,io,sys
import torch.nn as nn 
import torch
import numpy as np
import cv2
import time
from math import * 

# *** constants *** 
EPOCHS= 120
LR=0.001
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

        self.precision  = 0.3
        self.x = [] 
        self.y = [] 
        
        dir = os.listdir(DATA_FOLDER)
    
        omk = 0 


        for folder in dir:
            folder_dir = os.listdir(DATA_FOLDER+folder)
            
            for i in range(MAX_IMG_PER_FOLDER):
                img = folder_dir[i]
                img = load_and_scale_image(DATA_FOLDER+folder+"/"+img)
                img/=255

                self.x.append(img)
                self.y.append(np.array([omk]))
                
            omk+=1

        self.embedding_vector  = {
            i:{i:None for i in range(MAX_IMG_PER_FOLDER)} for i in range(omk)
        }

        self.mapped_imgs = { 
            i:[] for i in range(omk)
        }

        for i in range(len(self.x)):
            x,y = self.__getitem__(i)

            y= int(y.tolist()[0])
            self.mapped_imgs[y].append(x)


        self.x = torch.tensor(np.array(self.x)).float()
        self.y = torch.tensor(np.array(self.y)).float()

        indt = 0 

        for i in range(0,len(self.x)):
            if(indt >= MAX_IMG_PER_FOLDER):
                indt = 0 

            x,y = self.__getitem__(i)
   
            x = x.reshape(3,IMG_SIZE,IMG_SIZE)
            y=int(y.detach().numpy().tolist()[0])
        
        
            calc_formula = torch.mean(x)
            self.embedding_vector[y][indt] = calc_formula
            indt+=1


    def get_embedding_item(self,img,img_class):

        img = torch.mean(img)
        img_class_cut = self.embedding_vector[img_class]
        diffs = self.precision
        indx = None 
        for i in range(MAX_IMG_PER_FOLDER):
            diff = abs(img-img_class_cut[i])
            if(diff <= self.precision):
                diffs = diff
                indx = i 

        indx = self.mapped_imgs[img_class][indx]
        return indx


    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self,d):
        return self.x[d],self.y[d]