import os,io,sys
import numpy as np 
import torch
import torch.nn as nn 

sys.path.append("/Users/gavrilo/Desktop/vector_database/buffet")
if(os.getenv("TEST_PATH") != "MMAP_ASM"):
    os.environ["MAXIMG"] = str(32)
    os.environ["BSIZE"] = str(3)
    os.environ["DIR_PATH"]="../train.X1/"

from efficient import * 
from data_api import *
        
B0   =  (1.0, 1.0, 224, 0.2)

class Queuer(object):
    def __init__(self,model_path,dbg=False) -> None:
        super(Queuer,self).__init__()
        
        self.dataset  = Dataset()
        self.vals = torch.load(model_path)
        w,h,img_size,dropout = B0
        self.model = EfficientNet(w,h,dropout,len(self.dataset.x))
        self.model.load_state_dict(self.vals)
        if(dbg):
            print("*** model initialized *** ")
    
    def query_image(self,image):

        if(type(image) != np.ndarray and type(image) != torch.tensor):
            image = torch.tensor(np.array(image))

        elif(type(image) == np.ndarray):
            image = cv2.resize(image,(IMG_SIZE,IMG_SIZE))
            image= np.array(image,dtype=np.float32)
            image= torch.tensor(image).float()

        image= image.expand(3,IMG_SIZE,IMG_SIZE,3)
        image= image.reshape(3,3,IMG_SIZE,IMG_SIZE)
        out = self.model(image)
        out = out.clone().cpu().detach().numpy()[0]
        out = np.argmax(out,axis=-1)

        close_image= self.dataset.get_embedding_item(image,out)
        return close_image.numpy()
    