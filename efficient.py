import os,io,sys
import torch.nn as nn 
import torch
import numpy as np
import cv2

# *** constants *** 

LR = 3e-4 
EPOCHS = 100 
IMG_SIZE = 224 # will try 299 later.
MAX_IMG_PER_FOLDER = 128 # we need to limit this ,cause M1 is not that powerful.
BSIZE = 128

def load_and_scale_image(xpath):
    img = cv2.imread(xpath)
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    return np.array(img,dtype=np.float32)


class Dataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.x = [] 
        self.y = [] 

        dir = os.listdir("./train.X1")
    
        omk = 0 


        for folder in dir:
            folder_dir = os.listdir("./train.X1/"+folder)
            
            for i in range(MAX_IMG_PER_FOLDER):
                img = folder_dir[i]
                print(img)
                img = load_and_scale_image("./train.X1/"+folder+"/"+img)
                self.x.append(img)
                self.y.append(np.array([omk]))
                omk+=1


        self.x = torch.tensor(np.array(self.x)).float()
        self.y = torch.tensor(np.array(self.y)).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self,d):
        return self.x[d],self.y[d]


"""

    How does vector database work?

    Well as i understand, they are being sorted with nn.Embedding layer, and then saved somewhere.
    then when we are looking for an image in in some random class, what we do is we do not just use softmax and boom thats it
    that is very linear search , and vector databases need dynamic search.

    images that when nn.embedding layer is applied, look most likely the same, with some prefixed minimal distance.

    why are we using efficient net then?

    Because instead of applying softmax on the last layer.
    we are going to apply embedding layer with number of our images
    then the one that is the nearest there and looks like the nearest, is the image index that we will return.
    makes sense?

    Example:

    NUM_OF_IMGS= 4096
    X = 3 
    nn.Embedding(num_embeddings=NUM_OF_IMGS,embedding_dim=X)

    First order of job: implement efficient net.

    
    !!! 224x224 in early ConvNets, modern ConvNets tend to use 299x299 (Szegedy et al., 2016) or 331x331
(Zoph et al., 2018) for better accuracy !!!



"""
# Biggest help i ever got in my entire life for the actual MBConv block : 
# https://paperswithcode.com/method/inverted-residual-block 



class MBConv1(nn.Module):
    def __init__(self) -> None:
        super(MBConv1,self).__init__()  

        self.swish = nn.SiLU()
        self.sig =nn.Sigmoid()
        self.dwc_1 = nn.Conv2d(32,16,kernel_size=3,stride=1,padding=1,groups=2)
        self.bn = nn.BatchNorm2d(16)


    
    def forward(self,x):
        x = x.reshape(1,32,224,224)
        x = self.swish(self.bn(self.dwc_1(x)))
        print(x.shape)

        exit(1)
        return x



class EfficientNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #will return 32,224,224 ? 
        self.c1 = nn.Conv2d(3,32,3,1,1)
        self.rel = nn.ReLU() 
        self.mb1 = MBConv1()
 
    def forward(self,x):
        x = self.c1(x)
        x =self.rel(x) #probably not a good idea. still down to try it.
        x = self.mb1(x)
    
        print(x.shape)
        return x


def train_EfficientNet():

    loss = nn.CrossEntropyLoss()

    data =Dataset()
    net = EfficientNet()
    img = load_and_scale_image("./crap.png").reshape(3,IMG_SIZE,IMG_SIZE)
    loader = torch.utils.data.DataLoader(data,batch_size=BSIZE,shuffle=True)
    #net(img)

    optimizer = torch.optim.Adam(net.parameters(),lr=LR)

    for e in range(EPOCHS):
        floss = 0 
        nrm = 0 

        for b,(x,y) in enumerate(loader):

            optimizer.zero_grad()

            out = net(x)

            ls = loss(out,y)
            ls.backward()
            optimizer.step()
            floss+=ls.item()
            nrm+=1
            print(f"** Epoch: {e} |  Loss: {floss/nrm} ")


env = os.getenv("RUN_TYPE")
assert(env != None)
env = int(env)
if(env == 1):
    train_EfficientNet()


