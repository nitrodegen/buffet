import os,io,sys
import torch.nn as nn 
import torch
import numpy as np
import cv2

# *** constants *** 

LR = 3e-4
EPOCHS = 100 
IMG_SIZE = 224 # will try 299 later.
MAX_IMG_PER_FOLDER = 32 # we need to limit this ,cause M1 is not that powerful.
BSIZE = 12
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

class SEBlock(nn.Module):
    def __init__(self,ratio,input_shape) -> None:
        super(SEBlock,self).__init__()  

        
        self.ratio = ratio
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.l1 =nn.Linear(input_shape,input_shape//ratio,bias=False)
        self.l2 =nn.Linear(input_shape//ratio,input_shape,bias=False)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)



    
    def forward(self,inputx):

        b,c,_,_ = inputx.shape
        x= self.pool(inputx)
        x = x.view(b,c)

        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.sig(x)

        x = x.view(b,c,1,1)
        x = inputx*x

        return x



class MBConv1(nn.Module):
    def __init__(self,ksize) -> None:
        super(MBConv1,self).__init__()  

        self.swish = nn.SiLU()
        self.sig =nn.Sigmoid()
        self.dwc_1 = nn.Conv2d(32,16,kernel_size=ksize,stride=2,padding=1,groups=4)
        self.bn = nn.BatchNorm2d(16)
        self.se = SEBlock(ratio=4,input_shape=16)
        self.cn = nn.Conv2d(16,24,ksize,1,1)
        self.bnd = nn.BatchNorm2d(24)

    
    def forward(self,x):

        x = x.reshape(BSIZE,32,224,224)
        x = self.swish(self.bn(self.dwc_1(x)))
        x= self.se(x)
        x = self.cn(x)
        x = self.bnd(x)
        
        return x



class MBConv6(nn.Module):
    def __init__(self,input_shape,output_shape,ksize,st,pad) -> None:
        super(MBConv6,self).__init__()  

        self.c1 = nn.Conv2d(input_shape,output_shape,kernel_size=ksize,stride=1,padding=pad)
        self.bn = nn.BatchNorm2d(output_shape)

        self.swish = nn.SiLU()

        self.dwc_1 = nn.Conv2d(output_shape,output_shape,kernel_size=(ksize,ksize),stride=st,padding=pad,groups=4)
        self.bnd = nn.BatchNorm2d(output_shape)

        self.se = SEBlock(ratio=4,input_shape=output_shape)


        self.cn = nn.Conv2d(output_shape,output_shape,kernel_size=ksize,stride=1,padding=pad)
        self.bnd2 = nn.BatchNorm2d(output_shape)


    def forward(self,x):
   
        x= self.swish(self.bn(self.c1(x)))
        x = self.swish(self.bnd(self.dwc_1(x)))
        x = self.se(x)
        x= self.cn(x)
        x = self.bnd2(x)
       
        return x



class EfficientNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #will return 32,224,224 ? 
        self.c1 = nn.Conv2d(3,32,3,1,1)
        self.rel = nn.ReLU() 
        self.mb1 = MBConv1(3)

        #112x112
        self.mb6_1 = nn.ModuleList([MBConv6(24,24,3,1,1) for i in range(2)])

        #56x56
        self.mb6_2 = nn.ModuleList([MBConv6(24,40,5,1,2),MBConv6(40,40,5,2,1)])

        #28x28
        self.mb6_3 = nn.ModuleList([MBConv6(40,80,3,1,1),MBConv6(80,80,3,2,1),MBConv6(80,80,3,1,1)])
        
        #14x14
        self.mb6_4 = nn.ModuleList([MBConv6(80,112,5,1,1),MBConv6(112,112,5,1,2),MBConv6(112,112,5,1,2)])

        #14x14
        self.mb6_5 = nn.ModuleList([MBConv6(112,192,5,1,5),MBConv6(192,192,5,1,5),MBConv6(192,192,5,1,3),MBConv6(192,192,5,1,3)])

        #7x7
        self.mb6_6 = nn.ModuleList([MBConv6(192,320,3,2,1)])

        self.cfinal =nn.Conv2d(320,320,1,1,1)
        self.pool =nn.AdaptiveAvgPool2d(1)
        self.fc= nn.Linear(320*BSIZE,128*BSIZE)
        self.fc2= nn.Linear(128*BSIZE,64*BSIZE)
        self.fc3= nn.Linear(64*BSIZE,25*BSIZE)




        self.act_out = nn.Softmax(dim=0)
        self.swish= nn.SiLU()

    def forward(self,x):
        x = self.c1(x)
        x= self.swish(x)
        x = self.mb1(x) 
        #exit(1)
        for mb6 in self.mb6_1:
            x = mb6(x)
      
        for mb6 in self.mb6_2:
            x = mb6(x)
       
        for mb6 in self.mb6_3:
            x = mb6(x)

        for mb6 in self.mb6_4:
            x = mb6(x)
        for mb6 in self.mb6_5:
            x = mb6(x)

        for mb6 in self.mb6_6:
            x = mb6(x)


        x =self.cfinal(x)
        x = self.pool(x).view(-1)
        x = self.fc(x)
        x = self.fc2(x)
        x = self.fc3(x).reshape(BSIZE,25)



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
            x = x.reshape(BSIZE,3,IMG_SIZE,IMG_SIZE)
           

            out = net(x).reshape(BSIZE,25)
         #   print(out)
            kdo = torch.argmax(out.clone()[0].cpu().detach(),axis=-1)
            y=y.view(-1).long()
 #           print(y)
#            print(y.shape)

            #print(out.shape)
        #    exit(1)
            optimizer.zero_grad()
            print(y[0],kdo)
            ls = loss(out,y)
 
            ls.backward()
            optimizer.step()

            floss+=ls.item()
            nrm+=1
            print(f"** Epoch: {e} |  Loss: {floss/nrm} ")


env = os.getenv("RUN_MODE")
assert(env != None)
env = int(env)
if(env == 1):
    train_EfficientNet()


