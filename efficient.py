import os,io,sys
import torch.nn as nn 
import torch
import numpy as np
import cv2
from math import * 
from data_api import * 
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


class StohasticDepth(nn.Module):
    def __init__(self,prob=0.8) -> None:
        super().__init__()
        self.p = prob

    def forward(self,x):
        self.rand = torch.randn(x.shape[0],1,1,1) < self.p

        x = torch.div(x,self.p) * self.rand
        return x

# Biggest help i ever got in my entire life for the actual MBConv block : 
# https://paperswithcode.com/method/inverted-residual-block 

class ConvBlock(nn.Module):

    def __init__(self,inputd,outputd,kern_size=3,stride=1,padding=0,groups=1,bn=True,act=True,bias=False) -> None:
        super(ConvBlock,self).__init__()  
        self.cd = inputd
        self.outd = outputd
        self.c1 = nn.Conv2d(inputd,outputd,kernel_size=kern_size,stride=stride,padding=padding,groups=groups,bias=bias)
        self.bn = nn.Identity() if bn == False else nn.BatchNorm2d(outputd)
        self.act =nn.Identity() if act == False else nn.SiLU()


    
    def forward(self,x):
 
        x = self.c1(x)
        x = self.bn(x)
        x = self.act(x)
     
        return x
    

class MBConvN(nn.Module):
    def __init__(self,nin,nout,ksize=3,stride=1,expansion=6,R=4,survival_prob=0.8  ) -> None:
        super(MBConvN,self).__init__()  

        padding = (ksize-1)//2
        inter_channels=int(nin*expansion)

        self.nin = nin 
        self.nout = nout
        self.expand = nn.Identity() if(expansion == 6) else ConvBlock(nin,inter_channels,kern_size=1)
        if(expansion == 6):
           
            self.depthwise= ConvBlock(nin,inter_channels,kern_size=ksize,stride=stride,padding=padding,groups= nin)
        else:

            self.depthwise= ConvBlock(inter_channels,inter_channels,kern_size=ksize,stride=stride,padding=padding,groups= inter_channels)
        
        self.se= SEBlock(R,input_shape=inter_channels)
        self.skip_connection = (stride ==1 and nin == nout)
        self.drop_layer = StohasticDepth(prob=survival_prob)
        self.pointwise = ConvBlock(inter_channels,nout,kern_size=1,act=False)

    def forward(self,x):

        emd = x 
        
        x = self.expand(x)
        
        x = self.depthwise(x)
        
        x = self.se(x)
        
        x = self.pointwise(x)
       # print(self,"NIN:",self.nin,"NOUT:",self.nout)
        if(self.skip_connection):

            x = self.drop_layer(x)
            x+=emd

        return x




class SEBlock(nn.Module):
    def __init__(self,ratio,input_shape) -> None:
        super(SEBlock,self).__init__()  

        
        self.ratio = ratio
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(input_shape, input_shape//ratio, kernel_size=1),
                    nn.SiLU(),
                    nn.Conv2d(input_shape//ratio, input_shape, kernel_size=1),
                    nn.Sigmoid()
                )


    
    def forward(self,x):
        y=self.se(x)       
        return x*y


b0   =  (1.0, 1.0, 224, 0.2)

class EfficientNet(nn.Module):
    
    def __init__(self,width_m,height_m,dropout,num_of_images) -> None:
        super().__init__()
        width_m= 1.0 
        height_m= 1.0 
        dropout = 0.2
        
        last_channel = ceil(1280*width_m)
        self.lc = last_channel
        self.features = self.create_features(width_m,height_m,last_channel)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
    
        self.classify = nn.Sequential(
            nn.Dropout(dropout),        
            nn.Linear(last_channel,NUM_OF_CLASSES)
        )
    
        self.noi = num_of_images    
     


    def forward(self,x):

        x =self.features(x)

        x = self.pool(x)
       
        x =x.reshape(BSIZE,self.lc)
        x = self.classify(x)

        return x

    def create_features(self,w,h,lc):
        channels = 4*ceil(int(32*w) / 4)
        
        #defined in paper, im too lazy to do this myself so i copied it from some random implementation.
        layers =[ConvBlock(3,channels,kern_size=3,stride=2,padding=1)] 

        in_channels = channels
        
        kernels = [3, 3, 5, 3, 5, 5, 3]
        expansions = [1, 6, 6, 6, 6, 6, 6]
        num_channels = [16, 24, 40, 80, 112, 192, 320]
        num_layers = [1, 2, 2, 3, 3, 4, 1]
        strides =[1, 2, 2, 2, 1, 2, 1]

        scaled_num_channels = [4*ceil(int(c*w) / 4) for c in num_channels]
        scaled_num_layers = [int(d * h) for d in num_layers]
    
        for i in range(len(scaled_num_channels)):

         #   print(in_channels,scaled_num_channels[i])
            if(scaled_num_layers[i] <=1):

                lay= MBConvN(in_channels,scaled_num_channels[i],kernels[i],strides[i],expansions[i],4,0.8)
                
                in_channels = scaled_num_channels[i]
                layers.append(lay)
            elif scaled_num_layers[i] > 1:
                dmad =[] 
                
                lay= MBConvN(in_channels,scaled_num_channels[i],kernels[i],strides[i],expansions[i],4,0.8)
                
                dmad.append(lay)

                for _ in range(scaled_num_layers[i]):
                    l2= MBConvN(scaled_num_channels[i],scaled_num_channels[i],kernels[i],strides[i],expansions[i],4,0.8)
                    dmad.append(l2)
                
                for module in dmad:
                    layers.append(module)
                in_channels = scaled_num_channels[i]

     
        layers.append(ConvBlock(in_channels, lc, kern_size = 1, stride = 1, padding = 0))
    
        return nn.Sequential(*layers)


def train_EfficientNet():

    loss = nn.CrossEntropyLoss()

    data =Dataset()
    net = EfficientNet(b0[0],b0[1],b0[3],len(data.x))
   # img = load_and_scale_image("./crap.png").reshape(3,IMG_SIZE,IMG_SIZE)
    loader = torch.utils.data.DataLoader(data,batch_size=BSIZE,shuffle=True)
    
    optimizer = torch.optim.Adam(net.parameters(),lr=LR)

    for e in range(EPOCHS):
        floss = 0 
        nrm = 0 

        for b,(x,y) in enumerate(loader):

            x = x.reshape(BSIZE,3,IMG_SIZE,IMG_SIZE)           
            out = net(x).reshape(BSIZE,25)

            kdo = torch.argmax(out.clone()[0].cpu().detach(),axis=-1)
            y=y.view(-1).long()
 
            optimizer.zero_grad()

            ls = loss(out,y)
 
            ls.backward()
            optimizer.step()

            floss+=ls.item()
            nrm+=1
            print(f"** Epoch: {e} |  Loss: {floss/nrm} ")
    
    torch.save(net.state_dict(),"./test.pth")

def test_efficientNet():
    
    dt = time.time()
    data =Dataset()
    od = time.time()
    print("loading time (dataset):",(od-dt), "secs")
    net = EfficientNet(b0[0],b0[1],b0[3],len(data.x))
    net.load_state_dict(torch.load("./model.pth"))
  #  img = load_and_scale_image("./crap.png").reshape(3,IMG_SIZE,IMG_SIZE)
    loader = torch.utils.data.DataLoader(data,batch_size=BSIZE,shuffle=True)

    for b,(x,y) in enumerate(loader):
        
        x =x.reshape(BSIZE,3,IMG_SIZE,IMG_SIZE)
        
        out = net(x)
        out = out.clone().cpu().detach()[0]
        out = torch.argmax(out,axis=-1).numpy().tolist()  
        xd = x[0].numpy().reshape(IMG_SIZE,IMG_SIZE,3)*255

        dt = time.time()
        closest_neighbour = data.get_embedding_item(x[2],out)*255
        od = time.time()
        print("embedding vector time (ms):",(od-dt)*1000)


env = os.getenv("RUN_MODE")
if(env != None):
    env = int(env)
    if(env == 1):
        train_EfficientNet()
    else:
        test_efficientNet()


