from search_images import * 
import os,io,sys
import numpy as np 
import cv2 


image= cv2.imread("../train.X1/n01440764/n01440764_36.JPEG")

q = Queuer("../model.pth",True)
out = q.query_image(image)
print(type(out))
cv2.imwrite("./kt.png",out)



