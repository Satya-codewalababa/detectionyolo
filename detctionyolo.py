#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import tqdm as notebook_tqdm
import numpy as np
import pandas as pd
import cv2


# In[2]:


# Load the model of yolov5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


# In[3]:


model


# In[4]:


# Images
imgs = ['F:/helmet/archive/dataset/obj/10.jpg'] # batch of images


# In[5]:


results = model(imgs)
results.print()


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(np.squeeze(results.render()))


# In[8]:


results.show()


# In[7]:


np.array(results.render()).shape


# In[10]:


np.squeeze(results.render())


# In[11]:


results.xyxy


# In[12]:


results.show()


# In[8]:


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Make detections 
    results = model(frame)
    
    cv2.imshow('YOLO', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:


# In[44]:


get_ipython().system('cd yolov5-master && python train.py --img 416 --batch 16 --epochs 150 --data dataset.yaml --weights yolov5s.pt --cache')


# In[ ]:


get_ipython().system('cd yolov5-master && python train.py --img 320 --batch 100 --epochs 3000 --data dataset.yaml --weights yolov5s.pt --workers 2')


# In[9]:


model = torch.hub.load('ultralytics/yolov5','custom',path='F:/yolov5-master/yolov5-master/runs/train/exp19/weights/last.pt', force_reload=True)


# In[45]:


import os


# In[48]:


read_img = os.path.join('F:/yolov5-master/data','images','defog.jpg')


# In[57]:


read_img1 = os.path.join('F:/yolov5-master','Test_images','a3c2b8aa5b43ec61925d90f76502a380.jpg')


# In[58]:


result_final = model(read_img1)
result_final.print()


# In[59]:


plt.imshow(np.squeeze(result_final.render()))
plt.show()


# In[60]:


result_final.show()


# In[19]:


cap = cv2.VideoCapture('F:/yolov5-master/videoplayback.mp4')
while cap.isOpened():
    ret, frame = cap.read()
   
    
    # Make detections q
    results = model(frame)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:




