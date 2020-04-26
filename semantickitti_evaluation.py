#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.metrics import jaccard_score

evaluation_folder = '/home/wade/catkin_ws/src/BKISemanticMapping/data/dataset/sequences/04/evaluations/'

gt_all = np.array([])
pred_all = np.array([])
for i in range(60):
    print(i)
    
    result = np.loadtxt(evaluation_folder + str(i).zfill(6) + '.txt', dtype=np.uint32)
    gt = result[:,0]
    gt = gt & 0xFFFF
    pred = result[:,1]
    gt_all = np.concatenate((gt_all, gt))
    pred_all = np.concatenate((pred_all, pred))
    
# Ignore background and sky label
pred_all = pred_all[gt_all != 0]
gt_all = gt_all[gt_all != 0]
    
print(np.unique(np.concatenate((gt_all, pred_all), axis=0)) )
print(jaccard_score(gt_all, pred_all, average=None))

