import torch
import torch.nn as nn
import numpy as np

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet,self).__init__()
        self.line1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.line2 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax()
        
    def forward(self,x):
        out = self.line1(x)
        out = self.relu(out)
        out = self.line2(out)
        out = self.softmax(out)
        return out
    
    
# class KNN():
    
#     def __init__(self,k=1):
#         self.k = k
#         self.X = []
#         self.y = []
        
#     def fit(self,X=np.ndarray,y=np.ndarray):
#         self.X += X[:]
#         self.y += y[:]
     
#     def predict(self,X):
#         pred = np.zeros(X.shape[0])
        
#         def get_dist(val, x):
#             res = np.linalg.norm(val - x)
            
#             if res == 0:
#                 return 10000
#             else:
#                 return res
            
#         for indx, obj in enumerate(X):
            
            
#             dist_to_the_rest = np.apply_along_axis(lambda x: get_dist(obj, x) , 1, X)
            
# #             np.array([[np.linalg.norm(obj - other_obj), self.y[ind]]\
# #                                  for ind, other_obj \
# #                                  in enumerate(self.X) \
# #                                  if np.linalg.norm(obj - other_obj) != 0]) 
#             classes = np.zeros(self.k)
    
#             for i in range(self.k):
#                 elm_index = np.argmin(dist_to_the_rest)
                
#                 classes[i] += self.y[elm_index]
                
#                 elm = dist_to_the_rest[elm_index]
#                 np.delete(dist_to_the_rest, elm)
            
                
#             classes_uniq, count = np.unique(classes, return_counts=True)
            
#             pred[indx] += classes_uniq[np.argmax(count)]
            
#         return pred
    
    
