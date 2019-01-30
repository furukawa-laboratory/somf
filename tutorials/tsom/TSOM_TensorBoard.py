
# coding: utf-8

# In[ ]:


import os
import sys
import pprint

pprint.pprint(sys.path)


# In[ ]:


sys.path.append( '/Users/hatanohajime/Desktop/PycharmProjects/flib')


# In[ ]:


from libs.models.tsom_tf import TSOM2


# In[ ]:


from libs.datasets.artificial.animal import load_data
from libs.visualization.tsom.tsom2_viewer import TSOM2_Viewer as TSOM2_V
import numpy as np


# In[ ]:


N1, N2, observed_dim, latent_dim, epochs, resolution, SIGMA_MAX, SIGMA_MIN, TAU, init='random'


# In[ ]:


X,labels_animal,labels_feature=load_data(retlabel_animal=True,retlabel_feature=True)

tsom = TSOM2(N1=X.shape[0], N2=X.shape[1], observed_dim=1, latent_dim=2, epochs=50, resolution=10,SIGMA_MAX=2.0,SIGMA_MIN=0.2,TAU=50)
tsom.predict(X) 
comp=TSOM2_V(y=tsom.Y,winner1=tsom.k_star1,winner2=tsom.k_star2,
             label1=labels_animal,label2=labels_feature)
comp.draw_map()


# In[ ]:


X.shape[0]

