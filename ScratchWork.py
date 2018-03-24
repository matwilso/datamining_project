
# coding: utf-8

# In[56]:


# taken from: http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import json
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
import skimage.io, skimage.transform


data_path = './extraction/data/'
tfidf_path = './tfidf.json'
meta_path = './meta_tfidf.json'
meta = json.load(open(meta_path, 'r'))
good_id_to_filename = meta['ids']
i_to_id = meta['itoid']
thumbnail_ext = ".jpg"

D = np.array(json.load(open(tfidf_path, 'r'))['D'])[:100,:100]
names = np.asarray([good_id_to_filename[i_to_id[str(i)]] for i in range(len(D))])
np.savetxt("D.csv", D, delimiter=",")
np.savetxt("labels.csv", names, fmt="%s", delimiter=",",encoding='utf-8')


# In[57]:


X = D[:500,:500]


# In[39]:


X.shape


# In[40]:




def get_thumbnails():
    for i in range(len(X)):
        filename = data_path+good_id_to_filename[i_to_id[str(i)]]+thumbnail_ext
        image = skimage.io.imread(filename)
        image = skimage.transform.resize(image, (32, 32))
        yield image


# In[49]:


plt.rcParams['figure.figsize']


# In[ ]:


plt.Annotation


# In[52]:


#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = 32 * ((X - x_min) / (x_max - x_min))

    plt.figure(figsize=[24, 18])
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.plot(X[i, 0], X[i, 1])
        plt.text(X[i, 0]-0.05, X[i, 1]+0.05, names[i])

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            #if np.min(dist) < 4e-3:
            #    # don't show points that are too close
            #    continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(next(thumbnail_iter), zoom=1),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


# In[42]:


#----------------------------------------------------------------------
# t-SNE embedding of the digits dataset
print("Computing t-SNE embedding")
tsne = manifold.TSNE(n_components=2, random_state=0, metric="precomputed", n_iter=16000)
t0 = time()
X_tsne = tsne.fit_transform(X)


# In[53]:



# In[55]:


thumbnail_iter = get_thumbnails()
plot_embedding(X_tsne,
               "t-SNE embedding of the digits (time %.2fs)" %
               (time() - t0))
plt.show()

