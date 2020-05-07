from keras import models
import build_model
from sklearn.preprocessing import LabelEncoder
import os
import cv2
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
img_path = "D:/facenet/image"
epochs = 100
all_imag = os.listdir(img_path)
step = 3

img_arr = []
label = []
for i in all_imag:

    print("*************************************",img_path +"/"+i)
    img_p = os.listdir(img_path +"/"+i)
    print(img_p)
    for p in img_p:
        path = img_path + "/" + i +"/"+ p
        print(path)
        img_arr.append(cv2.resize(cv2.imread(path,cv2.IMREAD_COLOR),(250,250),interpolation=cv2.INTER_CUBIC))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        label.append(i)
le = LabelEncoder()
lab = label
label = le.fit_transform(label)
y_test = label
x_test = np.array(img_arr)
print("*****************************",np.shape(x_test))
model = models.load_model("D:/facenet/image/facenet_with_triplet_semihard.h5",custom_objects={'triplet_loss_adapted_from_tf':build_model.triplet_loss_adapted_from_tf})


print("model has been loaded successfuly")
embedding_size = 128
input_image_shape = (250,250,3)
testing_embeddings = build_model.create_base_network(input_image_shape,
                                             embedding_size=embedding_size)

for layer_target,layer_source in zip(testing_embeddings.layers,model.layers[2].layers):
    weights = layer_source.get_weights()
    layer_target.set_weights(weights)
    del weights
x_embeddings = testing_embeddings.predict(np.reshape(x_test, (len(x_test), 250, 250, 3)))
print(x_embeddings.shape)
x_embeddings = np.array(x_embeddings)
pca = PCA(n_components= 2)
vec_pca = pca.fit_transform(x_embeddings)
vec_pca = pd.DataFrame(vec_pca)
print(vec_pca.shape)
print("****************",vec_pca)
#label = set(lab)
#label = list(label)
label = pd.DataFrame(lab)
data = pd.concat([vec_pca,label],axis =1 ,ignore_index=True)
ax = sns.scatterplot(x=0, y=1, hue=2,
                      data=data)
plt.show()

