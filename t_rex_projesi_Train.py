import glob
import os
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

imgs = glob.glob("./img/*.png")

width = 125
height = 50

X = []
Y = []

for img in imgs:
    
    filename = os.path.basename(img)
    label = filename.split("_")[0]   #up/down/right elde ediyoruz
    im = np.array(Image.open(img).convert("L").resize((width, height)))   #resmi yeniden boyutlandirma
    im = im / 255   #normalize etme
    X.append(im)     #X=features(images)
    Y.append(label)  #Y = labeller
    
X = np.array(X)     #X'i arraya ceviriyoruz
X = X.reshape(X.shape[0], width, height, 1)

#sns.countplot(Y)

def onehot_labels(values):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse = False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

Y = onehot_labels(Y)
train_X, test_X, train_y, test_y = train_test_split(X, Y , test_size = 0.25, random_state = 2)    

#This code splits your dataset (X, y) into a training set (75%) and 
#a test set (25%). The train_test_split function is a quick and efficient 
#way to prepare your data for machine learning models.

#The random_state parameter is used for initializing the internal random 
#number generator, which will decide the splitting of data into train and 
#test indices. This is to ensure reproducibility. If you don’t specify or 
#pass an integer to the random_state parameter, you might get a different 
#output each time you split the data because of the shuffling.it should 
#always be the same if you want the same results everytime you run your code

# cnn model
model = Sequential()   
model.add(Conv2D(32, kernel_size = (3,3), activation = "relu", input_shape = (width, height, 1)))
model.add(Conv2D(64, kernel_size = (3,3), activation = "relu"))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Flatten()) #siniflandirma kismina geciyoruz
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.4))
model.add(Dense(3, activation = "softmax"))

#modelimizi olusturduk, simdi onu compile edecegiz

# if os.path.exists("./trex_weight.h5"):
#     model.load_weights("trex_weight.h5")
#     print("Weights yuklendi")    

model.compile(loss = "categorical_crossentropy", optimizer = "Adam", metrics = ["accuracy"])

# training islemine geciyoruz

model.fit(train_X, train_y, epochs = 35, batch_size = 215)

#35 kere egitim gerceklessin

score_train = model.evaluate(train_X, train_y)
print("Eğitim doğruluğu: %",score_train[1]*100)    
    
score_test = model.evaluate(test_X, test_y)
print("Test doğruluğu: %",score_test[1]*100)      
    
 
open("model_new.json","w").write(model.to_json())
model.save_weights("trex_weight_new.h5")   