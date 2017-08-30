from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.utils import np_utils
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import pickle
import numpy as np
import boto
import argparse

from boto.s3.key import Key
import matplotlib.pyplot as plt

conn = boto.connect_s3("AKIAIMQLHJNMP6DOUM4A","8dJAfPZlTjMR1SOcOetImclAmT+G02VkQiuHefdY")
b = conn.get_bucket('training-arrayfinal')
labels = conn.get_bucket('training-labelsfinal', validate=False)

X_key = b.get_key('ready_matrix.npy')
X_key.get_contents_to_filename('train_matrix.npy')
Y_key = labels.get_key('ready_labels.npy')
Y_key.get_contents_to_filename('train_labels.npy')

with open("train_matrix.npy", "rb") as ready_matrix:
    X = np.load(ready_matrix)

with open("train_labels.npy", "rb") as ready_labels:
    y = np.load(ready_labels)

model=Sequential()
model.add(Dense(243, input_dim=162))
model.add(Activation('relu'))
model.add(Dense(81, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
 
history = model.fit(X, y, validation_split=0.05)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# predictions = model.predict(X)

# predictionslist = np.argmax(predictions, axis=1)

# new_predict_list = []
# for i in range(12):
#     prediction = np.argmax(np.bincount(predictionslist[77602*i:77602*(i+1)]))
#     new_predict_list.append(prediction)

# Uses MLP Neural Net classifier to train a model
# def classify(event):
#     print("Hello")
#     conn = boto.connect_s3("AKIAIMQLHJNMP6DOUM4A","8dJAfPZlTjMR1SOcOetImclAmT+G02VkQiuHefdY")
#     b = conn.get_bucket("training-arrayfinal")
#     labels = conn.get_bucket("training-labelsfinal", validate=False)

#     #X_key = b.get_key('ready_matrix.npy')
#     #Y_key = labels.get_key('ready_labels.npy')
#     bucket_list = b.list()

#     for l in bucket_list:
#         if l.key == "ready_matrix.npy":
#             l.get_contents_to_filename('ready_matrix.npy')
#             Y_key = labels.get_key('ready_labels.npy')
#             Y_key.get_contents_to_filename('ready_labels.npy')

#     print("finished reading from bucket")

#     #X_key.get_contents_to_filename('/tmp/ready_matrix.npy')
#     #Y_key.get_contents_to_filename('/tmp/ready_labels.npy')

#     with open("ready_matrix.npy", "rb") as ready_matrix:
#         X = np.load(ready_matrix)

#     with open("ready_labels.npy", "rb") as ready_labels:
#         y = np.load(ready_labels)

#     print("About to train")

#     clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(52,32), random_state=1, max_iter=20)

#     clf.fit(X, y)

#     print("done training")

#     s = pickle.dumps(clf)

#     model_bucket = conn.get_bucket('models-train')

#     #model_k = model_bucket.new_key(event['model_name'])

#     model_k = model_bucket.new_key('nm')

#     with open("model", "wb") as model:
#         model.write(s)
    
#     model_k.set_contents_from_filename("model")

#     model_k.make_public()
#     return 1


# if __name__ == '__main__':
    

#     parser = argparse.ArgumentParser(description='Description of your program')

#     parser.add_argument('-f','--bucket_from', help='Description for foo argument', required=True)
#     parser.add_argument('-b','--bucket_from_labels', help='Description for bar argument', required=True)
#     args = vars(parser.parse_args())

#     classify(args)


#     print("done training")

