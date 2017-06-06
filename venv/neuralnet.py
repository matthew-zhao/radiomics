from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np
import boto

from boto.s3.key import Key

# Uses MLP Neural Net classifier to train a model
def classify(event, context):
    print("Hello")
    conn = boto.connect_s3("AKIAIMQLHJNMP6DOUM4A","8dJAfPZlTjMR1SOcOetImclAmT+G02VkQiuHefdY")
    b = conn.get_bucket(event['bucket_from'])
    labels = conn.get_bucket(event['bucket_from_labels'], validate=False)

    bucket_list = b.list()

    for l in bucket_list:
        if l.key == "ready_matrix.npy":
            l.get_contents_to_filename('ready_matrix.npy')
            Y_key = labels.get_key('ready_labels.npy')
            Y_key.get_contents_to_filename('ready_labels.npy')

    with open("ready_matrix.npy", "rb") as ready_matrix:
        X = np.load(ready_matrix)

    with open("ready_labels.npy", "rb") as ready_labels:
        y = np.load(ready_labels)

    print("About to train")

    clf = MLPClassifier(solver='adam', alpha=1e-3, hidden_layer_sizes=(82,), random_state=1, max_iter=20)

    clf.fit(X, y)

    s = pickle.dumps(clf)

    model_bucket = conn.get_bucket('models-train')

    model_bucket_name = event["model_bucket_name"]
    image_num = event["image_num"]
    model_k = model_bucket.new_key(model_bucket_name + image_num)

    with open("model", "wb") as model:
        model.write(s)
    
    model_k.set_contents_from_filename("model")

    model_k.make_public()
    return 1




