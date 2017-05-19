from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np
import boto

from boto.s3.key import Key

def predict(event, context):
    conn = boto.connect_s3("AKIAIMQLHJNMP6DOUM4A","8dJAfPZlTjMR1SOcOetImclAmT+G02VkQiuHefdY")
    classifier = event['classifier']
    test_bucket = conn.get_bucket(event['bucket_from'])
    model_bucket = conn.get_bucket(event['model_bucket'])
    
    train_key = test_bucket.get_key('ready_matrix.npy')
    train_key.get_contents_to_filename('/tmp/ready_matrix.npy')

    if classifier == 'neural':
        key = model_bucket.get_key('nm')
    elif classifier == 'knn':
        key = model_bucket.get_key('nn')
    elif classifier == 'decision_tree':
        key = model_bucket.get_key('dt')
    elif classifier == 'forest':
        key = model_bucket.get_key('forest')
    elif classifier == 'bagging':
        key = model_bucket.get_key('bagging')

    key.get_contents_to_filename('/tmp/key')

    with open("/tmp/key", "rb") as keyfile:
        contents = keyfile.read()
        clf = pickle.loads(contents)

    with open("/tmp/ready_matrix.npy", "rb") as ready_matrix:
        X = np.load(ready_matrix)

    X_converted = X.astype(np.float)

    predictions = clf.predict(X_converted)

    return {"predictions": predictions.tolist()} 