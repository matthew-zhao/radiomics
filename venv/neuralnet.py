from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np
import boto

from boto.s3.key import Key

# Uses MLP Neural Net classifier to train a model
def classify(event, context):
    conn = boto.connect_s3()
    b = conn.get_bucket(event['bucket_from'])
    labels = conn.get_bucket(event['bucket_from_labels'])

    X_key = b.get_key('ready_matrix.npy')
    Y_key = labels.get_key('ready_labels.npy')

    X_key.get_contents_to_filename('/tmp/ready_matrix.npy')
    Y_key.get_contents_to_filename('/tmp/ready_labels.npy')

    X = np.load('/tmp/ready_matrix.npy')
    y = np.load('/tmp/ready_labels.npy')

    X_converted = X.astype(np.float)
    y_converted = y.astype(np.float)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 2), random_state=1)

    clf.fit(X_converted, y_converted)

    s = pickle.dumps(clf)

    model_bucket = conn.get_bucket('models')
    model_k = model_bucket.new_key(event['model_name'])
    model_k.set_contents_from_filename(s)

    model_k.make_public()
    return 1

