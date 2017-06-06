from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np
import boto
import argparse

from boto.s3.key import Key

# Uses MLP Neural Net classifier to train a model
def classify(event):
    print("Hello")
    conn = boto.connect_s3("AKIAIMQLHJNMP6DOUM4A","8dJAfPZlTjMR1SOcOetImclAmT+G02VkQiuHefdY")
    b = conn.get_bucket(event['bucket_from'])
    labels = conn.get_bucket(event['bucket_from_labels'], validate=False)
    model_bucket = conn.get_bucket('models-train')

    #X_key = b.get_key('ready_matrix.npy')
    #Y_key = labels.get_key('ready_labels.npy')
    bucket_list = b.list()

    for l in bucket_list:
        if l.key == "ready_matrix.npy":
            l.get_contents_to_filename('ready_matrix.npy')
            Y_key = labels.get_key('ready_labels.npy')
            Y_key.get_contents_to_filename('ready_labels.npy')

    existing_model = model_bucket.get_key('nm')

    print("finished reading from bucket")

    #X_key.get_contents_to_filename('/tmp/ready_matrix.npy')
    #Y_key.get_contents_to_filename('/tmp/ready_labels.npy')

    with open("ready_matrix.npy", "rb") as ready_matrix:
        X = np.load(ready_matrix)

    with open("ready_labels.npy", "rb") as ready_labels:
        y = np.load(ready_labels)

    print("About to train")

    if existing_model:
        existing_model.get_contents_to_filename('key')
        with open("key", "rb") as keyfile:
            contents = keyfile.read()
            clf = pickle.loads(contents)
    else:
        clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(82,), random_state=1, warm_start=True, max_iter=20)

    # TODO: this may not be true if things are not one-hot encoded
    clf.classes_ = [0, 1]

    clf.partial_fit(X, y)

    print("done training")

    s = pickle.dumps(clf)

    #model_k = model_bucket.new_key(event['model_name'])

    model_k = model_bucket.new_key('nm')

    with open("model", "wb") as model:
        model.write(s)
    
    model_k.set_contents_from_filename("model")

    model_k.make_public()
    return 1


if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description='Description of your program')

    parser.add_argument('-f','--bucket_from', help='Description for foo argument', required=True)
    parser.add_argument('-b','--bucket_from_labels', help='Description for bar argument', required=True)
    args = vars(parser.parse_args())

    classify(args)


    print("done training")

