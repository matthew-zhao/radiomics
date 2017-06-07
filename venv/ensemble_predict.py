from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
import pickle
import numpy as np
import boto

from boto.s3.key import Key

def predict(event, context):
    conn = boto.connect_s3("AKIAIMQLHJNMP6DOUM4A","8dJAfPZlTjMR1SOcOetImclAmT+G02VkQiuHefdY")
    classifier = event['classifier']
    test_bucket = conn.get_bucket(event['bucket_from'])
    model_bucket = conn.get_bucket(event['model_bucket'])
    result_bucket = conn.get_bucket(event['result_bucket'])

    num_items = event['num_items']
    
    train_key = test_bucket.get_key('ready_matrix.npy')
    train_key.get_contents_to_filename('/tmp/ready_matrix.npy')

    model_list = model_bucket.list()

    i = 0
    clf_list = []
    for model in model_list:
        model.get_contents_to_filename("/tmp/model" + str(i))

        with open("/tmp/model" + str(i), "rb") as keyfile:
            contents = keyfile.read()
            clf_list.append((str(i), pickle.loads(contents)))

        i += 1

    ensembler = VotingClassifier(estimators=clf_list, voting='hard')

    with open("/tmp/ready_matrix.npy", "rb") as ready_matrix:
        X = np.load(ready_matrix)

    X_converted = X.astype(np.float)

    #predictions = clf.predict(X_converted)
    predictions = ensembler.predict(X_converted)
    pred = np.array(predictions)
    pred.reshape((77602*num_items, 5))

    predictionslist = np.argmax(pred, axis=1)

    new_predict_list = []
    for i in range(num_items):
        prediction = np.argmax(np.bincount(predictionslist[77602*i:77602*(i+1)]))
        new_predict_list.append(prediction)

    result_k = result_bucket.new_key(event['result_name'])
    with open("/tmp/results", "wb") as results:
        new_predict = ''.join(str(e) for e in new_predict_list)
        results.write(new_predict)

    result_k.set_contents_from_filename("/tmp/results")

    result_k.make_public()

    return 1 