from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
import pickle
import numpy as np
import boto
from scipy import stats

from boto.s3.key import Key

def predict(event, context):
    conn = boto.connect_s3("AKIAIMQLHJNMP6DOUM4A","8dJAfPZlTjMR1SOcOetImclAmT+G02VkQiuHefdY")
    classifier = event['classifier']
    test_bucket = conn.get_bucket(event['bucket_from'])
    model_bucket = conn.get_bucket(event['model_bucket'])
    result_bucket = conn.get_bucket(event['result_bucket'])
    image_name = event["image_name"]
    feature = event["feature"]
    
    print(image_name + "_" + feature)
    train_key = test_bucket.get_key(image_name + "_" + feature + "-processed.npy")
    train_key.get_contents_to_filename('/tmp/ready_matrix.npy')

    model_list = model_bucket.list()

    with open("/tmp/ready_matrix.npy", "rb") as ready_matrix:
        X = np.load(ready_matrix)

    X_converted = X.astype(np.float)

    print("preparations done")

    i = 0
    clf_list = []
    total = []
    for model in model_list:
        model.get_contents_to_filename("/tmp/model" + str(i))

        with open("/tmp/model" + str(i), "rb") as keyfile:
            contents = keyfile.read()
        
        clf = pickle.loads(contents)
        predictions = clf.predict(X_converted)
        predictionslist = np.argmax(predictions, axis=1)
        total.append(predictionslist)
        i += 1

    total_array = np.array(total)
    labels_region = stats.mode(total_array, axis=0).mode

    #predictions = clf.predict(X_converted)
    #predictions = ensembler.predict(X_converted)
    #pred = np.array(predictions)
    #pred.reshape((77602*num_items, 5))

    #predictionslist = np.argmax(pred, axis=1)

    #new_predict_list = []
    prediction = stats.mode(labels_region, axis=1).mode[0][0]
    #new_predict_list.append(prediction)

    result_k = result_bucket.new_key(event['result_name'])
    with open("/tmp/results", "wb") as results:
        new_predict = str(prediction)
        results.write(new_predict)

    result_k.set_contents_from_filename("/tmp/results")

    result_k.make_public()

    return 1 