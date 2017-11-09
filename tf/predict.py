#from sklearn.neural_network import MLPClassifier
import tensorflow as tf
import pickle
import numpy as np
import boto
#from scipy import stats

from boto.s3.key import Key

def predict(event, context):
    conn = boto.connect_s3("AKIAIMQLHJNMP6DOUM4A","8dJAfPZlTjMR1SOcOetImclAmT+G02VkQiuHefdY")
    classifier = event['classifier']
    test_bucket = conn.get_bucket(event['bucket_from'])
    model_bucket = conn.get_bucket(event['model_bucket'])
    result_bucket = conn.get_bucket(event['result_bucket'])
    model_bucket_name = event['model_bucket_name']

    image_num = event["image_num"]
    feature = event["feature"]

    num_items = event['num_items']
    
    print(str(image_num) + "_" + feature)
    train_key = test_bucket.get_key(str(image_num) + "_" + feature + "-processed.npy")
    train_key.get_contents_to_filename('/tmp/ready_matrix.npy')

    if classifier == 'neural':
        key = model_bucket.get_key(model_bucket_name)
    elif classifier == 'knn':
        key = model_bucket.get_key('nn')
    elif classifier == 'decision_tree':
        key = model_bucket.get_key('dt')
    elif classifier == 'forest':
        key = model_bucket.get_key('forest')
    elif classifier == 'bagging':
        key = model_bucket.get_key('bagging')

    key_checkpoint = model_bucket.get_key(model_bucket_name + '-checkpoint')
    key_index = model_bucket.get_key(model_bucket_name + '-index')
    key_data = model_bucket.get_key(model_bucket_name + '-data')

    key.get_contents_to_filename('/tmp/model.meta')
    key_checkpoint.get_contents_to_filename('/tmp/checkpoint')
    key_index.get_contents_to_filename('/tmp/model.index')
    key_data.get_contents_to_filename('/tmp/model.data-00000-of-00001')

    print("preparation ready")

    with open("/tmp/ready_matrix.npy", "rb") as ready_matrix:
        X = np.load(ready_matrix)

    X_converted = X.astype(np.float)

    print("data loaded")

    # with open("/tmp/key", "rb") as keyfile:
    #     contents = keyfile.read()
    #     clf = pickle.loads(contents)

    with tf.Session() as sess:
        old_saver = tf.train.import_meta_graph('/tmp/model.meta')
        old_saver.restore(sess, '/tmp/model')

        graph = tf.get_default_graph()
        X_train = graph.get_tensor_by_name("X_train:0")

        prediction_func = tf.get_collection('predict')[0]

        predictions = sess.run(prediction_func, feed_dict={X_train: X_converted})

    print("finished predicting")
    #pred = np.array(predictions)

    #predictionslist = np.argmax(predictions, axis=1)

    #new_predict_list = []
    #for i in range(num_items):
    #    prediction = np.argmax(np.bincount(predictionslist[77602*i:77602*(i+1)]))
    #    new_predict_list.append(prediction)

    #prediction = stats.mode(predictionslist).mode[0]

    # prediction = stats.mode(predictions).mode[0]

    result_k = result_bucket.new_key(event['result_name'])
    with open("/tmp/results", "wb") as results:
        #new_predict = ''.join(str(e) for e in predictionslist)
        #results.write(new_predict)
        #results.write(prediction)
        #new_predict = str(prediction)
        #results.write(new_predict)
        results.write(predictions)

    result_k.set_contents_from_filename("/tmp/results")

    result_k.make_public()

    return 1