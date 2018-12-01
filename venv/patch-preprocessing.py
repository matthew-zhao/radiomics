import scipy
import numpy as np
import boto3
import boto
from boto.s3.connection import S3Connection
from boto.s3.key import Key
import uuid
import csv
from StringIO import StringIO
from boto3 import client as boto3_client
import json

def preprocess(event, context):
    #connecting to appropriate S3 bucket
    conn = boto.connect_s3("AKIAJRKPLMXU3JRGWYCA","LFfFCpaEsdCCz4KiEBKoTFS5ehXIcPjsPq3yqxjj")
    image_path = event["image_path"]

    #check whether it is training or testing set
    is_train = event["is_train"]
    has_labels = event["has_labels"]
    image_name = event["image_name"]
    model_bucket_name = event["model_bucket_name"]
    image_num = event["image_num"]
    bucket_from = event["bucket_from"]
    bucket_from_labels = event["bucket_from_labels"]

    actual_name, extension = image_path.split(".")

    # get array of labels
    if has_labels:
        if event["label_style"] == "array":
            npy_filename = actual_name.split("/")[-1] + ".npy"
            print(npy_filename)
            labels_key = labels.get_key(npy_filename)
            labels_key.get_contents_to_filename("/tmp/labels-" + npy_filename)

            with open("/tmp/labels-" + npy_filename, "rb") as npy:
                label_arr = np.load(npy)
        elif event["label_style"] == "single":
            label = np.array([event["label"]])

    if is_train:
        b = conn.get_bucket('training-array')
        b2 = conn.get_bucket('training-labels')
        k2 = b2.new_key('matrix' + str(image_name) + '.npy')
    else:
        b = conn.get_bucket('testing-array')

    value_dict, labels = analyze((img, label_arr), event, has_labels)
    
    if is_train and has_labels:
        upload_path_labels = '/tmp/matrix' + str(image_name) + '-labels.npy'
        np.save(upload_path_labels, labels)
        k2.set_contents_from_filename(upload_path_labels)

    for feature in value_dict:
        k = b.new_key('matrix' + str(image_name) + '_' + feature + '.npy')

        #creating a temp image numpy file
        upload_path = '/tmp/matrix' + str(image_name) + '_' + feature + '.npy'
    
        #saving image to S3 bucket
        np.save(upload_path, value_dict[feature])
    
        k.set_contents_from_filename(upload_path)

        if is_train and has_labels:
            msg = {"is_train": is_train, "image_name": str(image_name), "feature": str(feature), "bucket_from": "training-array", "bucket_from_labels": "training-labels", "has_labels": "True", "model_bucket_name": model_bucket_name, "image_num": image_num,
                    "queue_name": model_bucket_name + '.fifo', "queue_name1": model_bucket_name + '1.fifo', "num_classes": event["num_classes"], "num_machines": event["num_machines"]}
        elif is_train and not has_labels:
            msg = {"is_train": is_train, "image_name": str(image_name), "feature": str(feature), "bucket_from": "training-array", "bucket_from_labels": "", "has_labels": "", "model_bucket_name": model_bucket_name, "image_num": image_num,
                    "queue_name": model_bucket_name + '.fifo', "queue_name1": model_bucket_name + '1.fifo', "num_classes": event["num_classes"], "num_machines": event["num_machines"]}
        else:
            msg = {"is_train": is_train, "image_name": str(image_name), "feature": str(feature), "bucket_from": "testing-array", "bucket_from_labels": "", "has_labels": "", "model_bucket_name": model_bucket_name, "image_num": image_num,
                    "queue_name": model_bucket_name + '.fifo', "queue_name1": model_bucket_name + '1.fifo', "num_classes": event["num_classes"], "num_machines": event["num_machines"], "xscale": 322, "yscale": 241}
        lambda_client = boto3_client('lambda')
        lambda_client.invoke(FunctionName="preprocess3", InvocationType='Event', Payload=json.dumps(msg))

    return 0

#calculating all the first-order statistics on all the images
def analyze(arr_arg, event, has_labels):
    result = {}
    arr = np.array(arr_arg[0])
    label_arr = arr_arg[1]
    h = scipy.histogram(arr, 256)
    dim = len(arr.shape)
    filter_size = int(event['filter_size'])

    total = []
    labels = []
    if dim == 3:
        for i in range(arr.shape[0] - filter_size + 1):
            for j in range(arr.shape[1] - filter_size + 1):
                for k in range(arr.shape[2] - filter_size + 1):
                    row = arr[i:i+filter_size, j:j+filter_size, k:k+filter_size].flatten()
                    
                    if has_labels:
                        label_row = label_arr[i:i+filter_size, j:j+filter_size, k:k+filter_size].flatten()
                        label = np.argmax(np.bincount(label_row))
                        labels.append(label)
                    total.append(row)
        result["total"] = np.array(total)
        
        x = arr.shape[0] - filter_size + 1
        y = arr.shape[1] - filter_size + 1
        z = arr.shape[2] - filter_size + 1
        total_size = x * y * z
    elif dim == 2:
        for i in range(arr.shape[0] - filter_size + 1):
            for j in range(arr.shape[1] - filter_size + 1):
                row = arr[i:i+filter_size, j:j+filter_size].flatten()
                if has_labels:
                    label_row = label_arr[i:i+filter_size, j:j+filter_size].flatten()
                    label = np.argmax(np.bincount(label_row))
                    labels.append(label)
                total.append(row)
        result["total"] = np.array(total)
        x = arr.shape[0] - filter_size + 1
        y = arr.shape[1] - filter_size + 1
        total_size = x * y
    if has_labels:
        final_label_arr = np.array(labels)
        return result, final_label_arr
    else:
        return result, None