# Lambda 2
import scipy
import numpy as np
import boto3
import boto
from boto.s3.connection import S3Connection
from boto.s3.key import Key
import uuid
from scipy import stats
from scipy import ndimage
import csv
from PIL import Image
from StringIO import StringIO
from boto3 import client as boto3_client
import json
import dropbox

def preprocess(event, context):
    #connecting to appropriate S3 bucket
    conn = boto.connect_s3("AKIAIMQLHJNMP6DOUM4A","8dJAfPZlTjMR1SOcOetImclAmT+G02VkQiuHefdY")
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

    if event["is_dropbox"]:
        #access the dropbox folder using auth token and image path
        client = dropbox.Dropbox(event["auth_token"])
        f, metadata = client.files_download(image_path)
        data = metadata.content

        if extension == "dcm":
            f2 = open("/tmp/response_content.dcm", "wb")
            f2.write(data)
            f2.close()
        else:
            img_raw = Image.open(StringIO(data))
    else:
        b = conn.get_bucket("train-data")
        img_key = b.get_key(image_path)

        if extension == "dcm":
            img_key.get_contents_to_filename('/tmp/response_content.dcm')
        else:
            img_key.get_contents_to_filename('/tmp/' + image_path)
            img_raw = Image.open('/tmp/' + image_path)


    if extension == "dcm":
        f2 = open("/tmp/response_content.dcm", "rb")
        ds = dicom.read_file(f2)
        img_raw = ds.pixel_array
        f2.close()
        xscale = 324.0 / img_raw.shape[0]
        yscale = 243.0 / img_raw.shape[1]
        img = scipy.ndimage.interpolation.zoom(img_raw, [xscale, yscale])
    else:
        img_resized = img_raw.resize((324, 243), Image.ANTIALIAS)
        img = scipy.array(img_resized)

    labels_bucket = conn.get_bucket(bucket_from_labels)
    npy_filename = actual_name + ".npy"
    labels_key = labels_bucket.get_key(npy_filename)

    if is_train:
        b = conn.get_bucket('training-array')
        b2 = conn.get_bucket('training-labels')
        k2 = b2.new_key('matrix' + str(image_name) + '.npy')
    else:
        b = conn.get_bucket('testing-array')

    k = b.new_key('matrix' + str(image_name) + '.npy')

    labels_key.get_contents_to_filename("/tmp/labels-" + npy_filename)
    with open("/tmp/labels-" + npy_filename, "rb") as npy:
        label_arr = np.load(npy)

    # last_bool = event['last']
    value_matrix, labels = analyze((img, label_arr), event, has_labels)

    #creating a temp image numpy file
    upload_path = '/tmp/matrix' + str(image_name) + '.npy'

    #saving image to S3 bucket
    np.save(upload_path, value_matrix)

    k.set_contents_from_filename(upload_path)

    if is_train and has_labels:
        upload_path_labels = '/tmp/matrix' + str(image_name) + '-labels.npy'
        np.save(upload_path_labels, labels)
        k2.set_contents_from_filename(upload_path_labels)

    if is_train and has_labels:
        msg = {"is_train": is_train, "image_name": image_name, "bucket_from": "training-array", "bucket_from_labels": "training-labels", "has_labels": "True", "model_bucket_name": model_bucket_name, "image_num": image_num}
    elif is_train and not has_labels:
        msg = {"is_train": is_train, "image_name": image_name, "bucket_from": "training-array", "bucket_from_labels": "", "has_labels": "", "model_bucket_name": model_bucket_name, "image_num": image_num}
    else:
        msg = {"is_train": is_train, "image_name": image_name, "bucket_from": "testing-array", "bucket_from_labels": "", "has_labels": "", "model_bucket_name": model_bucket_name, "image_num": image_num}
    lambda_client = boto3_client('lambda')
    lambda_client.invoke(FunctionName="preprocess3", InvocationType='Event', Payload=json.dumps(msg))

    return 0

#calculating all the first-order statistics on all the images
def analyze(arr_arg, event, has_labels):
    result = None
    arr = np.array(arr_arg[0])
    label_arr = arr_arg[1]
    h = scipy.histogram(arr, 256)
    dim = len(arr.shape)
    filter_size = int(event['filter_size'])

    mean = scipy.ndimage.generic_filter(arr, scipy.mean, size = filter_size, mode = 'constant')
    
    maximum = scipy.ndimage.generic_filter(arr, np.amax, size = filter_size, mode = 'constant')
    
    minimum = scipy.ndimage.generic_filter(arr, np.amin, size = filter_size, mode = 'constant')
    

    def energy(arr):
        return np.sum(np.square(arr))
    energy_val = scipy.ndimage.generic_filter(arr, energy, size = filter_size, mode = 'constant')

    def std(arr):
        return np.std(arr, ddof=1)
    std_val = scipy.ndimage.generic_filter(arr, std, size = filter_size, mode = 'constant')

    if dim == 3:
        total = []
        labels = []
        for i in range(arr.shape[0] - filter_size + 1):
            for j in range(arr.shape[1] - filter_size + 1):
                for k in range(arr.shape[2] - filter_size + 1):
                    row = arr[i:i+filter_size, j:j+filter_size, k:k+filter_size].flatten()
                    row = np.append(row, mean[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                    row = np.append(row, maximum[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                    row = np.append(row, minimum[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                    row = np.append(row, energy_val[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                    row = np.append(row, std_val[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                    if has_labels:
                        label_row = label_arr[i:i+filter_size, j:j+filter_size, k:k+filter_size].flatten()
                        label = np.argmax(np.bincount(label_row))
                        labels.append(label)
                    total.append(row)
        result = np.array(total)
        x = arr.shape[0] - filter_size + 1
        y = arr.shape[1] - filter_size + 1
        z = arr.shape[2] - filter_size + 1
        total_size = x * y * z
    elif dim == 2:
        total = []
        for i in range(arr.shape[0] - filter_size + 1):
            for j in range(arr.shape[1] - filter_size + 1):
                row = arr[i:i+filter_size, j:j+filter_size].flatten()
                row = np.append(row, mean[i:i+filter_size,j:j+filter_size].flatten())
                row = np.append(row, maximum[i:i+filter_size,j:j+filter_size].flatten())
                row = np.append(row, minimum[i:i+filter_size,j:j+filter_size].flatten())
                row = np.append(row, energy_val[i:i+filter_size,j:j+filter_size].flatten())
                row = np.append(row, std_val[i:i+filter_size,j:j+filter_size].flatten())
                if has_labels:
                    label_row = label_arr[i:i+filter_size, j:j+filter_size].flatten()
                    label = np.argmax(np.bincount(label_row))
                    labels.append(label)
                total.append(row)
        result = np.array(total)
        x = arr.shape[0] - filter_size + 1
        y = arr.shape[1] - filter_size + 1
        total_size = x * y
    if has_labels:
        final_label_arr = np.array(result)
        return result, final_label_arr
    else:
        return result, None