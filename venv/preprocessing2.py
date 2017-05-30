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
    #access the dropbox folder using auth token and image path
    client = dropbox.Dropbox(event["auth_token"])
    image_path = event["image_path"]
    f, metadata = client.files_download(image_path)
    data = metadata.content

    #check whether it is training or testing set
    is_train = event["is_train"]

    actual_name, extension = image_path.split(".")

    if extension == "dcm":
        f2 = open("/tmp/response_content.dcm", "wb")
        f2.write(data)
        f2.close()
        f2 = open("/tmp/response_content.dcm", "rb")
        ds = dicom.read_file(f2)
        img_raw = ds.pixel_array
        f2.close()
        xscale = 324.0 / img_raw.shape[0]
        yscale = 243.0 / img_raw.shape[1]
        img = scipy.ndimage.interpolation.zoom(img_raw, [xscale, yscale])
    else:
        img_raw = Image.open(StringIO(data))
        img_resized = img_raw.resize((324, 243), Image.ANTIALIAS)
        img = scipy.array(img_resized)


    #connecting to appropriate S3 bucket
    conn = boto.connect_s3("AKIAIMQLHJNMP6DOUM4A","8dJAfPZlTjMR1SOcOetImclAmT+G02VkQiuHefdY")
    b = conn.get_bucket('training-array')
    b2 = conn.get_bucket('training-labels')
    k = b.new_key('matrix' + str(event["image_name"]) + '.npy')
    k2 = b2.new_key('matrix' + str(event["image_name"]) + '.npy')

    label = event['label']
    # last_bool = event['last']
    value_matrix, labels = analyze((img, label), event)

    #creating a temp image numpy file
    upload_path = '/tmp/matrix' + str(event["image_name"]) + '.npy'
    upload_path_labels = '/tmp/matrix' + str(event["image_name"]) + '-labels.npy'

    #saving image to S3 bucket
    np.save(upload_path, value_matrix)
    np.save(upload_path_labels, labels)
    k.set_contents_from_filename(upload_path)
    k2.set_contents_from_filename(upload_path_labels)

    #invoking 5 minute timer to ensure that preprocessing3 is called after all preprocessing2 processes are finished
    # msg = {"is_train": is_train}
    # lambda_client = boto3_client('lambda')
    # lambda_client.invoke(FunctionName="timer", InvocationType='Event', Payload=json.dumps(msg))
    # if last_bool:
    #     msg = {"bucket_from": "training-array", "bucket_from_labels": "training-labels", "is_train": is_train}
    #     lambda_client = boto3_client('lambda')
    #     lambda_client.invoke(FunctionName="preprocessing3", InvocationType='Event', Payload=json.dumps(msg))

    return 0

#calculating all the first-order statistics on all the images
def analyze(arr_arg, event):
    result = None
    arr = np.array(arr_arg[0])
    label = arr_arg[1]
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
        for i in range(arr.shape[0] - filter_size + 1):
            for j in range(arr.shape[1] - filter_size + 1):
                for k in range(arr.shape[2] - filter_size + 1):
                    row = arr[i:i+filter_size, j:j+filter_size, k:k+filter_size].flatten()
                    row = np.append(row, mean[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                    row = np.append(row, maximum[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                    row = np.append(row, minimum[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                    row = np.append(row, energy_val[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                    row = np.append(row, std_val[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
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
                row = arr[i:i+filter_size, j:j+filter_size, k:k+filter_size].flatten()
                row = np.append(row, mean[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                row = np.append(row, maximum[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                row = np.append(row, minimum[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                row = np.append(row, energy_val[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
                row = np.append(row, std_val[i:i+filter_size,j:j+filter_size,k:k+filter_size].flatten())
            total.append(row)
        result = np.array(total)
        x = arr.shape[0] - filter_size + 1
        y = arr.shape[1] - filter_size + 1
        total_size = x * y
    label_arr = np.full((1,total_size), label)
    return result, label_arr