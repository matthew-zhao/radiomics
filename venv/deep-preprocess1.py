# http://stackoverflow.com/questions/31714788/can-an-aws-lambda-function-call-another
# Lambda 1

import boto3
import json
import dropbox
import csv
import os, errno
import boto
import nrrd
import numpy as np
import random
from boto.s3.key import Key

from StringIO import StringIO

from process_dicom import *
from welford import Welford
import ctypes

# for d, dirs, files in os.walk('lib'):
#     for f in files:
#         if f.endswith('.a') or f.startswith('.'):
#             continue
#         ctypes.cdll.LoadLibrary(os.path.join(d, f))
import image

lambda_client = boto3.client('lambda')

dicom_image_types = {"dicom", "nrrd"}

def determine_key(image_type, image_path=None, image_key_name=None):
    image_num = -1
    if image_type in dicom_image_types:
        # if it's a dicom series, we will set "image_num" as the slice # in the series
        actual_path = image_key_name
        label_path, image_num = actual_path.rsplit('_', 1)
        return label_path, image_num
    else:
        image_name = image_path.split("/")[-1]
        actual_name, _ = image_name.rsplit(".", 1)
        return actual_name, None

def clear_tmp_dir():
    # clear tmp directory
    for file in os.listdir("/tmp"):
        filepath = os.path.join("/tmp", file)
        try:
            if os.path.isfile(filepath):
                os.unlink(filepath)
        except Exception as e:
            print(e)

def mkdir_recursive(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def get_S3_results_as_iterator(key_obj):        
    unfinished_line = ''
    for byte in key_obj:
        byte = unfinished_line + byte
        #split on whatever, or use a regex with re.split()
        lines = byte.split('\n')
        unfinished_line = lines.pop()
        for line in lines:
            yield line

def invoke_lambda(event, context):
    clear_tmp_dir()
    conn = boto.connect_s3("AKIAJRKPLMXU3JRGWYCA","LFfFCpaEsdCCz4KiEBKoTFS5ehXIcPjsPq3yqxjj")
    is_dcm_series = False
    is_nrrd_labels = False
    has_slices = False
    # by default use 5 epochs. if want to use advanced stopping, will have to pass in
    num_epochs = 5  

    is_train = event["is_train"]
    has_labels = event["has_labels"]
    model_bucket_name = event["model_bucket_name"]
    shape_dir = None
    filter_size = 3
    last = False

    labels_key_name = None
    label_dtype = "string"
    if "label_dtype" in event:
        label_dtype = event["label_dtype"]

    if "labels_key_name" in event:
        labels_key_name = event["labels_key_name"]

    # is the input folders each with a series inside them?
    if "image_type" in event and event["image_type"] == "dicom":
        is_dcm_series = True

    if "label_type" in event and event["label_type"] == "nrrd":
        is_nrrd_labels = True

    if "num_epochs" in event:
        num_epochs = int(event["num_epochs"])

    # by default, we want to reserve these lambdas to only be training or test,
    # but we give users the options to specify train-dev-test split
    if is_train:
        train_split = 1.0
        dev_split = 0.0
        test_split = 0.0
    else:
        train_split = 0.0
        dev_split = 0.0
        test_split = 1.0

    if "train_split" in event:
        train_split = float(event["train_split"])
    if "dev_split" in event:
        dev_split = float(event["dev_split"])
    if "test_split" in event:
        test_split = float(event["test_split"])

    image_to_num_slices = {}
    num_slices = 0
    # are the input folders in 3d
    if "has_slices" in event:
        has_slices = event["has_slices"]

    problems = {"classification", "segmentation", "regression"}

    if "problem" in event:
        if event["problem"] not in problems:
            return "Type of problem doesn't exist!"
        problem = event["problem"]
    else:
        problem = "classification"

    image_bucket_name = event["images_bucket"]
    b = conn.get_bucket(image_bucket_name)
    bucket_list = b.list()

    paths = []
    label_paths = []
    image_key_names = []

    b_labels = conn.get_bucket(event["images_labels_bucket"])
    if has_labels:
        csv_key = b_labels.get_key(labels_key_name)
        csv_key.get_contents_to_filename("/tmp/trainLabels.csv")

        c_reader = csv.reader(open('/tmp/trainLabels.csv', 'r'), delimiter = ',')
        columns = list(zip(*c_reader))
        images = columns[0][1:]
        if label_dtype == "int":
            label_value = list(map(int, columns[1][1:]))
        elif label_dtype == "float":
            label_value = list(map(float, columns[1][1:]))
        else:
            label_value = list(columns[1][1:])
        image_label_tups = zip(images, label_value)

        random.shuffle(image_label_tups)

        # split dictionary into train, dev, and test set
        total_size = len(image_label_tups)
        train_dict_size = 0
        dev_dict_size = 0
        test_dict_size = 0
        train_dict = {}
        dev_dict = {}
        test_dict = {}

        if train_split > 0:
            train_dict_size = int(len(image_label_tups) * train_split)
            train_dict = dict(image_label_tups[:train_dict_size])

        if dev_split > 0:
            dev_dict_size = int(len(image_label_tups) * dev_split)
            dev_dict = dict(image_label_tups[train_dict_size:train_dict_size+dev_dict_size])

        if test_split > 0:
            test_dict_size = total_size - train_dict_size - test_dict_size
            test_dict = dict(image_label_tups[train_dict_size+dev_dict_size:])

        # TODO: NOT SURE WE NEED THIS PIECE OF CODE
        # if not is_dcm_series:
        #     #gets rid of image paths who do not have a label in label_dict
        #     for image_path in paths:
        #         image_name = image_path.split("/")[-1]

        #         actual_name, extension = image_name.rsplit(".", 1)

        #         if not label_dict.has_key(actual_name):
        #             paths.remove(image_paths)
    else:
        has_labels = ""

    mean = 0
    std_dev = 0
    if event["is_dropbox"]:
        # dclient = dropbox.client.DropboxClient(event["auth_token"])
        client = dropbox.Dropbox(event["auth_token"])
        #metadata = dclient.metadata(event["folder_name"])
        list_folder_result = client.files_list_folder(event["folder_name"])

        for metadata in list_folder_result.entries:
            # remove this if check because we are okay with folders, we look at dicom images
            if isinstance(metadata, dropbox.files.FileMetadata):
                paths.append(metadata.path_display) #adds files to paths
                #paths is list of paths to files, has extensions

        if event["label_style"] == "array":
            list_label_folder_result = client.files_list_folder(event["label_folder_name"])

            for label_metadata in list_label_folder_result.entries:
                if isinstance(label_metadata, dropbox.files.FileMetadata):
                    label_paths.append(label_metadata.path_display)
    else:
        last_dir = None
        total_value = 0
        total_size = 0
        #running_std_welford = Welford()
        mean = 0
        std_dev = 0
        for l in bucket_list:
            if l.key[-4:] == ".csv" or l.key.endswith(os.sep):
                continue

            if is_dcm_series and l.key[-4:] == ".dcm":
                # we will assume that each folder represents one patient
                filename = os.path.join("/tmp", l.key)
                parent_dir = os.path.dirname(filename)
                if last_dir and last_dir != parent_dir:
                    image_key_names, paths = process_dcm_series(last_dir, image_key_names, paths, b)
                    clear_tmp_dir()
                mkdir_recursive(parent_dir)
                folder_key = b.get_key(l.key)
                folder_key.get_contents_to_filename(filename)
                last_dir = parent_dir
            elif l.key[-5:] == ".nrrd":
                # figure out how many slices the nrrd image has
                filename = os.path.join("/tmp", l.key)
                parent_dir = os.path.dirname(filename)
                mkdir_recursive(parent_dir)
                #key_iterator = get_S3_results_as_iterator(l)
                nrrd_key = b.get_key(l.key)
                fh = StringIO(nrrd_key.get_contents_as_string())
                header = nrrd.read_header(fh)
                data = nrrd.read_data(header, fh, filename=None)

                total_size += data.size
                total_value += np.sum(data)
                #running_std_welford.add_data(data, slices=True)

                image_num_slices = data.shape[2]
                num_slices += image_num_slices
                image_to_num_slices[l.key] = image_num_slices
                paths.append(l.key)
            # as long as item in bucket is an image
            else:
                paths.append(l.key)
        #mean = running_std_welford.mean()
        #std_dev = running_std_welford.std()
        if total_size > 0 or total_value > 0:
            mean = total_value / total_size
            
            total_std = 0
            for l in bucket_list:
                if l.key[-5:] == ".nrrd":
                    nrrd_key = b.get_key(l.key)
                    fh = StringIO(nrrd_key.get_contents_as_string())
                    header = nrrd.read_header(fh)
                    data = nrrd.read_data(header, fh, filename=None)

                    total_std += np.sum(np.square(data - mean))
            std_dev = np.sqrt(total_std / total_size)


        print("Mean is " + str(mean))
        print("Std dev is " + str(std_dev))

    client = boto3.client('sqs')
    response = client.create_queue(
        QueueName=model_bucket_name + '.fifo',
        Attributes={
            'FifoQueue': 'true',
            'ContentBasedDeduplication': 'true'
        }
    )
    response = client.create_queue(
        QueueName=model_bucket_name + '1.fifo',
        Attributes={
            'FifoQueue': 'true',
            'ContentBasedDeduplication': 'true'
        }
    )
    response = client.create_queue(
        QueueName='called-' + model_bucket_name + '.fifo',
        Attributes={
            'FifoQueue': 'true',
            'ContentBasedDeduplication': 'true'
        }
    )

    # this queue used to determine whether every test image has been predicted on
    response = client.create_queue(
        QueueName='predict-' + model_bucket_name + '.fifo',
        Attributes={
            'FifoQueue': 'true',
            'ContentBasedDeduplication': 'true'
        }
    )

    queue_url = client.get_queue_url(QueueName='called-' + model_bucket_name + '.fifo')
    msg_response = client.send_message(
        QueueUrl=queue_url['QueueUrl'], 
        MessageBody='called', 
        MessageDeduplicationId="deduplicationId", 
        MessageGroupId="groupId"
    )

    # scale of images is by default 227x227
    if "x_scale" in event:
        x_scale = event["x_scale"]
    else:
        x_scale = 227

    if "y_scale" in event:
        y_scale = event["y_scale"]
    else:
        y_scale = 227

    image_num = 1
    # generally, we want to consider the number of 
    # counted images, but we will take number of
    # slices as precedent (if > 0 will likely be true # of preprocessings)
    num_invocations = len(paths)
    if num_slices > 0:
        num_invocations = num_slices
        volume_index = 0
    for i in range(num_invocations):
        if num_slices > 0:
            image_path = paths[volume_index]
            slices_left = image_to_num_slices[image_path]
            if slices_left <= 0:
                volume_index += 1
                image_path = paths[volume_index]
                slices_left = image_to_num_slices[image_path]
            image_num = slices_left - 1
            label_key = image_path
            image_to_num_slices[image_path] = slices_left - 1
        else:
            image_path = paths[i]
            image_key_name = None
            if i < len(image_key_names):
                image_key_name = image_key_names[i]
                label_key, image_num = determine_key(event["image_type"], image_path, image_key_name)
        #print(label_key)
        #print(train_dict)
        #print(dev_dict)
        #print(test_dict)
        if has_labels and is_train and label_key in train_dict:
            print("supervised training")
            # if label is None, means that we will rely on separate images_labels_bucket to get labels (if label_style is array)
            # or something is wrong (if label_style is single)
            if event["label_style"] == "single":
                args = {"image_path": image_path, "x_scale": x_scale, "y_scale": y_scale, "filter_size": filter_size, "image_num": image_num,
                        "auth_token": event["auth_token"], "is_train": event["is_train"], "has_labels": has_labels, "model_bucket_name": model_bucket_name,
                        "bucket_from": image_bucket_name, "bucket_from_labels": "", "is_dropbox": event["is_dropbox"], "queue_name": model_bucket_name + '.fifo', "queue_name1": model_bucket_name + '1.fifo',
                        "num_classes": event["num_classes"], "num_machines": event["num_machines"], "label_style": event["label_style"], "label": train_dict[label_key], "num_channels": event["num_channels"], 
                        "label_dict_type": "train_dict", "num_epochs": num_epochs, "has_slices": has_slices, "mean": mean, "std_dev": std_dev}
            else:
                args = {"image_path": image_path, "x_scale": x_scale, "y_scale": y_scale, "filter_size": filter_size, "image_num": image_num,
                        "auth_token": event["auth_token"], "is_train": event["is_train"], "has_labels": has_labels, "model_bucket_name": model_bucket_name,
                        "bucket_from": image_bucket_name, "bucket_from_labels": event["images_labels_bucket"], "is_dropbox": event["is_dropbox"], "queue_name": model_bucket_name + '.fifo', "queue_name1": model_bucket_name + '1.fifo',
                        "num_classes": event["num_classes"], "num_machines": event["num_machines"], "label_style": event["label_style"], "label": train_dict[label_key], "num_channels": event["num_channels"], "label_type": event["label_type"],
                        "label_dict_type": "train_dict", "num_epochs": num_epochs, "has_slices": has_slices, "mean": mean, "std_dev": std_dev}

        if not has_labels and is_train:
            print("unsupervised training")
            args = {"image_path": image_path, "x_scale": x_scale, "y_scale": y_scale, "filter_size": filter_size, "image_num": image_num,
                    "auth_token": event["auth_token"], "is_train": event["is_train"], "has_labels": has_labels, "model_bucket_name": model_bucket_name,
                    "bucket_from": image_bucket_name, "bucket_from_labels": "", "is_dropbox": event["is_dropbox"], "queue_name": model_bucket_name + '.fifo', "queue_name1": model_bucket_name + '1.fifo',
                    "num_classes": event["num_classes"], "num_machines": event["num_machines"], "num_channels": event["num_channels"], "num_epochs": num_epochs, "has_slices": has_slices, "mean": mean, "std_dev": std_dev}

        # if we're doing train-dev-test split, we need to hold off on calling predict...
        if not is_train or label_key in dev_dict or label_key in test_dict:
            print("testing")
            args = {"image_path": image_path, "x_scale": x_scale, "y_scale": y_scale, "filter_size": filter_size, "image_num": image_num,
                    "auth_token": event["auth_token"], "is_train": event["is_train"], "has_labels": has_labels, "model_bucket_name": model_bucket_name,
                    "bucket_from": image_bucket_name, "bucket_from_labels": "", "is_dropbox": event["is_dropbox"], "queue_name": model_bucket_name + '.fifo', "queue_name1": model_bucket_name + '1.fifo',
                    "num_classes": event["num_classes"], "num_channels": event["num_channels"], "label_style": event["label_style"], "has_slices": has_slices, "mean": mean, "std_dev": std_dev}

            if is_train:
                if event["label_style"] == "array":
                    args["label_type"] = event["label_type"]
                # since it's training, it's reasonable to expect labels bucket
                args["bucket_from_labels"] = event["images_labels_bucket"]
                if label_key in dev_dict:
                    args["label_dict_type"] = "dev_dict"
                    args["label"] = dev_dict[label_key]
                elif label_key in test_dict:
                    args["label_dict_type"] = "test_dict"
                    args["label"] = test_dict[label_key]

                if labels_key_name:
                    args["labels_key_name"] = labels_key_name

                # this queue is used by averager to determine how to call
                # preprocess2 for dev and test sets (for analysis)
                training_msg_response = client.send_message(
                    QueueUrl=queue_url['QueueUrl'], 
                    MessageBody=json.dumps(args),
                    MessageDeduplicationId="deduplicationId", 
                    MessageGroupId="groupId"
                )

                # this queue will be used after dev and test set
                # to determine when predict is done and we should analyze_results
                # TODO: Not sure if there is a way to 
                predict_done_queue_url = client.get_queue_url(QueueName='predict-' + model_bucket_name + '.fifo')
                predict_msg_response = client.send_message(
                    QueueUrl=predict_done_queue_url['QueueUrl'], 
                    MessageBody="blank", 
                    MessageDeduplicationId="deduplicationId", 
                    MessageGroupId="groupId"
                )
                # in this case, we don't want to invoke preprocess2
                continue

        if problem == "segmentation":
            invoke_response = lambda_client.invoke(FunctionName="deep-seg-preprocess2", InvocationType='Event', Payload=json.dumps(args))
        else:
            invoke_response = lambda_client.invoke(FunctionName="deep-preprocess2", InvocationType='Event', Payload=json.dumps(args))
        #image_num += 1


