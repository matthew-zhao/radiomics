# http://stackoverflow.com/questions/31714788/can-an-aws-lambda-function-call-another
# Lambda 1

import boto3
import json
import dropbox
import csv
import os, errno
import boto
import numpy as np
from boto.s3.key import Key

lambda_client = boto3.client('lambda')

def clear_tmp_dir():
    # clear tmp directory
    for file in os.listdir("/tmp"):
        filepath = os.path.join("/tmp", file)
        try:
            if os.path.isfile(filepath):
                os.unlink(filepath)
        except Exception as e:
            print(e)

def invoke_lambda(event, context):
    clear_tmp_dir()
    conn = boto.connect_s3("AKIAJRKPLMXU3JRGWYCA","LFfFCpaEsdCCz4KiEBKoTFS5ehXIcPjsPq3yqxjj")
    is_dcm_series = False
    is_nrrd_labels = False
    is_3d = False

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

    # are the input folders in 3d
    if "is_3d" in event:
        is_3d = event["is_3d"]

    problems = {"classification", "segmentation", "regression"}

    if "problem" in event:
        if event["problem"] not in problems:
            return "Type of problem doesn't exist!"
        problem = event["problem"]
    else:
        problem = "classification"

    b = conn.get_bucket(event["images_bucket"])
    bucket_list = b.list()

    paths = []
    label_paths = []
    image_key_names = []

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
        # b_labels = conn.get_bucket(event["images_labels_bucket"])
        last_dir = None
        for l in bucket_list:
            if l.key[-4:] == ".csv":
                continue

            if is_dcm_series and l.key[-4:] == ".dcm":
                # we will assume that each folder represents one patient
                filename = os.path.join("/tmp", l.key)
                parent_dir = os.path.dirname(filename)
                if last_dir and last_dir != parent_dir:
                    image_key_names, paths = process_dcm_series(last_dir, image_key_names, paths, b)
                    clear_tmp_dir()
                if not os.path.exists(parent_dir):
                    try:
                        os.makedirs(parent_dir)
                    except OSError as e:
                        if e.errno != errno.EEXIST:
                            raise
                folder_key = b.get_key(l.key)
                folder_key.get_contents_to_filename(filename)
                last_dir = parent_dir
            # as long as item in bucket is an image
            else:
                paths.append(l.key)

    if has_labels:
        csv_key = b.get_key(labels_key_name)
        csv_key.get_contents_to_filename("/tmp/trainLabels.csv")

        c_reader = csv.reader(open('/tmp/trainLabels.csv', 'r'), delimiter = ',')
        columns = list(zip(*c_reader))
        images = columns[0][1:]
        if label_dtype == "int":
            levels = list(map(int, columns[1][1:]))
        elif label_dtype == "float":
            levels = list(map(float, columns[1][1:]))
        else:
            levels = list(columns[1][1:])
        label_dict = dict(zip(images, levels))

        if not is_dcm_series:
            #gets rid of image paths who do not have a label in label_dict
            for image_path in paths:
                image_name = image_path.split("/")[-1]

                actual_name, extension = image_name.rsplit(".", 1)

                if not label_dict.has_key(actual_name):
                    paths.remove(image_paths)
    else:
        has_labels = ""


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

    queue_url = client.get_queue_url(QueueName='called-' + model_bucket_name + '.fifo')
    msg_response = client.send_message(QueueUrl=queue_url['QueueUrl'], MessageBody='called', MessageDeduplicationId="deduplicationId", MessageGroupId="groupId")

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
    #this loop always happens
    for i in range(len(paths)):
        image_path = paths[i]
        if has_labels and is_train:
            print("supervised training")
            # if label is None, means that we will rely on separate images_labels_bucket to get labels (if label_style is array)
            # or something is wrong (if label_style is single)
            label = None
            if is_dcm_series:
                # if it's a dicom series, we will set "image_num" as the slice # in the series
                actual_path = image_key_names[i]
                label_path, image_num = actual_path.rsplit('_', 1)
                label = label_dict[label_path]
            else:
                image_name = image_path.split("/")[-1]
                actual_name, _ = image_name.rsplit(".", 1)
                label = label_dict[actual_name]
            if event["label_style"] == "single":
                args = {"image_path": image_path, "x_scale": x_scale, "y_scale": y_scale, "filter_size": filter_size, "image_num": image_num,
                        "auth_token": event["auth_token"], "is_train": event["is_train"], "has_labels": has_labels, "model_bucket_name": model_bucket_name,
                        "bucket_from": event["images_bucket"], "bucket_from_labels": "", "is_dropbox": event["is_dropbox"], "queue_name": model_bucket_name + '.fifo', "queue_name1": model_bucket_name + '1.fifo',
                        "num_classes": event["num_classes"], "num_machines": event["num_machines"], "label_style": event["label_style"], "label": label, "num_channels": event["num_channels"]}
            else:
                args = {"image_path": image_path, "x_scale": x_scale, "y_scale": y_scale, "filter_size": filter_size, "image_num": image_num,
                        "auth_token": event["auth_token"], "is_train": event["is_train"], "has_labels": has_labels, "model_bucket_name": model_bucket_name,
                        "bucket_from": event["images_bucket"], "bucket_from_labels": event["images_labels_bucket"], "is_dropbox": event["is_dropbox"], "queue_name": model_bucket_name + '.fifo', "queue_name1": model_bucket_name + '1.fifo',
                        "num_classes": event["num_classes"], "num_machines": event["num_machines"], "label_style": event["label_style"], "label": label, "num_channels": event["num_channels"], "is_dcm_series": is_dcm_series}
        elif not has_labels and is_train:
            print("unsupervised training")
            args = {"image_path": image_path, "x_scale": x_scale, "y_scale": y_scale, "filter_size": filter_size, "image_num": image_num,
                    "auth_token": event["auth_token"], "is_train": event["is_train"], "has_labels": has_labels, "model_bucket_name": model_bucket_name,
                    "bucket_from": event["images_bucket"], "bucket_from_labels": "", "is_dropbox": event["is_dropbox"], "queue_name": model_bucket_name + '.fifo', "queue_name1": model_bucket_name + '1.fifo',
                    "num_classes": event["num_classes"], "num_machines": event["num_machines"], "num_channels": event["num_channels"]}

        else:
            print("testing")
            args = {"image_path": image_path, "x_scale": x_scale, "y_scale": y_scale, "filter_size": filter_size, "image_num": image_num,
                    "auth_token": event["auth_token"], "is_train": event["is_train"], "has_labels": has_labels, "model_bucket_name": model_bucket_name,
                    "bucket_from": event["images_bucket"], "bucket_from_labels": "", "is_dropbox": event["is_dropbox"], "queue_name": model_bucket_name + '.fifo', "queue_name1": model_bucket_name + '1.fifo',
                    "num_classes": event["num_classes"], "num_channels": event["num_channels"], "label_style": event["label_style"]}

        if problem == "segmentation":
            invoke_response = lambda_client.invoke(FunctionName="deep-seg-preprocess2", InvocationType='Event', Payload=json.dump(args))
        else:
            invoke_response = lambda_client.invoke(FunctionName="deep-preprocess2", InvocationType='Event', Payload=json.dumps(args))
        image_num += 1


