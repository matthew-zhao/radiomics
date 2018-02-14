# http://stackoverflow.com/questions/31714788/can-an-aws-lambda-function-call-another
# Lambda 1

import boto3
import json
import dropbox
import csv
import scipy
import boto
from boto.s3.key import Key

lambda_client = boto3.client('lambda')

def invoke_lambda(event, context):
    conn = boto.connect_s3("AKIAJRKPLMXU3JRGWYCA","LFfFCpaEsdCCz4KiEBKoTFS5ehXIcPjsPq3yqxjj")

    paths = []
    if event["is_dropbox"]:
        # dclient = dropbox.client.DropboxClient(event["auth_token"])
        client = dropbox.Dropbox(event["auth_token"])
        #metadata = dclient.metadata(event["folder_name"])
        list_folder_result = client.files_list_folder(event["folder_name"])

        for metadata in list_folder_result.entries: 
            if isinstance(metadata, dropbox.files.FileMetadata):
                paths.append(metadata.path_display) #adds files to paths
                #paths is list of paths to files, has extensions
    else:
        b = conn.get_bucket(event["images_bucket"])
        bucket_list = b.list()

        for l in bucket_list:
            # as long as item in bucket is an image
            if l.key[-4:] != ".csv":
                paths.append(l.key)

    is_train = event["is_train"]
    has_labels = event["has_labels"]
    model_bucket_name = event["model_bucket_name"]
    shape_dir = None
    filter_size = 3
    last = False

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

    if has_labels:
        b = conn.get_bucket(event["images_bucket"])
        csv_key = b.get_key('trainLabels.csv')
        csv_key.get_contents_to_filename("/tmp/trainLabels.csv")

        c_reader = csv.reader(open('/tmp/trainLabels.csv', 'r'), delimiter = ',')
        columns = list(zip(*c_reader))
        images = columns[0][1:]
        levels = list(map(int, columns[1][1:]))
        label_dict = dict(zip(images, levels))

        image_list = []


        #gets rid of image paths who do not have a label in label_dict
        for image_path in paths:
            image_name = image_path.split("/")[-1]
            actual_name, extension = image_name.split(".")

            if not label_dict.has_key(actual_name):
                paths.remove(image_paths)
    else:
        has_labels = ""


    image_num = 1
    #this loop always happens
    for key in paths:
        image_path = key
        image_name = image_path.split("/")[-1]
        actual_name, extension = image_name.split(".")
        if has_labels and event["is_train"]:
            print("supervised training")
            if event["label_style"] == "single":
                label = label_dict[actual_name]
                args = {"image_path": image_path, "image_name": actual_name, "filter_size": filter_size, "image_num": image_num,
                        "auth_token": event["auth_token"], "is_train": event["is_train"], "has_labels": has_labels, "model_bucket_name": model_bucket_name,
                        "bucket_from": event["images_bucket"], "bucket_from_labels": "", "is_dropbox": event["is_dropbox"], "queue_name": model_bucket_name + '.fifo', "queue_name1": model_bucket_name + '1.fifo',
                        "num_classes": event["num_classes"], "num_machines": event["num_machines"], "label_style": event["label_style"], "label": label, "num_channels": event["num_channels"]}
            else:
                args = {"image_path": image_path, "image_name": actual_name, "filter_size": filter_size, "image_num": image_num,
                        "auth_token": event["auth_token"], "is_train": event["is_train"], "has_labels": has_labels, "model_bucket_name": model_bucket_name,
                        "bucket_from": event["images_bucket"], "bucket_from_labels": event["images_labels_bucket"], "is_dropbox": event["is_dropbox"], "queue_name": model_bucket_name + '.fifo', "queue_name1": model_bucket_name + '1.fifo',
                        "num_classes": event["num_classes"], "num_machines": event["num_machines"], "label_style": event["label_style"], "num_channels": event["num_channels"]}
        elif not has_labels and event["is_train"]:
            print("unsupervised training")
            args = {"image_path": image_path, "image_name": actual_name, "filter_size": filter_size, "image_num": image_num,
                    "auth_token": event["auth_token"], "is_train": event["is_train"], "has_labels": has_labels, "model_bucket_name": model_bucket_name,
                    "bucket_from": event["images_bucket"], "bucket_from_labels": "", "is_dropbox": event["is_dropbox"], "queue_name": model_bucket_name + '.fifo', "queue_name1": model_bucket_name + '1.fifo',
                    "num_classes": event["num_classes"], "num_machines": event["num_machines"], "num_channels": event["num_channels"]}

        else:
            print("testing")
            args = {"image_path": image_path, "image_name": actual_name, "filter_size": filter_size, "image_num": image_num,
                    "auth_token": event["auth_token"], "is_train": event["is_train"], "has_labels": has_labels, "model_bucket_name": model_bucket_name,
                    "bucket_from": event["images_bucket"], "bucket_from_labels": "", "is_dropbox": event["is_dropbox"], "queue_name": model_bucket_name + '.fifo', "queue_name1": model_bucket_name + '1.fifo',
                    "num_classes": event["num_classes"], "num_machines": event["num_machines"], "num_channels": event["num_channels"]}

        invoke_response = lambda_client.invoke(FunctionName="deep-preprocess2", InvocationType='Event', Payload=json.dumps(args))
        image_num += 1


