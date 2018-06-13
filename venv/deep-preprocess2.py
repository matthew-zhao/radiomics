# preprocessing3
import dropbox
import numpy as np
import boto
import boto3
import json
import random
import scipy
import dicom

from boto.s3.key import Key
from PIL import Image, ImageFile
from StringIO import StringIO

# Function to be called by lambda2 when all images are done being preprocessed
# to squish them together
lambda_client = boto3.client('lambda', region_name='us-west-2')
ImageFile.LOAD_TRUNCATED_IMAGES = True

def squish(event, context):
    conn = boto.connect_s3("AKIAJRKPLMXU3JRGWYCA","LFfFCpaEsdCCz4KiEBKoTFS5ehXIcPjsPq3yqxjj")
    #conn = boto.s3.connect_to_region('s3.us-west-2.amazonaws.com', aws_access_key_id="AKIAJRKPLMXU3JRGWYCA", aws_secret_access_key="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")
    b = conn.get_bucket(event['bucket_from'])
    image_path = event["image_path"]

    has_labels = event["has_labels"]
    is_train = event["is_train"]
    image_name = event["image_name"]
    model_bucket_name = event["model_bucket_name"]
    image_num = event["image_num"]
    queue_name = event["queue_name"]
    queue_name1 = event["queue_name1"]
    x_scale = event["x_scale"]
    y_scale = event["y_scale"]

    final_trainimages_bucket = 'train-deepnorm'
    final_testimages_bucket = 'test-deepnorm'
    final_labels_bucket = 'train-deeplabels'

    bucket_list = b.list()

    if has_labels and event["label_style"] == "array":
        labels = conn.get_bucket(event['bucket_from_labels'])

    actual_name, extension = image_path.split(".")

    # Get image from dropbox or S3
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
        b = conn.get_bucket(event["bucket_from"])
        img_key = b.get_key(image_path)

        if extension == "dcm":
            img_key.get_contents_to_filename('/tmp/response_content.dcm')
        else:
            img_key.get_contents_to_filename('/tmp/' + image_path)
            img_raw = Image.open('/tmp/' + image_path)

    # Convert image to scipy array and scale image down
    if extension == "dcm":
        f2 = open("/tmp/response_content.dcm", "rb")
        ds = dicom.read_file(f2)
        img_raw = ds.pixel_array
        f2.close()
        xscale = x_scale / img_raw.shape[1]
        yscale = y_scale / img_raw.shape[0]
        img = scipy.ndimage.interpolation.zoom(img_raw, [xscale, yscale])
    else:
        img_resized = img_raw.resize((x_scale, y_scale), Image.ANTIALIAS)
        img = scipy.array(img_resized)

    # get array of labels or single label
    if has_labels:
        if event["label_style"] == "array":
            npy_filename = actual_name.split("/")[-1] + ".npy"
            print(npy_filename)
            labels_key = labels.get_key(npy_filename)
            labels_key.get_contents_to_filename("/tmp/labels-" + npy_filename)

            with open("/tmp/labels-" + npy_filename, "rb") as npy:
                label_arr = np.load(npy)
        elif event["label_style"] == "single":
            label_arr = np.array([event["label"]])

    training_arr = img

    X_converted = training_arr.astype(np.float16)
    X_train = X_converted / 255

    if has_labels:
        y_converted = label_arr.astype(np.float16)
        targets = y_converted.reshape(-1)
        y_train = np.eye(int(event["num_classes"]))[targets.astype('int8')]

    # Create new buckets for the array and its corresponding labels
    if is_train:
        b2 = conn.get_bucket(final_trainimages_bucket)
        bucket_location = b2.get_location()
        if bucket_location:
            conn = boto.s3.connect_to_region(bucket_location)
            b2 = conn.get_bucket(final_trainimages_bucket)
    else:
        b2 = conn.get_bucket(final_testimages_bucket)

    k = b2.new_key(str(image_num) + "-processed.npy")

    # Save the numpy arrays to temp .npy files on lambda
    upload_path = '/tmp/resized-matrix.npy'
    np.save(upload_path, X_train)

    # Take the tempfile that has the concatanated final array and set the contents
    # of the new bucket's key
    k.set_contents_from_filename(upload_path)

    # make the file public so it can be accessed publicly
    # using a URL like http://s3.amazonaws.com/bucket_name/key
    k.make_public()

    if has_labels:
        labels2 = conn.get_bucket(final_labels_bucket)
        k2 = labels2.new_key(str(image_num) + "label-processed.npy")

        upload_path_labels = '/tmp/resized-labels.npy'
        np.save(upload_path_labels, y_train)
        #np.save(upload_path_labels, label_arr)

        k2.set_contents_from_filename(upload_path_labels)
        k2.make_public()



    #if not training, each deep-preprocess2 calls a predict
    if not is_train:
        args = {"classifier": "neural", "bucket_from": final_testimages_bucket, "model_bucket": "model-train", "model_bucket_name": model_bucket_name, "result_bucket": "prediction-labels", "image_name": image_name, "result_name": image_name + str(image_num),
            "image_num": str(image_num), "xscale": x_scale, "yscale": y_scale, "label_style": event["label_style"], "num_classes": event["num_classes"], "num_channels": event["num_channels"]}
        invoke_response = lambda_client.invoke(FunctionName="deep-predict", InvocationType='Event', Payload=json.dumps(args))

    else:
        sqs = boto3.client('sqs')
        num = random.randrange(1,3)
        if num == 1:
            queue_url = sqs.get_queue_url(QueueName=queue_name)
        else:
            queue_url = sqs.get_queue_url(QueueName=queue_name1)
        receipt_msg = sqs.receive_message(
            QueueUrl=sqs.get_queue_url(QueueName='called-' + queue_name)['QueueUrl'],
            AttributeNames=['All'],
            MaxNumberOfMessages=1,
            MessageAttributeNames=['All'],
            VisibilityTimeout=2,
            WaitTimeSeconds=20
        )
        is_called = None
        if 'Messages' in receipt_msg:
            message = receipt_msg['Messages'][0]
            receipt_handle = message['ReceiptHandle']
            is_called = message['Body']
            print(is_called)

        response = sqs.send_message(QueueUrl=queue_url['QueueUrl'], MessageBody=str(image_num), MessageDeduplicationId="deduplicationId" + str(image_num), MessageGroupId="groupId")
        print(response)
        #only invoke neuralnet.py once, by the first preprocessing3 to finish
        if is_called:
            print("Neuralnet invoked from " + str(image_num))
            sqs.delete_message(
                QueueUrl=sqs.get_queue_url(QueueName='called-' + queue_name)['QueueUrl'],
                ReceiptHandle=receipt_handle
            )

            args = {"bucket_from": final_trainimages_bucket, "bucket_from_labels" : final_labels_bucket, "model_bucket_name": model_bucket_name, 
                    "image_num": str(image_num), "image_name": image_name, "xscale": x_scale, "yscale": y_scale, "queue_name": queue_name, "queue_name1": queue_name1, 
                    "num_classes": event["num_classes"], "num_machines": event["num_machines"], "called_from": "pre3", "num_channels": event["num_channels"]}
            invoke_response = lambda_client.invoke(FunctionName="deep-averager", InvocationType='Event', Payload=json.dumps(args))
            print(invoke_response)

    return 0