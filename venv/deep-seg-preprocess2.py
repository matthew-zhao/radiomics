# preprocessing3
import dropbox
import numpy as np
import boto
import boto3
import json
import random
import pydicom
import nrrd
import dicom2nifti
import skimage

from boto.s3.key import Key
from PIL import Image, ImageFile
from StringIO import StringIO

# Function to be called by lambda2 when all images are done being preprocessed
# to squish them together
lambda_client = boto3.client('lambda', region_name='us-west-2')
ImageFile.LOAD_TRUNCATED_IMAGES = True

# add to special extensions that cannot be resized with PIL/need special attention
SPECIAL_EXTENSION = set(["dcm", "npy", "nrrd"])

def stream_from_s3(key):
    data_string = StringIO(key.get_contents_to_string())
    return np.load(data_string)

def download_from_s3(bucket, key_name, filename):
    key = bucket.get_key(key_name)
    key.get_contents_to_filename(filename)

def squish(event, context):
    conn = boto.connect_s3("AKIAJRKPLMXU3JRGWYCA","LFfFCpaEsdCCz4KiEBKoTFS5ehXIcPjsPq3yqxjj")
    #conn = boto.s3.connect_to_region('s3.us-west-2.amazonaws.com', aws_access_key_id="AKIAJRKPLMXU3JRGWYCA", aws_secret_access_key="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")
    b = conn.get_bucket(event['bucket_from'])
    image_path = event["image_path"]

    has_labels = event["has_labels"]
    is_train = event["is_train"]
    image_name = os.path.normpath(image_path).split(os.path.sep)[-1]
    actual_name, extension = image_name.rsplit(".", 1)
    model_bucket_name = event["model_bucket_name"]
    image_num = event["image_num"]
    queue_name = event["queue_name"]
    queue_name1 = event["queue_name1"]
    x_scale = event["x_scale"]
    y_scale = event["y_scale"]

    is_dcm_series = False
    if "is_dcm_series" in event:
        is_dcm_series = event["is_dcm_series"]

    final_trainimages_bucket = 'train-deepnorm'
    final_testimages_bucket = 'test-deepnorm'
    final_labels_bucket = 'train-deeplabels'

    bucket_list = b.list()

    if has_labels and event["label_style"] == "array":
        labels = conn.get_bucket(event['bucket_from_labels'])

    # Get image from dropbox or S3
    if event["is_dropbox"]:
        #access the dropbox folder using auth token and image path
        client = dropbox.Dropbox(event["auth_token"])
        f, metadata = client.files_download(image_path)
        data = metadata.content

        if extension in SPECIAL_EXTENSION:
            f2 = open("/tmp/response_content." + extension, "wb")
            f2.write(data)
            f2.close()
        else:
            img_raw = Image.open(StringIO(data))
    else:
        b = conn.get_bucket(event["bucket_from"])
        img_key = b.get_key(image_path)

        if extension == "dcm" or extension == "nrrd":
            img_key.get_contents_to_filename('/tmp/response_content.' + extension)
        elif extension == "npy":
            # the reason why we decide to stream is because we don't want to put too much
            # stuff in tmp folder, which can only hold like 500 MB, while there is 3 GB memory available
            img_raw = stream_from_s3(img_key)
        else:
            img_key.get_contents_to_filename('/tmp/' + image_path)
            img_raw = Image.open('/tmp/' + image_path)

    # Convert image to np array and scale image down
    if extension in SPECIAL_EXTENSION:
        if extension == "dcm":
            f2 = open("/tmp/response_content.dcm", "rb")
            ds = pydicom.read_file(f2)
            img_raw = ds.pixel_array
            f2.close()
        xscale = x_scale / img_raw.shape[1]
        yscale = y_scale / img_raw.shape[0]
        #img = scipy.ndimage.interpolation.zoom(img_raw, [xscale, yscale])
        img = skimage.transform.resize(img_raw, (xscale, yscale), anti_aliasing=True)
    else:
        img_resized = img_raw.resize((x_scale, y_scale), Image.ANTIALIAS)
        img = np.array(img_resized)

    # get array of labels or single label
    if has_labels:
        label = event["label"]
        if event["label_style"] == "array":
            if is_dcm_series:
                # label is actually a path to label in s3, image_num is the slice # we need to use in the nrrd file (3rd dimension)
                # read in nrrd file from S3, it's stored in the same bucket as training images
                local_labels_path = os.path.normpath(label).split(os.path.sep)
                local_path_as_string = '_'.join(local_labels_path)
                final_local_path = os.path.join("/tmp", local_path_as_string)
                download_from_s3(labels, label, final_local_path)
                # readdata is the np array
                readdata, _ = nrrd.read(final_local_path)

                # get the appropriate slice of the image
                label_arr = readdata[...,int(image_num)]
            else:
                npy_filename = actual_name + ".npy"
                print(npy_filename)
                download_from_s3(labels, npy_filename, "/tmp/labels-" + npy_filename)

                with open("/tmp/labels-" + npy_filename, "rb") as npy:
                    label_arr = np.load(npy)
        elif event["label_style"] == "single":
            label_arr = np.array([label])

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

    k = b2.new_key(actual_name + str(image_num) + "-processed.npy")

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
        k2 = labels2.new_key(actual_name + str(image_num) + "label-processed.npy")

        upload_path_labels = '/tmp/resized-labels.npy'
        np.save(upload_path_labels, y_train)
        #np.save(upload_path_labels, label_arr)

        k2.set_contents_from_filename(upload_path_labels)
        k2.make_public()



    #if not training, each deep-preprocess2 calls a predict
    if not is_train:
        args = {"classifier": "neural", "bucket_from": final_testimages_bucket, "model_bucket": "model-train", "model_bucket_name": model_bucket_name, "result_bucket": "prediction-labels", "result_name": actual_name + str(image_num),
            "xscale": x_scale, "yscale": y_scale, "label_style": event["label_style"], "num_classes": event["num_classes"], "num_channels": event["num_channels"]}
        invoke_response = lambda_client.invoke(FunctionName="deep-seg-predict", InvocationType='Event', Payload=json.dumps(args))

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
                    "image_num": str(image_num), "xscale": x_scale, "yscale": y_scale, "queue_name": queue_name, "queue_name1": queue_name1, 
                    "num_classes": event["num_classes"], "num_machines": event["num_machines"], "called_from": "pre3", "num_channels": event["num_channels"]}
            invoke_response = lambda_client.invoke(FunctionName="deep-seg-averager", InvocationType='Event', Payload=json.dumps(args))
            print(invoke_response)

    return 0