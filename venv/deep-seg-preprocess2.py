import dropbox
import numpy as np
import boto
import boto3
import json
import random
import pydicom
import nrrd
#import dicom2nifti
import skimage
import skimage.transform
import os
import gc

import image
#import sklearn
#import sklearn.feature_extraction.image
#from sklearn.feature_extraction import image

from boto.s3.key import Key
from PIL import Image, ImageFile
from StringIO import StringIO
from io import BytesIO

# Function to be called by lambda2 when all images are done being preprocessed
# to squish them together
lambda_client = boto3.client('lambda', region_name='us-west-2')
ImageFile.LOAD_TRUNCATED_IMAGES = True

# add to special extensions that cannot be resized with PIL/need special attention
SPECIAL_EXTENSION = set(["dcm", "npy", "nrrd"])
TRAIN_TYPES = set(["train_dict"])
TEST_TYPES = set(["dev_dict", "test_dict"])

def normalize_image(image, mean, std_dev):
    return (image - mean) / std_dev

def stream_from_s3(bucket=None, key_name=None, key=None):
    if not key:
        key = bucket.get_key(key_name)
    data_string = StringIO(key.get_contents_as_string())
    return data_string

def download_from_s3(bucket, key_name, filename):
    key = bucket.get_key(key_name)
    key.get_contents_to_filename(filename)

def squish(event, context):
    conn = boto.connect_s3("AKIAJRKPLMXU3JRGWYCA","LFfFCpaEsdCCz4KiEBKoTFS5ehXIcPjsPq3yqxjj")
    #conn = boto.s3.connect_to_region('s3.us-west-2.amazonaws.com', aws_access_key_id="AKIAJRKPLMXU3JRGWYCA", aws_secret_access_key="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")
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
    has_slices = event["has_slices"]
    mean = float(event["mean"])
    std_dev = float(event["std_dev"])
    batch_size = 3

    # by default, we are only processing 1 image, but can be more
    num_patches = 1
    image_readdata_shape = None

    extract_patches = False
    label_dict_type = None
    if "label_dict_type" in event:
        label_dict_type = event["label_dict_type"]

    is_nrrd_labels = False
    if "label_type" in event and event["label_type"] == "nrrd":
        is_nrrd_labels = True
        image_header = None

    final_trainimages_bucket = 'train-deepnorm'
    final_testimages_bucket = 'test-deepnorm'
    final_labels_bucket = 'train-deeplabels'

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

        if extension == "dcm":
            img_key = b.get_key(image_path)
            img_key.get_contents_to_filename('/tmp/response_content.' + extension)
        elif extension == "nrrd":
            nrrd_fh = stream_from_s3(bucket=b, key_name=image_path)
            image_header = nrrd.read_header(nrrd_fh)
            image_readdata = nrrd.read_data(image_header, nrrd_fh, filename=None)
            #image_readdata = image_readdata.astype(np.float16)
            if len(image_readdata.shape) > 2:
                img_raw = image_readdata[...,int(image_num)]
            else:
                img_raw = image_readdata
            image_readdata_shape = image_readdata.shape
            del nrrd_fh
            del image_readdata
            gc.collect()
            img_raw = normalize_image(img_raw, mean, std_dev)
        elif extension == "npy":
            img_key = b.get_key(image_path)
            # the reason why we decide to stream is because we don't want to put too much
            # stuff in tmp folder, which can only hold like 500 MB, while there is 3 GB memory available
            img_raw = np.load(stream_from_s3(key=img_key))
        else:
            img_key = b.get_key(image_path)
            img_key.get_contents_to_filename('/tmp/' + image_path)
            img_raw = Image.open('/tmp/' + image_path)

    # Convert image to np array and scale image down
    if extension in SPECIAL_EXTENSION:
        if extension == "dcm":
            f2 = open("/tmp/response_content.dcm", "rb")
            ds = pydicom.read_file(f2)
            img_raw = ds.pixel_array
            f2.close()
        # TODO: only need these next two lines if we are not extracting patches
        # TODO2: also need to make sure labels have the same rescaling
        #img_resized = skimage.transform.resize(img_raw, (x_scale, y_scale), preserve_range=True, anti_aliasing=True)
        #img = np.rint(img_resized).astype(img_raw.dtype)

        if img_raw.shape[0] > 64 and img_raw.shape[1] > 64:
            # TODO: this might be too big to fit into memory
            # img_patches is either 3D or 4D array (n_patches, patch_height, patch_width)
            img = image.extract_patches_2d(img_raw, (64, 64), max_patches=64, random_state=0)
            del img_raw
            gc.collect()
            num_patches = img.shape[0]
            extract_patches = True
        else:
            img_resized = skimage.transform.resize(img_raw, (64, 64), preserve_range=True, anti_aliasing=True)
            img = np.rint(img_resized).astype(img_raw.dtype)
    else:
        img_resized = img_raw.resize((x_scale, y_scale), Image.ANTIALIAS)
        img = np.array(img_resized)

    # Create new buckets for the array and its corresponding labels
    if is_train and label_dict_type in TRAIN_TYPES:
        b2 = conn.get_bucket(final_trainimages_bucket)
        bucket_location = b2.get_location()
        if bucket_location:
            conn = boto.s3.connect_to_region(bucket_location)
            b2 = conn.get_bucket(final_trainimages_bucket)
    else:
        b2 = conn.get_bucket(final_testimages_bucket)

    k = b2.new_key(os.path.join(actual_name, str(image_num) + "-processed.npy"))

    # Save the numpy arrays to temp .npy files on lambda
    try:
        print("Trying to save to numpy file")
        upload_path = '/tmp/resized-matrix.npy'
        np.save(upload_path, img)
        del img
        gc.collect()
        k.set_contents_from_filename(upload_path)
    except:
        print("Excepting")
        output_file_object = BytesIO()
        np.save(output_file_object, img)
        del img
        gc.collect()
        output_file_object.seek(0)
        k.set_contents_from_file(output_file_object)
        del output_file_object
        gc.collect()
    #if extract_patches:
    print("succeeded in saving to numpy file")
    
    k.make_public()

    k = b2.new_key(os.path.join(actual_name, str(image_num) + "-numpatches.txt"))
    k.set_contents_from_string(str(num_patches))
    k.make_public()

    k = None
    output_file_object = None

    # get array of labels or single label
    if has_labels:
        if event["label_style"] == "array":
            labels = conn.get_bucket(event['bucket_from_labels'])
        label = event["label"]
        print(image_path)
        print(label)
        print(image_num)
        if event["label_style"] == "array":
            if is_nrrd_labels:
                # label is actually a path to label in s3, image_num is the slice # we need to use in the nrrd file (3rd dimension)
                # read in nrrd file from S3, it's stored in the same bucket as training images
                fh = stream_from_s3(bucket=labels, key_name=label)
                label_header = nrrd.read_header(fh)
                label_readdata = nrrd.read_data(label_header, fh, filename=None)

                origin_diffs = np.array(map(float, label_header['space origin'])) - np.array(map(float, image_header['space origin'])) 
                converted_space_directions = np.sum(np.array([map(float, row) for row in image_header['space directions']]), axis=1)
                pixel_diffs_front = np.divide(origin_diffs, converted_space_directions)

                pixel_coords_back = pixel_diffs_front + label_readdata.shape
                pixel_diffs_back = image_readdata_shape - pixel_coords_back

                pixel_diffs_front_int = np.rint(pixel_diffs_front).astype(int)
                pixel_diffs_back_int = np.rint(pixel_diffs_back).astype(int)

                label_readdata = np.pad(label_readdata, tuple(zip(pixel_diffs_front_int, pixel_diffs_back_int)), 'constant')
                # get the appropriate slice of the image
            else:
                npy_filename = actual_name + ".npy"
                print(npy_filename)
                label_readdata = np.load(stream_from_s3(bucket=labels, key_name=npy_filename))

            if len(label_readdata.shape) > 2:
                y_train = label_readdata[...,int(image_num)]
            else:
                y_train = label_readdata

            if y_train.shape[0] > 64 and y_train.shape[1] > 64 and extract_patches:
                # TODO: this might be too big to fit into memory
                # img_patches is either 3D or 4D array (n_patches, patch_height, patch_width)
                y_train_patches = image.extract_patches_2d(y_train, (64, 64), max_patches=64, random_state=0)
                del y_train
                gc.collect()
            else:
                y_train_resized = skimage.transform.resize(y_train, (64, 64), preserve_range=True, anti_aliasing=True)
                y_train = np.rint(y_train_resized).astype(y_train.dtype)

        elif event["label_style"] == "single":
            label_arr = np.array([label])

            y_converted = label_arr.astype(np.float16)
            targets = y_converted.reshape(-1)
            y_train = np.eye(int(event["num_classes"]))[targets.astype('int8')]

        labels2 = conn.get_bucket(final_labels_bucket)
        k = labels2.new_key(os.path.join(actual_name, str(image_num) + "label-processed.npy"))

        #upload_path_labels = '/tmp/resized-labels.npy'
        #np.save(upload_path_labels, y_train_patches)
        output_file_object = BytesIO()
        np.save(output_file_object, y_train_patches)
        del y_train_patches
        gc.collect()
        output_file_object.seek(0)

        #k.set_contents_from_filename(upload_path_labels)
        k.set_contents_from_file(output_file_object)
        k.make_public()


    #if not training, each deep-preprocess2 calls a predict
    if (not is_train) or label_dict_type in TEST_TYPES:
        args = {"classifier": "neural", "bucket_from": final_testimages_bucket, "model_bucket": "model-train", "model_bucket_name": model_bucket_name, "result_bucket": "prediction-labels", "result_name": os.path.join(actual_name, str(image_num)),
            "xscale": x_scale, "yscale": y_scale, "label_style": event["label_style"], "num_classes": event["num_classes"], "num_channels": event["num_channels"]}
        if "labels_key_name" in event:
            args["labels_key_name"] = event["labels_key_name"]

        # this would be the case if it's "training" but we are doing dev/test evaluation
        if label_dict_type in TEST_TYPES:
            args["final_labels_bucket"] = final_labels_bucket
        invoke_response = lambda_client.invoke(FunctionName="deep-seg-predict", InvocationType='Event', Payload=json.dumps(args))
    else:
        sqs = boto3.client('sqs')
        # because we use send_message_batch, we will send 10 messages at a time, each message contains
        num_loops = int(round(num_patches / (10.0*batch_size)))
        for i in range(num_loops):
            num = random.randrange(1,3)
            if num == 1:
                queue_url = sqs.get_queue_url(QueueName=queue_name)
            else:
                queue_url = sqs.get_queue_url(QueueName=queue_name1)

            entries = []
            for j in range(10):
                json_msg = json.dumps({"actual_name": str(actual_name), "image_num": str(image_num), "patch_num": str(i*10+j*batch_size), "num_patches": str(num_patches)})
                entry_dict = {'Id': str(j), 'MessageBody': json_msg, 'MessageDeduplicationId': "deduplicationId" + str(image_num) + str(i*10+j*batch_size), 'MessageGroupId': 'groupId'}
                entries.append(entry_dict)
            
            response = sqs.send_message_batch(QueueUrl=queue_url['QueueUrl'], Entries=entries)
            print(response)
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
            if message['Body'] == 'called':
                is_called = message['Body']
        #only invoke neuralnet.py once, by the first preprocessing3 to finish
        if is_called:
            print("Neuralnet invoked from " + str(image_num))
            sqs.delete_message(
                QueueUrl=sqs.get_queue_url(QueueName='called-' + queue_name)['QueueUrl'],
                ReceiptHandle=receipt_handle
            )

            args = {"bucket_from": final_trainimages_bucket, "bucket_from_labels" : final_labels_bucket, "model_bucket_name": model_bucket_name, 
                    "image_num": str(image_num), "xscale": x_scale, "yscale": y_scale, "queue_name": queue_name, "queue_name1": queue_name1, 
                    "num_classes": event["num_classes"], "num_machines": event["num_machines"], "called_from": "pre3", "num_channels": event["num_channels"],
                    "num_epochs": event["num_epochs"], "epoch": 1}
            invoke_response = lambda_client.invoke(FunctionName="deep-seg-averager", InvocationType='Event', Payload=json.dumps(args))
            print(invoke_response)

    return 0