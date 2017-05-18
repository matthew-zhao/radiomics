import numpy as np
import boto
from boto.s3.key import Key

# Function to be called by lambda2 when all images are done being preprocessed
# to squish them together
def squish(event, context):
    conn = boto.connect_s3("AKIAIMQLHJNMP6DOUM4A","8dJAfPZlTjMR1SOcOetImclAmT+G02VkQiuHefdY")
    b = conn.get_bucket(event['bucket_from'])
    labels = conn.get_bucket(event['bucket_from_labels'])
    bucket_list = b.list()
    arr_list = []
    labels_list = []

    # Go through each individual array in the list
    for l in bucket_list:
        # Save content of file into tempfile on lambda
        # TODO: this step may cause issues b/c of .npy data lost?
        if l.key[-4:] == ".npy":
        #if l.key == "matrix10_left.npy":
            l.get_contents_to_filename("/tmp/" + str(l.key))
            label_matrix = labels.get_key(str(l.key))
            label_matrix.get_contents_to_filename("/tmp/labels-" + str(l.key))

            # Load the numpy array from the tempfile and add it to list of np arrays
            #training_arr = None
            #label_arr = None
            #training_arr = np.load("/tmp/" + str(l.key))
            #label_arr = np.load("/tmp/labels-" + str(l.key))
            with open("/tmp/" + str(l.key), "rb") as npy:
                training_arr = np.load(npy)
            with open("/tmp/labels-" + str(l.key), "rb") as npy_label:
                label_arr = np.load(npy_label)
            arr_list.append(training_arr)
            labels_list.append(label_arr)

    # Concatenate all numpy arrays representing a single image together
    concat = np.concatenate(arr_list, axis=1)

    # Concatenate all label arrays representing a single image together in the same
    # order that the images were concatenated
    concat_labels = np.concatenate(labels_list)

    # Create new buckets for the array and its corresponding labels
    b2 = conn.get_bucket('training-arrayfinal')
    labels2 = conn.get_bucket('training-labelsfinal')
    k = b2.new_key('ready_matrix.npy')
    k2 = labels2.new_key('ready_labels.npy')

    # Save the numpy arrays to temp .npy files on lambda
    upload_path = '/tmp/resized-matrix.npy'
    upload_path_labels = '/tmp/resized-labels.npy'
    np.save(upload_path, concat)
    np.save(upload_path_labels, concat_labels)

    # Take the tempfile that has the concated final array and set the contents
    # of the new bucket's key
    k.set_contents_from_filename(upload_path)
    k2.set_contents_from_filename(upload_path_labels)

    # make the file public so it can be accessed publicly
    # using a URL like http://s3.amazonaws.com/bucket_name/key
    k.make_public()
    k2.make_public()
    return 0