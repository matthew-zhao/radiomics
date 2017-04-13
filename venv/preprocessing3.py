import numpy as np
import boto
from boto.s3.key import Key

# Function to take 
def squish(event, context):
    conn = boto.connect_s3()
    b = conn.get_bucket(event['bucket_from'])

    # Download
    bucket_list = b.list()
    arr_list = []

    # Go through each individual array in the list
    for l in bucket_list:
        # Save content of file into tempfile on lambda
        # TODO: this step may cause issues b/c of .npy data lost?
        l.get_contents_to_filename("/tmp/" + str(l.key))

        # Load the numpy array from the tempfile and add it to list of np arrays
        arr_list.append(np.load("/tmp/" + str(l.key)))

    # Concatenate all numpy arrays representing a single image together
    concat = np.concatenate(arr_list, axis=1)

    # Upload to file in new bucket
    b2 = conn.get_bucket('training-arrayfinal')
    k = b2.new_key('ready_matrix.npy')

    upload_path = '/tmp/resized-matrix.npy'
    np.save(upload_path, concat)

    # Take the tempfile that has the concated final array and set the contents
    # of the new bucket's key
    k.set_contents_from_filename(upload_path)

    # make the file public so it can be accessed publicly
    # using a URL like http://s3.amazonaws.com/bucket_name/key
    k.make_public()
    return 0