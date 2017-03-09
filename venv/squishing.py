import numpy as np
import boto
from boto.s3.key import Key

def squish(event, context):
	conn = boto.connect_s3()
	b = conn.get_bucket('training-arrayresized')
	k = b.new_key('matrix.npy')

	bucket_list = b.list()
	arr_list = []
	for l in bucket_list:
		k.get_contents_to_filename("/tmp/" + str(l.key))
		arr_list.append(np.load("/tmp/" + str(l.key)))

	concat = np.concatenate(arr_list, axis=1)

	upload_path = '/tmp/resized-matrix.npy'
	np.save(upload_path, concat)

	k.set_contents_from_filename(upload_path)
	return 0