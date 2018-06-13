import boto
import csv
from sklearn.metrics import confusion_matrix

def analyze_results(event, context):
	# we are looking for precision, recall, accuracy, f-scores, confusion matrices
	# for segmentation, dice index, jaccard similarity
	conn = boto.connect_s3("AKIAJRKPLMXU3JRGWYCA","LFfFCpaEsdCCz4KiEBKoTFS5ehXIcPjsPq3yqxjj")
	predictions_bucket = conn.get_bucket(event['result_bucket'])
	bucket_list = predictions_bucket.list()

	b = conn.get_bucket(event["images_bucket"])
	csv_key = b.get_key('trainLabels.csv')
	csv_key.get_contents_to_filename("/tmp/trainLabels.csv")

    c_reader = csv.reader(open('/tmp/trainLabels.csv', 'r'), delimiter = ',')
    columns = list(zip(*c_reader))
    images = columns[0][1:]
    levels = list(map(int, columns[1][1:]))
    label_dict = dict(zip(images, levels))

    correct_count = 0
    total_count = 0
    y_actual = []
    y_pred = []
	for prediction in bucket_list:
		prediction_keyfile = predictions_bucket.get_key(prediction.key)
		classified = prediction_keyfile.get_contents_as_string()

		# if can be converted to float, do so, this way, we can 
		# establish equivalence between any values (int vs float)
		try:
			classified = float(classified)
		except ValueError:
			continue

		# we assume that label_dict has the correct type for its values
		if classified == label_dict[prediction.key]:
			correct_count += 1
		total_count += 1

		# add to list for confusion matrix
		y_actual.append(label_dict[prediction.key])
		y_pred.append(classified)

	accuracy = correct_count / total_count
	conf_matrix = confusion_matrix(y_actual, y_pred)

	return accuracy, conf_matrix