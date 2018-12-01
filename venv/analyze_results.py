import boto
import csv
from StringIO import StringIO
#from sklearn.metrics import confusion_matrix

from utils import cleanup_s3_bucket, stream_from_s3

def analyze_results(event, context):
    # we are looking for precision, recall, accuracy, f-scores, confusion matrices
    # for segmentation, dice index, jaccard similarity
    conn = boto.connect_s3("AKIAJRKPLMXU3JRGWYCA","LFfFCpaEsdCCz4KiEBKoTFS5ehXIcPjsPq3yqxjj")
    predictions_bucket = conn.get_bucket(event['result_bucket'])
    bucket_list = predictions_bucket.list()

    threshold = None
    if "IOU_threshold" in event:
        threshold = event["IOU_threshold"]

    # images bucket is final_images_bucket
    # labels bucket is final_labels_bucket
    b = conn.get_bucket(event["images_bucket"])
    labels = conn.get_bucket(event["labels_bucket"])

    correct_count = 0
    total_count = 0
    y_actual = []
    y_pred = []
    dice_score_sum = 0
    for prediction in bucket_list:
        # get the name of the corresponding final_labels_bucket key
        final_label_name = prediction.key + "label-processed.npy"
        prediction_keyfile = predictions_bucket.get_key(prediction.key)
        classified = prediction_keyfile.get_contents_as_string()

        if event["label_style"] == "single":
            csv_key = b.get_key(event["labels_key_name"])
            csv_key.get_contents_to_filename("/tmp/trainLabels.csv")

            c_reader = csv.reader(open('/tmp/trainLabels.csv', 'r'), delimiter = ',')
            columns = list(zip(*c_reader))
            images = columns[0][1:]
            levels = list(map(int, columns[1][1:]))
            label_dict = dict(zip(images, levels))
            # if can be converted to float, do so, this way, we can 
            # establish equivalence between any values (int vs float)
            try:
                classified = float(classified)
            except ValueError:
                continue

            # we assume that label_dict has the correct type for its values
            if classified == label_dict[prediction.key]:
                correct_count += 1

            # add to list for confusion matrix
            y_actual.append(label_dict[prediction.key])
            y_pred.append(classified)
        elif event["label_style"] == "array":
            # if it's a numpy array, get it into numpy array format
            try:
                classified_fo = StringIO(classified)
                classified = np.load(classified_fo)
            except ValueError:
                continue

            #final_label_name = label_dict[prediction.key]
            gt = np.load(stream_from_s3(bucket=labels, key_name=final_label_name))

            dice = np.sum(classified[gt==1])*2.0 / (np.sum(classified) + np.sum(gt))
            dice_score_sum += dice

            if threshold and dice > threshold:
                correct_count += 1
        total_count += 1

    accuracy = correct_count / total_count
    #conf_matrix = confusion_matrix(y_actual, y_pred)
    average_dice_score = dice_score_sum / total_count

    cleanup_s3_bucket(b)
    cleanup_s3_bucket(labels)

    return accuracy, average_dice_score