from sklearn.neural_network import MLPClassifier
import pickle
import gspread
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials

def predict(event, context):
    key = event['key']
    classifier = event['classifier']
    scopes = ['https://spreadsheets.google.com/feeds']
    credentials = ServiceAccountCredentials.from_json_keyfile_name('Unknown.json', scopes=scopes)
    gc = gspread.authorize(credentials)
    worksheet2 = gc.open_by_key(key).get_worksheet(1)

    if classifier == 'neural':
        val = worksheet2.acell('A1').value
    elif classifier == 'knn':
        val = worksheet2.acell('A4').value
    elif classifier == 'decision_tree':
        val = worksheet2.acell('A6').value
    elif classifier == 'forest':
        val = worksheet2.acell('A7').value
    elif classifier == 'bagging':
        val = worksheet2.acell('A8').value

    clf = pickle.loads(val)

    worksheet3 = gc.open_by_key(key).get_worksheet(2)
    arr = np.array(worksheet3.get_all_values())
    X = arr[1:, :-1]
    X_converted = X.astype(np.float)

    predictions = clf.predict(X_converted)

    return {"predictions": predictions.tolist()} 