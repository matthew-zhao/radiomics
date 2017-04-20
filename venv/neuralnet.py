from sklearn.neural_network import MLPClassifier
import pickle
import gspread
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials

def classify(event, context):
    key = event['key']
    scopes = ['https://spreadsheets.google.com/feeds']
    credentials = ServiceAccountCredentials.from_json_keyfile_name('Unknown.json', scopes=scopes)
    gc = gspread.authorize(credentials)
    worksheet = gc.open_by_key(key).sheet1
    arr = np.array(worksheet.get_all_values())
    X = arr[1:, :-1]
    y = arr[1:, -1]

    X_converted = X.astype(np.float)
    y_converted = y.astype(np.float)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 2), random_state=1)

    clf.fit(X_converted, y_converted)

    s = pickle.dumps(clf)

    worksheet2 = gc.open_by_key(key).get_worksheet(1)
    worksheet2.update_acell('A1', s)
    return 1

