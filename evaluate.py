from sklearn.metrics import confusion_matrix, classification_report

def evaluate(y_true, y_pred):
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))
