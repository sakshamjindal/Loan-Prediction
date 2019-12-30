import scikitplot as skplt
from sklearn.metrics import matthews_corrcoef, f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

def evaluate_model_preds(y_true, y_pred, probas=None):
    
    # F1-Score
    f1 = f1_score(y_true, y_pred, pos_label=1, average='binary')
    print("F1: ", f1)
    
    # Confusion Matrix
    plt.figure(figsize=(20,20))
    skplt.metrics.plot_confusion_matrix(y_true=y_true, y_pred=y_pred,normalize=True)
    plt.show()
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy: ", accuracy)
    
    # Precision
    precision = precision_score(y_true, y_pred)
    print("Precision: ", accuracy)    
        
    # Recall
    recall = recall_score(y_true, y_pred)
    print("Recall: ", accuracy)    
    
    # Precision Recall Curve
    if probas is not None:
        skplt.metrics.plot_precision_recall(y_true=y_true, y_probas=probas)
        plt.show()