from sklearn.metrics import confusion_matrix, classification_report 

# generate the classification_report 
def evaluate_classification(y_true, y_pred):

    report = classification_report(y_true, y_pred, 
                                   output_dict=True)
    
    matrix = confusion_matrix(y_true,y_pred)

    return report, matrix 

# created seperate evaluation for same matrix reused across ML / DL / Transformers
# cleaner experiments , prevents rewriting extra coding