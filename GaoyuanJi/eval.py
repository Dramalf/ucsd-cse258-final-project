from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(groundtruth, prediction, task, metrics=None):
    """
    evaluate performance of a model by various metrics
    
    Args:
        groundtruth : True value
        prediction : predicted value
        task : Type of task ("classification" / "regression")
        metrics : metrics list
    
    
    """
    results = {}
    
    
    
    if task == 'classification':
        for metric in metrics:
            if metric == 'accuracy':
                results['accuracy'] = accuracy_score(groundtruth, prediction)
            elif metric == 'precision':
                results['precision'] = precision_score(groundtruth, prediction, average='weighted')
            elif metric == 'recall':
                results['recall'] = recall_score(groundtruth, prediction, average='weighted')
            elif metric == 'f1':
                results['f1'] = f1_score(groundtruth, prediction, average='weighted')
            
    
    
    elif task == 'regression':
        for metric in metrics:
            if metric == 'mse':
                results['mse'] = mean_squared_error(groundtruth, prediction)
            elif metric == 'mae':
                results['mae'] = mean_absolute_error(groundtruth, prediction)
            elif metric == 'r2':
                results['r2'] = r2_score(groundtruth, prediction)
            
    
    return results