import shap 
import numpy as np 

def explain_logistic_regression(model, x_sample, feature_names):
    """
    Explaines predictions of the linear-model using SHAP.
    """
    explainer = shap.LinearExplainer(model , x_sample, feature_names=feature_names)

    shap_values = explainer(x_sample)

    return shap_values