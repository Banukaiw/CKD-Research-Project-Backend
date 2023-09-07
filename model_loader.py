import pickle

def load_model():
    with open('rf_clf_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

def DecisionTree_model():
    with open('DTC_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model    

def Nephrotic_model():
    with open('Random Forest_Nephrotic_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model   

def CKD_model():
    with open('SVM_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

def AKI_model():
    with open('Random Forest_Aki_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model
