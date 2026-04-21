import joblib

def load_model(path):
    return joblib.load(path)

def predict(model, input_data):
    return model.predict(input_data)