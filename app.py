from flask import Flask, request, jsonify, g
import os
from flask_cors import CORS
from tensorflow.keras.models import load_model
import pickle
# import numpy as np
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)
CORS(app)




class Singleton:
    _instances = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[cls] = instance
        return cls._instances[cls]

class ModelLoader(Singleton):
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(app.root_path, 'static/models/AMORA MODEL (BERT)/AMORA_MODEL'))
            self.bert_tokenizer = AutoTokenizer.from_pretrained(os.path.join(app.root_path, 'static/models/AMORA MODEL (BERT)/AMORA_MODEL_TOKENIZER'))
            self.initialized = True

            
            
            
@app.before_request
def load_models():
    if 'model_loader' not in g:
        g.model_loader = ModelLoader()
    
    g.bert_model = g.model_loader.bert_model
    g.bert_tokenizer = g.model_loader.bert_tokenizer
  



@app.route('/ml', methods=['GET'])
def ml():
    input_text = request.args.get('text')
    if not input_text:
        return "No text provided", 400
    
    with open('static/models/ml_prediction_pipeline.pkl', 'rb') as file:
        prediction_pipeline = pickle.load(file)

    sample_data = [input_text]

    # y_pred = prediction_pipeline.predict(sample_data)

    if hasattr(prediction_pipeline.named_steps['classifier'], 'predict_proba'):
        confidence_levels = prediction_pipeline.predict_proba(sample_data)
        confidence = confidence_levels[0][1]  
    else:
        confidence = 1.0

    threshold = 0.40

    if confidence >= threshold:
        p_class = 'OPORTUNIDADE'
    else:
        p_class = 'NAO-OPORTUNIDADE'
    
    confidence_percentage = confidence * 100
    response = jsonify({'Class': p_class, 'Confidence': f'{confidence_percentage:.2f}'})
        
    return response



@app.route('/ann', methods=['GET'])
def ann():
    input_text = request.args.get('text')
    if not input_text:
        return "No text provided", 400
    
    with open('static/models/vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    
    loaded_model=load_model('static/models/ann_classifier.h5')
    
    sample_data = [input_text]
    vect_text=vectorizer.transform(sample_data)


    y_pred = loaded_model.predict(vect_text)

    confidence_percentage = y_pred[0][0] * 100

    if y_pred[0][0] >= 0.5:
        p_class = 'OPORTUNIDADE'
    else:
        p_class = 'NAO-OPORTUNIDADE'
    
    response = jsonify({'Class': p_class, 'Confidence': f'{confidence_percentage:.2f}'})
    return response



def preprocess_text(text, tokenizer):
    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors="pt")
    return inputs


def predict(text, model, tokenizer):
    inputs = preprocess_text(text, tokenizer)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence


def display_prediction(text, model, tokenizer):
    predicted_class, confidence = predict(text,model, tokenizer)
    
    if predicted_class == 1:
        output = 'OPORTUNIDADE'
    elif predicted_class == 0:
        output = 'NAO-OPORTUNIDADE'
    else:
        pass
    
    return output, confidence



@app.route('/updated_model', methods=['GET'])
def updated_model():
    input_text = request.args.get('text')
    if not input_text:
        return "No text provided", 400
    
    model = g.bert_model
    tokenizer = g.bert_tokenizer
     
    output, confidence = display_prediction(input_text, model, tokenizer)
    
    confidence = float(confidence) * 100
    formatted_confidence = f'{confidence:.2f}'
        
    response = jsonify({'Class': output, 'Confidence': formatted_confidence})
    
    return response


model_loader = ModelLoader()

def warm_up_model():
    dummy_text = "This is a warm-up call to load the model."
    model = model_loader.bert_model
    tokenizer = model_loader.bert_tokenizer
    _ = display_prediction(dummy_text, model, tokenizer)
    print("Model warm-up completed.")



if __name__ == "__main__":
    warm_up_model()  
    try:
        app.run(debug=True, host='0.0.0.0', threaded=True)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

