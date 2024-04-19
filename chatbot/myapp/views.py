from django.shortcuts import render
import random
import os
import google.generativeai as genai
from google.cloud import storage
import PIL.Image
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import  load_model
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.layers import Input, Embedding, LSTM , Dense,GlobalAveragePooling1D,Flatten, Dropout , GRU
# from tensorflow.keras.models import Sequential
import pickle
import numpy as np
# loaded_model = load_model('my_model.h5')

# with open('tokenizer.pkl', 'rb') as file:
#     tokenizer = pickle.load(file)

def index(request):
    return render(request,"index.html")
def authenticate_implicit_with_adc(project_id="gen-lang-client-0560619572"):
    storage_client = storage.Client(project=project_id)
    buckets = storage_client.list_buckets()
    print("Buckets:")
    for bucket in buckets:
        print(bucket.name)
    print("Listed all storage buckets.")
API='AIzaSyCu1hSKChXWQ1jCMh7wWiRHwHPsV8W8VDg'

genai.configure(api_key=API)
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.decorators import login_required
# from sklearn.preprocessing import LabelEncoder
import json
import string
# with open("intents.json") as query_dataset:
#   dataset = json.load(query_dataset)
def processing_json_dataset(dataset):
    tags = []
    inputs = []
    responses = {}

    for intent in dataset['intents']:
      # print(intent)
      # print(intent['responses'])
        responses[intent['tag']] = intent['responses']
    # print(responses)
    #     responses[intent[0]['tag']] = intent[0]['responses']
        for pattern in intent['patterns']:
            inputs.append(pattern)
            tags.append(intent['tag'])
    # print(responses)

    return tags, inputs, responses
# processing_json_dataset(dataset)
# Assuming 'dataset' contains the JSON data
# tags, inputs, responses = processing_json_dataset(dataset)
# import pandas as pd
# dataset = pd.DataFrame({"inputs":inputs,
#                      "tags":tags})
# dataset = dataset.sample(frac=1)
# import string
# dataset['inputs'] = dataset['inputs'].apply(lambda sequence:
#                                             [ltrs.lower() for ltrs in sequence if ltrs not in string.punctuation])
# dataset['inputs'] = dataset['inputs'].apply(lambda wrd: ''.join(wrd))
@login_required
def test(request):
    context={}
    if request.method=='POST':
        CH=request.POST.get('input')
        print(CH)
        uploaded_file = request.FILES.get('file')
        if uploaded_file is None:
            model = genai.GenerativeModel('gemini-pro')
            chat = model .start_chat(history=[])
            response = chat.send_message(CH)
            # DEVELOPED MODEL
            texts = []
            pred_input = CH
            pred_input = [letters.lower() for letters in pred_input if letters not in string.punctuation]
            pred_input = ''.join(pred_input)
            texts.append(pred_input)
            # pred_input = tokenizer.texts_to_sequences(texts)
            # pred_input = np.array(pred_input).reshape(-1)
            # pred_input = pad_sequences([pred_input],24)
            # output = loaded_model.predict(pred_input)
            # output = output.argmax()
            # le = LabelEncoder()
            # labels = le.fit_transform(dataset['tags'])
            # response_tag = le.inverse_transform([output])[0]
            # if output > 100:
            context['bot']=response.text
            context['ch']=CH
            # else:
            #         context['bot']=random.choice(responses[response_tag])
            #         context['ch']=CH
            return render(request,"test.html",context)
        else:
            vision_model = genai.GenerativeModel('gemini-pro-vision')
            fl=uploaded_file
            fs = FileSystemStorage()
            filename1 = fs.save(fl.name, fl)
            image = PIL.Image.open(filename1)
            response = vision_model.generate_content([CH,image])
            context['bot']=response.text
            context['ch']=CH
            return render(request,"test.html",context)
    return render(request,"test.html")

from django.urls import reverse_lazy
from django.views.generic import CreateView
from .import forms
# Create your views here.

class SignUp(CreateView):
    form_class=forms.UserCreateForm
    success_url=reverse_lazy('myapp:login')
    template_name='accounts/signup.html'