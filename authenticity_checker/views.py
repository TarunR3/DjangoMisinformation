from django.shortcuts import render, redirect
from .apps import AuthenticityCheckerConfig 

import numpy as np
import json
import requests
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .utils import AdvancedTextPreprocessor 
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from bs4 import BeautifulSoup
from django.contrib import messages

import os

from django.conf import settings
tokenizer_path = os.path.join(settings.BASE_DIR, 'resources', 'tokenizers', 'tokenizer.json')

with open(tokenizer_path) as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

preprocessor = AdvancedTextPreprocessor()

def home(request):
    example_articles = [
        {
            'title': 'Takeaways from Joe Biden’s State of the Union address - True',
            'url': 'https://www.cnn.com/2024/03/07/politics/takeaways-joe-biden-state-of-the-union-address/index.html',
            'image': 'https://media.cnn.com/api/v1/images/stellar/prod/240307231204-31-week-in-photos-030724.jpg',
        },
        {
            'title': 'Balloon boy dad suspects Clinton is a reptile - False',
            'url': 'https://foreignpolicy.com/2009/10/19/balloon-boy-dad-suspects-clinton-is-a-reptile/',
            'image': 'https://compote.slate.com/images/3ef6de7f-7114-4fe3-874f-bd898864af9e.jpeg?crop=1560%2C1040%2Cx0%2Cy0',
        },
        {
            'title': '"They\'re all high": Rats eat marijuana from police evidence room - True',
            'url': 'https://news.sky.com/story/theyre-all-high-rats-eat-marijuana-from-police-evidence-room-13094289',
            'image': 'https://e3.365dm.com/24/03/1600x900/skynews-rat-stock_6489728.jpg?20240314072043',
        },
        {
            'title': '"When will interest rates come down? Federal Reserve expected to offer clues Wednesday - True',
            'url': 'https://www.nbcnews.com/business/economy/interest-rates-when-will-they-come-down-march-2024-federal-reserve-rcna144066',
            'image': 'https://media-cldnry.s-nbcnews.com/image/upload/t_nbcnews-fp-1200-630,f_auto,q_auto:best/rockcms/2024-03/240306-jerome-powell-al-0939-b04de7.jpg',
        },
        {
            'title': '"Biden is giving Intel $8.5 billion for big semiconductor projects in 4 states - True',
            'url': 'https://www.npr.org/2024/03/20/1239533039/biden-chips-arizona-intel',
            'image': 'https://media.npr.org/assets/img/2024/03/19/gettyimages-1231364882_wide-6c68fca2f0782ea0ad1f1261d4a187983629d7d4-s1400-c100.jpg',
        },
        {
            'title': '"Abortion rights supporters score midterm victories in at least 4 states - True',
            'url': 'https://www.cbsnews.com/live-updates/abortion-2022-election-results-ballot-initiatives-states-voting/',
            'image': 'https://assets1.cbsnewsstatic.com/hub/i/r/2022/11/08/6f76c6ea-9a09-4618-97c3-07b107f6e6df/thumbnail/1200x630/21f564e0c1a5fc264ea1e4931bcd742e/gettyimages-1244584548.jpg?v=4baa656f7af774a52a8c6a88476cb826',
        },
    ]
    context = {'example_articles': example_articles[:6]}
    return render(request, 'authenticity_checker/index.html', context)

def your_prediction_view(request):
    if request.method == 'POST':
        messages.success(request, 'Thank you for your feedback!')

def predict(request):
    context = {}
    
    if request.method == 'POST':
        url = request.POST.get('url', '').strip()
        article_text = request.POST.get('text', '').strip()

        if url and not url.startswith(('http://', 'https://')):
            url = 'http://' + url

        if url:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')

                    title = soup.find('title').text
                    if title:
                        context['title'] = title
                    else:
                        context['title'] = 'Article Title Not Found'
                    thumbnail_url = soup.find('meta', property='og:image')
                    print(thumbnail_url)
                    if thumbnail_url:
                        context['thumbnail_url'] = thumbnail_url['content']
                    else:
                        context['thumbnail_url'] = None

                    article_text = soup.get_text()
                else:
                    context['error'] = 'Error fetching article content. Try Manually entering the text'
                    return render(request, 'authenticity_checker/index.html', context)
            except requests.exceptions.RequestException as e:
                context['error'] = str(e)
                return render(request, 'authenticity_checker/index.html', context)
        elif article_text:
            context['title'] = 'Provided Article Text'
            context['thumbnail_url'] = None
        else:
            context['error'] = 'Please provide an article URL or text.'
            return render(request, 'authenticity_checker/index.html', {
                'error': 'Please provide an article URL or text.'
            })
        
        processed_text = preprocessor.remove_noise(article_text)
        print(processed_text)
        
        sequence = tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, maxlen=256, padding='post')
        processed_article = np.array(padded_sequence)
        
        prediction_logits = AuthenticityCheckerConfig.model.predict(processed_article)
        prediction_probs = tf.sigmoid(prediction_logits).numpy()

        print(prediction_logits, prediction_probs)

        predicted_class = 'Real' if prediction_probs[0][0] >= 0.5 else 'Fake'
        context['prediction'] = predicted_class

        return render(request, 'authenticity_checker/result.html', context)
    else:
        return render(request, 'authenticity_checker/index.html')