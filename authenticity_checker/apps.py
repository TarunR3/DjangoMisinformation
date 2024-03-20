from django.apps import AppConfig
import os
from tensorflow.keras.models import load_model
from django.conf import settings

class AuthenticityCheckerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'authenticity_checker'
    model_path = os.path.join(settings.BASE_DIR, 'resources', 'models', 'my_lstm_model')

    # Load the model and tokenizer
    model = load_model(model_path, compile=False)