from transformers import pipeline

class emotion_classifier:
    #what each emotion from model maps to in azure tts
    model_emotions = {
        'neutral': 'default',
        'anger': 'angry',
        'annoyance': 'angry',
        'disgust': 'angry',
        'amusement': 'cheerful',
        'gratitude': 'cheerful',
        'joy': 'cheerful',
        'realisation': 'cheerful',
        'admiration': 'excited',
        'desire': 'excited',
        'excitement': 'excited',
        'optimism': 'excited',
        'approval': 'friendly',
        'curiosity': 'friendly',
        'pride': 'friendly',
        'relief': 'friendly',
        'caring': 'hopeful',
        'love': 'hopeful',
        'disappointment': 'sad',
        'grief': 'sad',
        'nervousness': 'sad',
        'remorse': 'sad',
        'sadness': 'sad',
        'embarrassment': 'terrified',
        'fear': 'terrified',
        'confusion': 'unfriendly',
        'disapproval': 'unfriendly',
    }

    def __init__(self):
        self.classifier = pipeline('text-classification', model='arpanghoshal/EmoRoBERTa')

    def map_to_azure_emotion(self, text):
        emotion_labels = self.classifier(text)
        score = emotion_labels[0]['score']
        emotion = emotion_labels[0]['label']
        if score >= 0.8:
            return self.model_emotions[emotion]
        else:
            return 'default'