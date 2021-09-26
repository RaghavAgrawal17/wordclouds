import pyaudio
import speech_recognition as sr
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import regex as re
from nltk.corpus import stopwords
import nltk
nltk.download("punkt")
from nltk.tokenize import word_tokenize

# initialize the recognizer
r = sr.Recognizer()
text = ""


with sr.Microphone() as source:
    # read the audio data from the default microphone
    audio_data = r.record(source, duration=10)
    print("Recognizing...")
    # convert speech to text
    text = r.recognize_google(audio_data)
    
    
#Converting to lower case
text = text.lower()

#Removing Numbers
text = re.sub(" \d+", " ", text)

#Removing common stopwords
text_tokens = word_tokenize(text)
tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
text = (" ").join(tokens_without_sw)

#Removing punctuation
text_tokens = word_tokenize(text)
new_words= [word for word in text_tokens if word.isalnum()]
text = (" ").join(new_words)

#Removing extra white spaces
text = re.sub(' +', ' ', text)

wc = WordCloud(background_color = 'white', width = 1920, height = 1080)
wc.generate_from_text(text)

# plot the WordCloud image                      
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wc)
plt.axis("off")
plt.tight_layout(pad = 0)
 
plt.show()
