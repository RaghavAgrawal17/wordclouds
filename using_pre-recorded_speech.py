import pyaudio
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence
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

def get_large_audio_transcription(path):
    """
    Splitting the large audio file into chunks
    and apply speech recognition on each of these chunks
    """
    # open the audio file using pydub
    sound = AudioSegment.from_wav(path)  
    # split audio sound where silence is 700 miliseconds or more and get chunks
    chunks = split_on_silence(sound,
        # experiment with this value for your target audio file
        min_silence_len = 500,
        # adjust this per requirement
        silence_thresh = sound.dBFS-14,
        # keep the silence for 1 second, adjustable as well
        keep_silence=500,
    )
    folder_name = "audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    # process each chunk 
    for i, audio_chunk in enumerate(chunks, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        # recognize the chunk
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            # try converting it to text
            try:
                text = r.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                print(chunk_filename, ":", text)
                whole_text += text
    # return the text for all chunks detected
    return whole_text

path = "barackobamaindianbusinesssummit_1.wav"
text = get_large_audio_transcription(path)


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
