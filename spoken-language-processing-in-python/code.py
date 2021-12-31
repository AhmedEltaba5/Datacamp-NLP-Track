import wave

# Create audio file wave object
good_morning = wave.open("good_morning.wav", 'r')

# Read all frames from wave object 
signal_gm = good_morning.readframes(-1)

# View first 10 frames
print(signal_gm[:10])

=======================

import numpy as np

# Open good morning sound wave and read frames as bytes
good_morning = wave.open('good_morning.wav', 'r')
signal_gm = good_morning.readframes(-1)

# Convert good morning audio bytes to integers
soundwave_gm = np.frombuffer(signal_gm, dtype='int16')

# View the first 10 sound wave values
print(soundwave_gm[:10])

=========================

# Read in sound wave and convert from bytes to integers
good_morning = wave.open('good_morning.wav', 'r')
signal_gm = good_morning.readframes(-1)
soundwave_gm = np.frombuffer(signal_gm, dtype='int16')

# Get the sound wave frame rate
framerate_gm = good_morning.getframerate()

# Find the sound wave timestamps
time_gm = np.linspace(start=0,
                      stop=len(soundwave_gm)/framerate_gm,
					  num=len(soundwave_gm))

# Print the first 10 timestamps
print(time_gm[:10])

=======================

# Setup the title and axis titles
plt.title('Good Afternoon vs. Good Morning')
plt.ylabel('Amplitude')
plt.xlabel('Time (seconds)')

# Add the Good Afternoon data to the plot
plt.plot(time_ga, soundwave_ga, label='Good Afternoon')

# Add the Good Morning data to the plot
plt.plot(time_gm, soundwave_gm, label='Good Morning',
   # Set the alpha to 0.5
   alpha=0.5)

plt.legend()
plt.show()

========================

# Importing the speech_recognition library
import speech_recognition as sr

# Create an instance of the Recognizer class
recognizer = sr.Recognizer()

# Set the energy threshold
recognizer.energy_threshold = 300

========================

# Create a recognizer class
recognizer = sr.Recognizer()

# Transcribe the support call audio
text = recognizer.recognize_google(
  audio_data=clean_support_call_audio, 
  language="en-US")

print(text)

========================

# Instantiate Recognizer
recognizer = sr.Recognizer()

# Convert audio to AudioFile
clean_support_call = sr.AudioFile("clean_support_call.wav")

# Convert AudioFile to AudioData
with clean_support_call as source:
    clean_support_call_audio = recognizer.record(source)

# Transcribe AudioData to text
text = recognizer.recognize_google(clean_support_call_audio,
                                   language="en-US")
print(text)

========================

# Convert AudioFile to AudioData
with nothing_at_end as source:
    nothing_at_end_audio = recognizer.record(source,
                                             duration=10,
                                             offset=None)

# Transcribe AudioData to text
text = recognizer.recognize_google(nothing_at_end_audio,
                                   language="en-US")

print(text)

=========================

# Convert AudioFile to AudioData
with static_at_start as source:
    static_art_start_audio = recognizer.record(source,
                                               duration=None,
                                               offset=3)

# Transcribe AudioData to text
text = recognizer.recognize_google(static_art_start_audio,
                                   language="en-US")

print(text)

=======================

# Create a recognizer class
recognizer = sr.Recognizer()

# Pass the Japanese audio to recognize_google
text = recognizer.recognize_google(japanese_audio, language="en-US")

# Print the text
print(text)

# Create a recognizer class
recognizer = sr.Recognizer()

# Pass the Japanese audio to recognize_google
text = recognizer.recognize_google(japanese_audio, language="ja")

# Print the text
print(text)

# Create a recognizer class
recognizer = sr.Recognizer()

# Pass the leopard roar audio to recognize_google
text = recognizer.recognize_google(leopard_audio, 
                                   language="en-US", 
                                   show_all=True)

# Print the text
print(text)

# Create a recognizer class
recognizer = sr.Recognizer()

# Pass charlie_audio to recognize_google
text = recognizer.recognize_google(charlie_audio, 
                                   language="en-US")

# Print the text
print(text)

======================

# Create a recognizer class
recognizer = sr.Recognizer()

# Recognize the multiple speaker AudioData
text = recognizer.recognize_google(multiple_speakers, 
                       			   language="en-US")

# Print the text
print(text)

=======================

recognizer = sr.Recognizer()

# Multiple speakers on different files
speakers = [sr.AudioFile("speaker_0.wav"), 
            sr.AudioFile("speaker_1.wav"), 
            sr.AudioFile("speaker_2.wav")]

# Transcribe each speaker individually
for i, speaker in enumerate(speakers):
    with speaker as source:
        speaker_audio = recognizer.record(source)
    print(f"Text from speaker {i}:")
    print(recognizer.recognize_google(speaker_audio,
         				  language="en-US"))

========================

recognizer = sr.Recognizer()

# Record the audio from the clean support call
with clean_support_call as source:
  clean_support_call_audio = recognizer.record(source)

# Transcribe the speech from the clean support call
text = recognizer.recognize_google(clean_support_call_audio,
					   language="en-US")

print(text)

========================

recognizer = sr.Recognizer()

# Record the audio from the noisy support call
with noisy_support_call as source:
  noisy_support_call_audio = recognizer.record(source)

# Transcribe the speech from the noisy support call
text = recognizer.recognize_google(noisy_support_call_audio,
                         language="en-US",
                         show_all=True)

print(text)

=======================

recognizer = sr.Recognizer()

# Record the audio from the noisy support call
with noisy_support_call as source:
	# Adjust the recognizer energy threshold for ambient noise
    recognizer.adjust_for_ambient_noise(source, duration=1)
    noisy_support_call_audio = recognizer.record(noisy_support_call)
 
# Transcribe the speech from the noisy support call
text = recognizer.recognize_google(noisy_support_call_audio,
                                   language="en-US",
                                   show_all=True)

print(text)

=========================

recognizer = sr.Recognizer()

# Record the audio from the noisy support call
with noisy_support_call as source:
	# Adjust the recognizer energy threshold for ambient noise
    recognizer.adjust_for_ambient_noise(source, duration=0.5)
    noisy_support_call_audio = recognizer.record(noisy_support_call)
 
# Transcribe the speech from the noisy support call
text = recognizer.recognize_google(noisy_support_call_audio,
                                   language="en-US",
                                   show_all=True)

print(text)

========================

# Import AudioSegment from Pydub
from pydub import AudioSegment

# Create an AudioSegment instance
wav_file = AudioSegment.from_file(file="wav_file.wav", 
                                  format="wav")

# Check the type
print(type(wav_file))

========================

# Import AudioSegment and play
from pydub import AudioSegment
from pydub.playback import play

# Create an AudioSegment instance
wav_file = AudioSegment.from_file(file="wav_file.wav", 
                                  format="wav")

# Play the audio file
play(wav_file)

=========================

# Import audio file
wav_file = AudioSegment.from_file(file="wav_file.wav")

# Find the frame rate
print(wav_file.frame_rate)

# Import audio file
wav_file = AudioSegment.from_file(file="wav_file.wav")

# Find the number of channels
print(wav_file.channels)

# Import audio file
wav_file = AudioSegment.from_file(file="wav_file.wav")

# Find the max amplitude
print(wav_file.max)

# Import audio file
wav_file = AudioSegment.from_file(file="wav_file.wav")

# Find the length
print(len(wav_file))

========================

# Import audio file
wav_file = AudioSegment.from_file(file="wav_file.wav")

# Create a new wav file with adjusted frame rate
wav_file_16k = wav_file.set_frame_rate(16000)

# Check the frame rate of the new wav file
print(wav_file_16k.frame_rate)

# Import audio file
wav_file = AudioSegment.from_file(file="wav_file.wav")

# Set number of channels to 1
wav_file_1_ch = wav_file.set_channels(1)

# Check the number of channels
print(wav_file_1_ch.channels)

# Import audio file
wav_file = AudioSegment.from_file(file="wav_file.wav")

# Print sample_width
print(f"Old sample width: {wav_file.sample_width}")

# Set sample_width to 1
wav_file_sw_1 = wav_file.set_sample_width(1)

# Check new sample_width
print(f"New sample width: {wav_file_sw_1.sample_width}")

======================

from pydub import AudioSegment

# Import audio file
volume_adjusted = AudioSegment.from_file("volume_adjusted.wav")

# Lower the volume by 60 dB
quiet_volume_adjusted = volume_adjusted - 60

from pydub import AudioSegment

# Import audio file
volume_adjusted = AudioSegment.from_file("volume_adjusted.wav")

# Increase the volume by 15 dB
louder_volume_adjusted = volume_adjusted + 15

======================

# Import AudioSegment and normalize
from pydub import AudioSegment
from pydub.effects import normalize

# Import target audio file
loud_then_quiet = AudioSegment.from_file("loud_then_quiet.wav")

# Normalize target audio file
normalized_loud_then_quiet = normalize(loud_then_quiet)

=====================

from pydub import AudioSegment

# Import part 1 and part 2 audio files
part_1 = AudioSegment.from_file("part_1.wav")
part_2 = AudioSegment.from_file("part_2.wav")

# Remove the first four seconds of part 1
part_1_removed = part_1[4000:]

# Add the remainder of part 1 and part 2 together
part_3 = part_1_removed + part_2

====================

# Import AudioSegment
from pydub import AudioSegment

# Import stereo audio file and check channels
stereo_phone_call = AudioSegment.from_file("stereo_phone_call.wav")
print(f"Stereo number channels: {stereo_phone_call.channels}")

# Split stereo phone call and check channels
channels = stereo_phone_call.split_to_mono()
print(f"Split number channels: {channels[0].channels}, {channels[1].channels}")

# Save new channels separately
phone_call_channel_1 = channels[0]
phone_call_channel_2 = channels[1]

=====================

from pydub import AudioSegment

# Import the .mp3 file
mp3_file = AudioSegment.from_file("mp3_file.mp3")

# Export the .mp3 file as wav
mp3_file.export(out_f="mp3_file.wav",
                format="wav")

======================

# Loop through the files in the folder
for audio_file in folder:
    
	# Create the new .wav filename
    wav_filename = os.path.splitext(os.path.basename(audio_file))[0] + ".wav"
        
    # Read audio_file and export it in wav format
    AudioSegment.from_file(audio_file).export(out_f=wav_filename, 
                                              format="wav")
        
    print(f"Creating {wav_filename}...")

 =======================

 for audio_file in folder:
    file_with_static = AudioSegment.from_file(audio_file)

    # Cut the 3-seconds of static off
    file_without_static = file_with_static[3000:]

    # Increase the volume by 10dB
    louder_file_without_static = file_without_static + 10
    
    # Create the .wav filename for export
    wav_filename = os.path.splitext(os.path.basename(audio_file))[0] + ".wav"
    
    # Export the louder file without static as .wav
    louder_file_without_static.export(wav_filename, format="wav")
    print(f"Creating {wav_filename}...")

========================

# Create function to convert audio file to wav
def convert_to_wav(filename):
  """Takes an audio file of non .wav format and converts to .wav"""
  # Import audio file
  audio = AudioSegment.from_file(filename)
  
  # Create new filename
  new_filename = filename.split(".")[0] + ".wav"
  
  # Export file as .wav
  audio.export(new_filename, format="wav")
  print(f"Converting {filename} to {new_filename}...")
 
# Test the function
convert_to_wav("call_1.mp3")

========================

def show_pydub_stats(filename):
  """Returns different audio attributes related to an audio file."""
  # Create AudioSegment instance
  audio_segment = AudioSegment.from_file(filename)
  
  # Print audio attributes and return AudioSegment instance
  print(f"Channels: {audio_segment.channels}")
  print(f"Sample width: {audio_segment.sample_width}")
  print(f"Frame rate (sample rate): {audio_segment.frame_rate}")
  print(f"Frame width: {audio_segment.frame_width}")
  print(f"Length (ms): {len(audio_segment)}")
  return audio_segment

# Try the function
call_1_audio_segment = show_pydub_stats("call_1.wav")

========================

def transcribe_audio(filename):
  """Takes a .wav format audio file and transcribes it to text."""
  # Setup a recognizer instance
  recognizer = sr.Recognizer()
  
  # Import the audio file and convert to audio data
  audio_file = sr.AudioFile(filename)
  with audio_file as source:
    audio_data = recognizer.record(source)
  
  # Return the transcribed text
  return recognizer.recognize_google(audio_data)

# Test the function
print(transcribe_audio("call_1.wav"))

========================

# Convert mp3 file to wav
convert_to_wav("call_1.mp3")

# Check the stats of new file
call_1 = show_pydub_stats("call_1.wav")

# Split call_1 to mono
call_1_split = call_1.split_to_mono()

# Export channel 2 (the customer channel)
call_1_split[1].export("call_1_channel_2.wav",
                       format="wav")

# Transcribe the single channel
print(transcribe_audio("call_1_channel_2.wav"))

========================

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create SentimentIntensityAnalyzer instance
sid = SentimentIntensityAnalyzer()

# Let's try it on one of our phone calls
call_2_text = transcribe_audio("call_2.wav")

# Display text and sentiment polarity scores
print(call_2_text)
print(sid.polarity_scores(call_2_text))

=======================

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create SentimentIntensityAnalyzer instance
sid = SentimentIntensityAnalyzer()

# Transcribe customer channel of call 2
call_2_channel_2_text = transcribe_audio("call_2_channel_2.wav")

# Display text and sentiment polarity scores
print(call_2_channel_2_text)
print(sid.polarity_scores(call_2_channel_2_text))

=======================

# Import sent_tokenize from nltk
from nltk import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create SentimentIntensityAnalyzer instance
sid = SentimentIntensityAnalyzer()

# Split call 2 channel 2 into sentences and score each
for sentence in sent_tokenize(call_2_channel_2_text):
    print(sentence)
    print(sid.polarity_scores(sentence))

======================

# Import sent_tokenize from nltk
from nltk import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create SentimentIntensityAnalyzer instance
sid = SentimentIntensityAnalyzer()

# Split channel 2 paid text into sentences and score each
for sentence in sent_tokenize("call_2_channel_2_paid_api_text"):
    print(sentence)
    print(sid.polarity_scores(sentence))

=======================

import spacy

# Transcribe call 4 channel 2
call_4_channel_2_text = transcribe_audio("call_4_channel_2.wav")

# Create a spaCy language model instance
nlp = spacy.load("en_core_web_sm")

# Create a spaCy doc with call 4 channel 2 text
doc = nlp(call_4_channel_2_text)

# Check the type of doc
print(type(doc))

====================

import spacy

# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")

# Create a spaCy doc with call 4 channel 2 text
doc = nlp(call_4_channel_2_text)

# Show tokens in doc
for token in doc:
    print(token.text, token.idx)

===================

import spacy

# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")

# Create a spaCy doc with call 4 channel 2 text
doc = nlp(call_4_channel_2_text)

# Show sentences in doc
for sentence in doc.sents:
    print(sentence)

====================

import spacy

# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")

# Create a spaCy doc with call 4 channel 2 text
doc = nlp(call_4_channel_2_text)

# Show named entities and their labels
for entity in doc.ents:
    print(entity.text, entity.label_)

===================

# Import EntityRuler class
from spacy.pipeline import EntityRuler

# Create EntityRuler instance
ruler = EntityRuler(nlp)

# Define pattern for new entity
ruler.add_patterns([{"label": "PRODUCT", "pattern": "smartphone"}])

# Update existing pipeline
nlp.add_pipe(ruler, before="ner")

# Test new entity
for entity in doc.ents:
  print(entity.text, entity.label_)

==================

# Convert post purchase
for file in post_purchase:
    print(f"Converting {file} to .wav...")
    convert_to_wav(file)

# Convert pre purchase
for file in pre_purchase:
    print(f"Converting {file} to .wav...")
    convert_to_wav(file)

===================

def create_text_list(folder):
  # Create empty list
  text_list = []
  
  # Go through each file
  for file in folder:
    # Make sure the file is .wav
    if file.endswith(".wav"):
      print(f"Transcribing file: {file}...")
      
      # Transcribe audio and append text to list
      text_list.append(transcribe_audio(file))   
  return text_list

create_text_list(folder)

====================

# Transcribe post and pre purchase text
post_purchase_text = create_text_list(post_purchase_wav_files)
pre_purchase_text = create_text_list(pre_purchase_wav_files)

# Inspect the first transcription of post purchase
print(post_purchase_text[0])

=====================

import pandas as pd

# Make dataframes with the text
post_purchase_df = pd.DataFrame({"label": "post_purchase",
                                 "text": post_purchase_text})
pre_purchase_df = pd.DataFrame({"label": "pre_purchase",
                                "text": pre_purchase_text})

# Combine DataFrames
df = pd.concat([post_purchase_df, pre_purchase_df])

# Print the combined DataFrame
print(df.head())

=====================

# Build the text_classifier as an sklearn pipeline
text_classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB()),
])

# Fit the classifier pipeline on the training data
text_classifier.fit(train_df.text, train_df.label)

======================

# Build the text_classifier as an sklearn pipeline
text_classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB()),
])

# Fit the classifier pipeline on the training data
text_classifier.fit(train_df.text, train_df.label)

# Evaluate the MultinomialNB model
predicted = text_classifier.predict(test_df.text)
accuracy = 100 * np.mean(predicted == test_df.label)
print(f'The model is {accuracy}% accurate')












































