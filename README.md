# Wikipedia Qusetion Answering using transformers

## Preamble

The goal of our project was to create a question answering algorithm, that wouldn't use any search engines or paid databases.

## Architecure
The following flowchart represents the whole pipeline:

![](https://user-images.githubusercontent.com/69817199/137589158-b0585ed2-728a-4408-aedf-af84b3607c33.jpg)


We created a pipeline, that receives either voice message or text messsage with question.

IF WE RECEIVE VOICE MESSAGE:
We are first processing the `.ogg` file using `AudioSegment`.

Afterwards we split everything into chunks to make recognition more simple.

Then using `sr.Recognizer` the audio is being recognized by google api and its question is being received.

(By the way, the following function also creates `.wav` file and dumps it into given directory.)

```python
import speech_recognition as sr 
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence

def get_large_audio_transcription(path):
    r = sr.Recognizer()
    """
    Splitting the large audio file into chunks
    and apply speech recognition on each of these chunks
    """
    # open the audio file using pydub
    sound = AudioSegment.from_ogg(path)  
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
        # the folder_name directory.
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
````
Afterwards the keywords are being extracted from our question using `KeyBert` module.

