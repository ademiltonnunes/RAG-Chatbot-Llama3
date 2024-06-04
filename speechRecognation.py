import os
from pydub import AudioSegment
import io
import subprocess

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
os.environ['GOOGLE_APPLICATION_CREDENTIALS']

from google.cloud import speech_v1p1beta1 as speech
from gtts import gTTS

client = speech.SpeechClient()

import subprocess

class SpeechRecognation:
    def transcribe_audio(self, audio_path):
        try:
            # Convert MP3 to WAV using ffmpeg
            wav_path = audio_path.replace(".mp3", ".wav")
            # subprocess.run(['ffmpeg', '-i', audio_path, '-ac', '1', '-ar', '16000', wav_path])
            subprocess.run(['ffmpeg', '-y', '-i', audio_path, '-ac', '1', '-ar', '16000', wav_path])


            # Load the WAV file
            with open(wav_path, "rb") as wav_file:
                content = wav_file.read()

            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US"
            )

            response = client.recognize(config=config, audio=audio)

            # Extract the transcribed text
            transcribed_text = ""

            if len(response.results) == 0:
                raise Exception("No audio detected")
            
            for result in response.results:
                transcribed_text += result.alternatives[0].transcript

            return transcribed_text

        except Exception as e:            
            e = (f"Error transcribing audio - {e}")
            raise Exception(e)


    def text_to_audio(self, text):
        mp3_obj = gTTS(text=text, lang="en", slow=False)
        return mp3_obj

