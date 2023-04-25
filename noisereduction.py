from scipy.io import wavfile
import noisereduce as nr
import speech_recognition as sr

# Initialize the recognizer
r = sr.Recognizer()

# Record audio from the microphone
with sr.Microphone() as source:
    print("Speak now")
    audio = r.record(source, duration=5)

# Save the audio as a .wav file
with open("output.wav", "wb") as f:
    f.write(audio.get_wav_data())
# load data
rate, data = wavfile.read("output.wav")
# perform noise reduction
reduced_noise = nr.reduce_noise(y=data, sr=rate)
wavfile.write("output_reduced_noise.wav", rate, reduced_noise)
