import speech_recognition as sr
import requests
from voiceassistant import get_audio

# Initialize the recognizer
r = sr.Recognizer()

# Record audio from the microphone
with sr.Microphone() as source:
    print("Speak now")
    city = get_audio()

api_url = 'https://api.api-ninjas.com/v1/weather?city={}'.format(city)
response = requests.get(api_url, headers={'X-Api-Key': 'YOUR_API_KEY'})
if response.status_code == requests.codes.ok:
    print(response.text)
else:
    print("Error:", response.status_code, response.text)


