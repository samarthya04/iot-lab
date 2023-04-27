import json
import requests
from bs4 import BeautifulSoup
import pyttsx3
import speech_recognition as sr
import datetime
x = requests.get('https://api.iotkiit.in/items/light')
y = requests.get('https://api.iotkiit.in/items/weather')

#print the response text (lights on and off):
print(x.text)

#print the response text( weather):
print(y.text)

import openai
openai.api_key = "YOUR-KEY-HERE"

def speak(data):
    engine = pyttsx3.init()
    engine.say(data)
    engine.runAndWait()


def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak:")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
        # said = ""
        said = r.recognize_google(audio)

        try:
            print("You said: \n" + said)
        except sr.UnknownValueError:
            print("Sorry, could not understand your speech.")
        except sr.RequestError as e:
            print("Request error; {0}".format(e))

    return said.lower()


def get_audio():
    engine = pyttsx3.init()
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("Speak:")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
        voice_command = r.recognize_google(audio)
    try:
        print(f"User said: {voice_command}")
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
        engine.say("Sorry, I did not understand that.")
        engine.runAndWait()
        exit()
    except sr.RequestError:
        print("Sorry, I could not process your request.")
        engine.say("Sorry, I could not process your request.")
        engine.runAndWait()
    return voice_command


def query(user_query):
    url = "https://www.google.co.in/search?q=" + user_query
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'
    }
    page = requests.get(url, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    results = soup.find(class_='Z0LcW t2b5Cf').get_text()
    print(results)
    return results

def weather():
    city = get_audio()
    api_url = 'https://api.iotkiit.in/items/weather'.format(city)
    response = requests.get(api_url, headers={'X-Api-Key': 'YOUR_API_KEY'})
    if response.status_code == requests.codes.ok:
        print(response.text)
    else:
        print("Error:", response.status_code, response.text)

def chatgpt(user_query):
    response = openai.Completion.create(engine='text-davinci-003',
                                        prompt=user_query,
                                        n=1,
                                        temperature=0.5,
                                        max_tokens=50,
                                        top_p=1)
    return response['choices'][0]['text']


WAKE = "hello"
speak("Active")

while True:
    flag = 0
    url = 'https://api.iotkiit.in/items/light'
    headers = {
        "Content-Type": "application/json"
    }
    on_payload = json.dumps({
        'request': 1
    })
    off_payload = json.dumps({
        'request': 0
    })
    print("Active")
    # speak("Active")
    texts = get_audio()

    if texts.count(WAKE) > 0:
        print("I am listening")
        speak("I am listening")
        try:
            text = get_audio()
            # result = query(text)

            # on = ["turn on the lights", "switch on the lights", "lights on"]
            # for phrase in on:
            if 'switch on' in text:
                flag = 1
                speak("Turning on the lights")
                # do something
                response = requests.request("PATCH", url, headers=headers, data=on_payload)
                print(response.text)
            # off = ["Turn off the lights", "Switch off the lights", "Lights off"]
            # for phrase in off:
            if 'switch off' in text:
                flag = 1
                speak("Turning off the lights")
                print()
                # do something
                response = requests.request("PATCH", url, headers=headers, data=off_payload)
                print(response.text)

            # time = ["What is the time"]
            # for phrase in time:
            if 'time now' in text:
                flag = 1
                now = datetime.datetime.now()
                time_string = now.strftime("The time is %I:%M %p.")
                speak(time_string)
                print(time_string)

            # date = ["What is the date"]
            # for phrase in time:
            if 'date today' in text:
                flag = 1
                now = datetime.datetime.now()
                date_string = now.strftime("Today's date is %B %d, %Y.")
                speak(date_string)
                print(date_string)

            # if flag != 1:
            #     result = chatgpt(text)
            #     print(result)
            #     speak(result)
        except sr.UnknownValueError:
            # print('Sorry no result')
            speak('Sorry no result')
    if 'iot' in texts:
        speak('Terminating')
        print('Terminating')
        exit()
