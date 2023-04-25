import pyttsx3

text_speech = pyttsx3.init()

answer = input("Text: ")

# voices = text_speech.getProperty('voices')
# text_speech.setProperty('voice', 2)

text_speech.setProperty('rate', 150)
text_speech.setProperty('volume', 1)

text_speech.say(answer)

text_speech.runAndWait()