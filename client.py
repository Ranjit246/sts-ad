import requests

# Server base URL
BASE_URL = "http://127.0.0.0"

# Endpoint for processing audio
ENDPOINT = "/process_audio"

# Complete URL
URL = BASE_URL + ENDPOINT

# Audio file path
FILE_PATH = "test/test_3sec.wav"

if __name__ == "__main__":
    # Open the audio file
    with open(FILE_PATH, "rb") as file:
        # Package the file to send as part of the POST request
        values = {"file": (FILE_PATH, file, "audio/wav")}  # Changed FILE_PATH to a constant string
        
        # Send the POST request
        response = requests.post(URL, files=values, timeout=100)
        
        # Process the response
        if response.status_code == 200:
            data = response.json()
            transcribed_text = data['transcribed_text']
            translated_text = data['translated_text']
            audio_path = data['audio_path']
            print("Transcribed Text:", transcribed_text)
            print("Translated Text:", translated_text)
            print("Audio synthesized at:", audio_path)
        else:
            print(values)
            print(response)
            print("Error Happening:", response.status_code)
