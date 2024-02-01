import discord 
import openai
from openai import OpenAI
import csv # CSV file I/O (e.g., writing to a CSV file)
import re # Regular expression library
import requests # For API requests
import json # For parsing JSON
from llama_cpp import Llama
import asyncio # For asynchronous operations
import time
import io # For file operations
from google.oauth2 import service_account # For Google Cloud authentication
from google.cloud import speech # For Google Cloud STT
import os # For file operations
import sys
from google.cloud import texttospeech_v1 # For Google Cloud TTS
from datetime import datetime # For timestamps
from object_detection_yolo import OD # For object detection
from dense_image_captioning import img_cap # For image captioning
from OCR import ocr_with_easyocr # For OCR
from PIL import Image # For image operations
import random
import pytz
from datetime import datetime
import urllib.request

korea_tz = pytz.timezone('Asia/Seoul')
korea_time = datetime.now(korea_tz)

intents = discord.Intents.default() # Enable default intents
intents.message_content = True # Enable message content

# Set the path to your service account key
SERVER_ID =  # Discord server ID
CHANNEL_ID =  # Discord channel ID
papago_client_id =  # 개발자센터에서 발급받은 Client ID 값
papago_client_secret =  # 개발자센터에서 발급받은 Client Secret 값
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "TextToSpeechKey.json" # Google Cloud service account key
client_file = "SpeechToTextKey.json" # Google Cloud service account key
credentials = service_account.Credentials.from_service_account_file(client_file) # Google Cloud service account credentials
stt_client = speech.SpeechClient(credentials=credentials) # Google Cloud STT client
token =  # Discord bot token
openai.api_key =  # OpenAI API key
client = discord.Client(intents=intents) # Create a new client instance (bot)

# Initialize LLAMA model once
model_path = "llama/llama-2-13b-chat/ggml-model-f16.gguf" # LLAMA2 model path
llama_model = Llama(model_path = model_path,
            n_ctx = 4096,            # context window size
            n_gpu_layers = 4,        # enable GPU
            use_mlock = True)        # enable memory lock so not swap

def get_latest_file(path=".", file_type="audio"):
    # Define extensions for audio and image files
    extensions = {
        "audio": ['.mp3', '.ogg'],
        "image": ['.jpg', '.jpeg', '.png']
    }
    chosen_extensions = extensions.get(file_type, [])
    if not chosen_extensions:
        return None  # Return early if no file_type matched
    # List all files in the directory with the desired extensions
    list_of_files = [os.path.join(path, file) for file in os.listdir(path)
                     if os.path.isfile(os.path.join(path, file)) and
                     any(file.lower().endswith(ext) for ext in chosen_extensions)]
    if not list_of_files:  # No files found
        return None
    # Get the latest file
    latest_file = max(list_of_files, key=os.path.getmtime)
    return latest_file

# Define a function to perform speech-to-text
def speech_to_text(file): 
    with io.open(file, 'rb') as f: # Open the audio file in read binary mode
        content = f.read() # Read the contents of the audio file
        audio = speech.RecognitionAudio(content=content) # Create a RecognitionAudio object
    # Create a RecognitionConfig object
    config = speech.RecognitionConfig(
        encoding = speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
        sample_rate_hertz = 48000,
        language_code='ko-KR',
    )
    config2 = speech.RecognitionConfig(
        encoding = speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
        sample_rate_hertz = 48000,
        language_code='en-GB',
    )
    response = stt_client.recognize(config=config, audio=audio) # Perform STT
    res=''
    # Iterate through results
    for result in response.results: 
        res=result.alternatives[0].transcript+res
    return res

# Define a function to perform text-to-speech
def text_to_speech(text, language, Name, gender):
    tts_client = texttospeech_v1.TextToSpeechClient() # Create a TTS client
    # Configure the text input
    input_text = texttospeech_v1.SynthesisInput(text=text) # Set the input text
    # Configure the desired voice
    # Set the voice parameters
    voice = texttospeech_v1.VoiceSelectionParams(
            language_code=language,
            name=Name,
            ssml_gender=texttospeech_v1.SsmlVoiceGender[gender],
        )
    # Configure the audio output format
    audio_config = texttospeech_v1.AudioConfig(
        audio_encoding=texttospeech_v1.AudioEncoding['MP3']
    )
    # Get the synthesized audio
    response = tts_client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )
    # Save the audio to a file
    with open("speech/response.mp3", "wb") as out: 
        out.write(response.audio_content)

# Split the message into chunks of 2000 characters
def split_message(msg, limit=2000):
    return [msg[i:i+limit] for i in range(0, len(msg), limit)]

# Define a function to perform LLAMA
def LLAMA(messages, token, temp):
    response = llama_model(messages, max_tokens=token, temperature=temp)
    return response["choices"][0]["text"]

# Define a function to perform Google search
def google_search(search_term, **kwargs):
    API_KEY = "AIzaSyAKkhPl_12R-LvfO2YAvmG4QkWBMh15PZ4"
    CSE_ID = "57de78b736d694693"
    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': search_term,
        'key': API_KEY,
        'cx': CSE_ID
    }
    for key, value in kwargs.items():
        params[key] = value
    response = requests.get(base_url, params=params)
    results=response.json()
    if 'items' in results:
        res=[]
        for item in results['items']:
            if 'snippet' in item:
                res.append(
                    GPT4([{"role": "user", "content": f"Summarize snippets simply from this list.: {item['snippet']}"}])
                    )
            else:
                res.append((item['title'], item['link']))
    else:
        # Optionally handle the case when there are no search results or an error occurred.
        print("No search results found or an error occurred:", results.get('error', {}).get('message', 'Unknown error'))
        res=[]
    return res
    
def GPT4(messages):
    client = OpenAI(api_key = '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')
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=0.8,
        max_tokens=200
    )
    return response.choices[0].message.content

def GPT4_L(messages):
    client = OpenAI(api_key = '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')
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=0.8,
        max_tokens=400
    )
    return response.choices[0].message.content

def remember(input:list):
    # Open a CSV file for appending in newline mode
    with open("remember.csv", "a", newline="") as f:
        writer = csv.writer(f)
        for input_string in input:
            writer.writerow(input_string.split(":"))
            
def recall():
    # Open the CSV file for reading in newline mode
    with open("remember.csv", "r", newline="") as f:
        reader = csv.reader(f)
        sentence=''
        for row in reader:
            sentence=row[0]+ ' is' + row[1]+' ,'+ sentence
        sentence=sentence[:-1]
        return sentence

def extract_patterns(input_str):
    # The regular expression looks for the patterns (/r xx: xx) or (/r: xx: xx)
    # We are now allowing spaces, underscores, and any Unicode word characters in the "xx" parts.
    pattern = r'\(/r\s?:?\s?[\w_ ]+:\s?[\w_ ]+\)'
    # Extract matches
    matches = re.findall(pattern, input_str)
    # Remove the '(/r' or '(/r:' prefix and ')' suffix from each match and return the cleaned list
    return [match.split(" ", 1)[-1][:-1].strip() for match in matches]

history = dict()

def get_history(user:str)->list:
    if not user in history:
        history['J.A.R.V.I.S#7987'] = [{"role": "assistant", "content": "Your name is JARVIS who is very friendly assistant for user and remember the following elements:" + recall()}]
    return history[user]

def add_history(user:str, prompt: str, answer: str):
    history[user].append({"role": "user", "content": prompt})
    if answer is not None:
        history[user].append({"role": "assistant", "content": answer})
    # Here, if the history exceeds 15 messages, we'll keep the initial message and the last 14.
    if len(history[user]) > 15:
        history[user] = [history[user][0]] + history[user][-14:]

def prompt_to_chat(user: str, prompt: str) -> list:
    previous = get_history(user)
    # Add the current prompt for OpenAI
    openai_prompt = prompt
    messages = previous + [{"role": "user", "content": openai_prompt}]
    # If there are more than 15 messages (5 exchanges), we'll keep the initial message and the last 14. 
    # This ensures that the initial message is always in the conversation context.
    if len(messages) > 15:
        messages = [messages[0]] + messages[-14:]
    return messages

def prompt_to_chat2(user: str, prompt: str) -> str:
    previous = get_history(user)
    messages = previous + [{"role": "user", "content": prompt}]
    # If there are more than 15 messages (5 exchanges), we'll keep the initial message and the last 14.
    if len(messages) > 15:
        messages = [messages[0]] + messages[-14:]
    formatted_chat = ""
    for message in messages:
        if message["role"] == "user":
            formatted_chat += f"user: {message['content']} \n"
        else:  # assistant
            formatted_chat += f"you: {message['content']} \n"
    return formatted_chat.strip()  # strip to remove any trailing newline


def image_generation(prompt):
    client = OpenAI(
        api_key = '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'
    )
    # Ensure the img_gen directory exists
    if not os.path.exists('img_gen'):
        os.makedirs('img_gen')
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            quality="hd",
            size="1024x1024"
        )
        # Download the image
        image_url = response.data[0].url
        image_content = requests.get(image_url).content
        # Save the image to the img_gen folder
        current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        image_path = os.path.join('img_gen', f'generated_image({current_time}).jpg')
        with open(image_path, 'wb') as img_file:
            img_file.write(image_content)

    except Exception as err:
        image_url = "Image generation failed."
        print('Failed:', err)
    return image_url

def generate_random_number(min_value, max_value):
    random_number = random.randint(min_value, max_value)
    return random_number

def lang_trans(input_string, source, target):
    if source == 'auto':
        source = lang_detect(input_string)
    encText = urllib.parse.quote(input_string)
    data = f"source={source}&target={target}&text=" + encText
    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",papago_client_id)
    request.add_header("X-Naver-Client-Secret",papago_client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if(rescode==200):
        res = json.loads(response.read().decode("utf-8"))
    else:
        print("Error Code:" + rescode)
    return res['message']['result']['translatedText']

def lang_detect(input_string):
    encQuery = urllib.parse.quote(input_string)
    data = "query=" + encQuery
    url = "https://openapi.naver.com/v1/papago/detectLangs"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",papago_client_id)
    request.add_header("X-Naver-Client-Secret",papago_client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if(rescode==200):
        res = json.loads(response.read().decode('utf-8'))
    else:
        print("Error Code:" + rescode)
    return res['langCode']

async def send_md_content_to_channel(channel):
    with open("README.md", "r", encoding="utf-8") as file:
        content = file.read()
    await channel.send(content)

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))
    # Replace YOUR_CHANNEL_ID with the ID of the channel where you want to send the message
    channel = client.get_channel(CHANNEL_ID)  
    if channel:  # Check if the channel exists and the bot has access to it
        print("Hello, I am JARVIS, I am ready to assist you!")
        #await channel.send("Hello, I am JARVIS, I am ready to assist you!")
        #await send_md_content_to_channel(channel)
    client.loop.create_task(periodic_task())
     

@client.event
async def periodic_task():
    first=''
    while True:
        channel = client.get_channel(1160889134777384960)
        if channel:
            rand = generate_random_number(1, 9)
            current_hour_string = korea_time.strftime('%H')
            current_min_string = korea_time.strftime('%M')
            current_date_string = korea_time.strftime('%Y-%m-%d')
            if int(current_hour_string) >= 11 and rand == 3 and first != f'{current_hour_string}':
                first = f'{current_hour_string}'
                #gpt_message =[{"role": "user", "content": f"start conversation with me with casually and start chat regard your side is first. Now the time is {current_hour_string}:{current_min_string} and date {current_date_string}."}]
                message =[{"role": "user", "content": f"sending one of trivial information among my interestings and regarding the date or your base knowlege to me with casually like friend. In my case, I have interesting in AI, Machine learning, Science, Environment, World culture, Beautiful Girls, Automobiles, Technology, South Korea, Japan, Europe, USA, Movies, Games and Animations. Also Now the time is {current_hour_string}:{current_min_string} and date {current_date_string}."}]
                #message = f"sending one of trivial information among my interestings and regarding the date or your base knowlege to me with casually like friend. In my case, I have interesting in AI, Machine learning, Science, Environment, World culture, Beautiful Girls, Automobiles, Technology, South Korea, Japan, Europe, USA, Movies, Games and Animations. Also Now the time is {current_hour_string}:{current_min_string} and date {current_date_string}. "
                #response = LLAMA(message, 400, 0.8)
                response = GPT4_L(message)
                response = lang_trans(response, 'en', 'ko')
                if len(response) > 1200:
                    split = split_message(response)
                    for num, msg in enumerate(split):
                        if num == len(split) - 1:
                            msg = msg + '\n\n'
                        await channel.send(msg)
                else:
                    await channel.send(response + '\n\n')
        else:
            print(f'Channel not found.')
        await asyncio.sleep(600) 
    
@client.event
async def on_message(message):
    first=''
    speech_switch=False
    img_gen_switch=False
    image_analysis_switch=False
    skip=False
    # Avoid bot responding to its own messages
    user = message.author
    user = str(user)
    
    if message.author == client.user:
        return
    username = str(message.author).split('#')[0]
    channel = str(message.channel.name)
    history[user] = [{"role": "assistant", "content": "Your name is JARVIS who is very friendly assistant for user and remember the following elements:" + recall()}]
    if message.attachments:  # Check if there are attachments in the message
        user_message = 'file attached'
        for attachment in message.attachments:
            # Determine the folder based on content type
            folder = ''
            if attachment.content_type.startswith('audio'):
                wait_msg=await message.channel.send("Wait...")
                folder = 'audio'
                # Make sure the folder exists
                if not os.path.exists(folder):
                    os.makedirs(folder)
                # Generate a timestamp for the filename
                current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                file_extension = os.path.splitext(attachment.filename)[1]  # get file extension from the original filename
                # Use the timestamp for the filename and append the appropriate extension
                new_file_name = f"{current_time}{file_extension}"
                file_path = f"./{folder}/{new_file_name}"
                await attachment.save(file_path)
                user_message = speech_to_text(get_latest_file(folder, 'audio'))
            elif attachment.content_type.startswith('image'):
                wait_msg=await message.channel.send("Consider the attached image is characters for OCR or image for captioning")
                # Make sure the folder exists
                folder = 'image'
                file_extension = os.path.splitext(attachment.filename)[1]  # get file extension from the original filename
                new_file_name = f"User_image{file_extension}"
                temp_file_path = f"./{folder}/temp_{new_file_name}"  # Temporary save
                await attachment.save(temp_file_path)

                # Convert and save the image in the desired format
                with Image.open(temp_file_path) as img:
                    # Convert the image to 'RGB' if it's not a jpeg
                    if img.mode == "RGBA" and file_extension not in ['.jpg', '.jpeg']:
                        img = img.convert("RGB")
                    new_file_name = f"User_image.jpg"  # Change to ".png" if you want PNG format
                    final_file_path = f"./{folder}/{new_file_name}"
                    img.save(final_file_path)
                os.remove(temp_file_path)  # Remove the temporary saved file
                ocr_input=ocr_with_easyocr('image/User_image.jpg')
                if ocr_input==[]:
                    cate, obj_img, tot_img=OD('image/User_image.jpg')
                    ic=img_cap(cate, tot_img, obj_img)
                image_analysis_switch=True
            await wait_msg.delete()
    else:
        user_message = str(message.content)
    
    print(username + ' said ' + user_message.lower() + ' in ' + channel)
    MAX_RETRIES = 3  # Adjust this as needed

    if channel == 'jarvis':
        # Send the "thinking" message
        thinking_msg = await message.channel.send("Thinking...")
        retries = 0
        while retries < MAX_RETRIES:
            try:
                speak_prefixes = ['/speech', '/spk', '/speak', '/말해', '/s']
                llama_prefixes = ['/lama', '/llama', '/라마']
                img_gen_prefixes=['/gen', '/image', '/img', '/이미지', '/생성']
                voice_prefixes=['/voice', '/목소리']
                trans_prefixes=['/trans', '/번역', '/tr']
                with open('setting.csv', 'r') as csvfile:
                    reader = csv.reader(csvfile)
                    rows = list(reader)
                # Assuming the number you want to modify is in the second column of the first row
                current_value = int(rows[0][1])
                if any(prefix in user_message for prefix in voice_prefixes):
                    updated_value = current_value + 1  # incrementing the value
                    # Modify the relevant data
                    rows[0][1] = str(updated_value)
                    # Step 3: Write the modified data back to the CSV file
                    with open('setting.csv', 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerows(rows)
                    filtered_message = user_message
                    for prefix in voice_prefixes:
                        if prefix in filtered_message:
                            filtered_message = filtered_message.replace(prefix, '', 1).strip()
                            break  # Exit once a prefix is found and replaced
                    user_message = filtered_message
                    if user_message == '':
                        skip = True
                        output=user_message
                        c_time=0
                if any(prefix in user_message for prefix in speak_prefixes):
                    speech_switch=True
                    filtered_message = user_message
                    for prefix in llama_prefixes:
                        if prefix in filtered_message:
                            user_message = filtered_message.replace(prefix, '', 1).strip()
                if '/r' in user_message:
                    user_message=extract_patterns(user_message)
                    remember(extract_patterns(user_message))
                if any(prefix in user_message for prefix in llama_prefixes):
                    filtered_message = user_message
                    for prefix in llama_prefixes:
                        if prefix in filtered_message:
                            filtered_message = filtered_message.replace(prefix, '', 1).strip()
                            break  # Exit once a prefix is found and replaced
                    #user_messages = prompt_to_chat2(username, filtered_message)
                    msgs=await message.channel.send("LLAMA2 Computing Start")
                    s_time = time.time()
                    output = LLAMA(filtered_message,0,0)
                    e_time = time.time()
                    c_time = e_time - s_time
                    await msgs.delete()
                elif any(prefix in user_message for prefix in img_gen_prefixes):
                    filtered_message = user_message
                    for prefix in img_gen_prefixes:
                        if prefix in filtered_message:
                            filtered_message = filtered_message.replace(prefix, '', 1).strip()
                            break  # Exit once a prefix is found and replaced
                    img_gen_switch=True
                    if any([ord(char) >= 0x0041 and ord(char) <= 0x007A for char in filtered_message]) or \
                        any([ord(char) >= 0x0061 and ord(char) <= 0x007A for char in filtered_message]):
                        filtered_message = filtered_message
                    else:
                        filtered_message=GPT4([{"role": "user", "content": f"Translate input prompt and write image generation prompt into Enlglish. Output with only generated string only.: {filtered_message}"}])
                    msgs=await message.channel.send("Image Generation Start")
                    s_time = time.time()
                    output=image_generation(filtered_message)
                    e_time = time.time()
                    c_time = e_time - s_time
                    user_message = f'Image Generation prompt: {filtered_message}'
                    await msgs.delete()
                elif image_analysis_switch:
                    msgs=await message.channel.send("Image Analysis Start")
                    s_time = time.time()
                    if ocr_input!=[]:
                        output=f'-Detected texts: {ocr_input}'
                    else:
                        output=GPT4([{"role": "user", "content": f"Complete a detailed image captioning without mention of provided captioned result of the image using the image captioning results provided as a list which captioned the entire image, and cropped images of each detected object. Consider that object detection and image captioning results are not always correct, so please complete the captioning that is most correct and Please Do Not mention about the result of object detection, image captioning also do not use negative words. Image captioning sentences:{ic}"}])
                        #output=f'-Simple image captioning: {ic_extra} \n\n-AI Thinking image captioning: {output}'
                    e_time = time.time()
                    c_time = e_time - s_time
                    image_analysis_switch=False
                    await msgs.delete()
                elif any(prefix in user_message for prefix in trans_prefixes):
                    filtered_message = user_message
                    for prefix in trans_prefixes:
                        if prefix in filtered_message:
                            user_message = filtered_message.replace(prefix, '', 1).strip()
                    s_time = time.time()
                    output = lang_trans(user_message, 'auto', 'ko')
                    e_time = time.time()
                    c_time = e_time - s_time
                else:
                    if skip:
                        await message.channel.send('Skip Computaing')
                    else:
                        msgs=await message.channel.send("GPT4 Computing Start")
                        s_time = time.time()
                        messages_for_api = prompt_to_chat(username, user_message)
                        gpt_client = OpenAI(api_key = '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')
                        response = gpt_client.chat.completions.create(
                            model="gpt-4-1106-preview",
                            messages=messages_for_api,
                            temperature=0.8,
                            max_tokens=1000
                        )
                        output = response.choices[0].message.content
                        e_time = time.time()
                        c_time = e_time - s_time
                        await msgs.delete()
                
                rethink_words=["I don't know", "not sure", "my latest training data", "I can't", "current knowledge", "up to December", "real-time capabilities", "As of my knowledge", "updated till", "October 2021", "real-time", "cut-off", 'as an AI']
                if any(prefix in output for prefix in rethink_words):
                    msgs=await message.channel.send("Searching in the Google")
                    keywords = GPT4([{"role": "user", "content": f"Extract Keywords for google search from this prompt: {user_message}"}])
                    google_search_res = google_search(keywords)
                    combined_res = '\n\n'.join(google_search_res)
                    await msgs.delete()
                    output = f"I searched keywords on the Google. \n\nBased on a Google search ({keywords}). \n\nCheck out summarized result or relevant link below: \n{combined_res}"
                        
                if speech_switch:
                    sample=output[:20]
                    if any([ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in sample]):
                        language="ko-KR"
                        if current_value%2==0:
                            Name='ko-KR-Standard-A'
                            gender='FEMALE'
                        else:
                            Name='ko-KR-Standard-C'
                            gender='MALE'
                    elif any([ord(char) >= 0x0041 and ord(char) <= 0x007A for char in sample]) or \
                        any([ord(char) >= 0x0061 and ord(char) <= 0x007A for char in sample]):
                        language="en-GB"
                        if current_value%2==0:
                            Name='en-GB-News-G'
                            gender='FEMALE'
                        else:
                            Name='en-GB-News-J'
                            gender='MALE'
                    else:
                        language="ja-JP"
                        if current_value%2==0:
                            Name='ja-JP-Standard-A'
                            gender='FEMALE'
                        else:
                            Name='ja-JP-Standard-C'
                            gender='MALE'
                    text_to_speech(output, language, Name, gender)
                    try:
                        await message.channel.send(file=discord.File(get_latest_file('speech', 'audio')))
                    except Exception as e:
                        print(f"Error: {e}")
                    speech_switch=False

                if img_gen_switch:
                    output=output
                    await message.channel.send(f"Image prompt: {filtered_message}")
                if len(output) > 1200:
                    split_output = split_message(output)
                    for msg in split_output:
                        await message.channel.send(msg)
                else:
                    # Send the actual response
                    if skip:
                        await message.channel.send('Voice Changed')
                    else:
                        await message.channel.send(output)
                await message.channel.send(f"-(The code ran for {c_time:.2f} seconds.)-")
                # Save both user's message and bot's response to the history
                add_history(username, user_message, output)
                await thinking_msg.delete()
                break
            
            except openai.error.OpenAIError as e:
                print(f"OpenAI error: {e}")
                retries += 1  # Increment the retry count
                await asyncio.sleep(1)  # Give some delay before retrying
            
        # Send a message if max retries reached
        if retries == MAX_RETRIES:
            await thinking_msg.delete()
            await message.channel.send("I encountered an error multiple times. Please try again later.")


@client.event
async def on_resumed():
    # Assuming you have a default channel to send the reconnect message
    default_channel_id = CHANNEL_ID  # Replace with your default channel ID
    channel = client.get_channel(default_channel_id)
    if channel:
        #await channel.send("Successfully resumed and reconnected!")
        print("Successfully resumed and reconnected!")

client.run(token, reconnect=True)
