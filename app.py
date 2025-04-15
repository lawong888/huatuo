import os
from glob import glob
import gradio as gr
import redis
from openai import OpenAI
# from groq import Groq
import tiktoken
import datetime
import time
import azure.cognitiveservices.speech as speechsdk
import asyncio
import sys
import re

# used only for local testing
try:
    import config
except ImportError:
    # If config.py doesn't exist, we'll just ignore it
    pass

# Set event loop policy for Windows
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def remove_markdown(text):
    text = re.sub(r'\*+', '', text)  # Remove bold/italic markers
    text = re.sub(r'`', '', text)  # Remove backticks
    text = re.sub(r'#+\s', '', text)  # Remove hashtags (headers)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # Remove URL formatting
    return text.strip()

client = OpenAI()
""" #Groq
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)
"""
# footer
css_style = """
    footer {
        display: None !important;
    }

    header {
        background-color: #333; /* Background color for the header */
        color: #fff; /* Text color */
        padding: 20px; /* Padding around the header */
    }
   
"""

# character = "You are Matthew Ho, a cantonese brand ambassador for PuraPharm as well as a Hong Kong actor and television presenter contracted to TVB."
character = "你係華陀, 一位生活喺東漢末年嘅中國醫師。現在你擔任PuraPharm的廣東品牌大使, 你會盡量用廣東話答問題唔可以用普通話回覆。請唔好喺你嘅回應中使用任何Markdown格式，例如星號或者底線。"
image_files = glob(os.path.join(".", "images", "*"))       
image_dict = {image_path.split("/")[-1].split(".")[-2].strip("images\\"): image_path
    for image_path in image_files}


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0301":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    
def transcribe(audio):
    global transcript_text
    if audio is None:
        transcript_text = "你好"
    else: 
        #print(audio)
        audio_file = open(audio, "rb")
        transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file, 
            prompt="培力, 農本方, PuraPharm"
            )
        transcript_text = transcript.text
        #print(transcript)
    return transcript_text

def chat(transcript_text):
    global messages, chat_transcript

    messages = [{"role": "system", "content": character},
               {"role": "user", "content": "介紹吓培力集團"},
               {"role": "assistant", "content": "自1998年成立至今,培力集團一直致力於中醫藥的國際化及現代化，並與多位國際知名的學者夥拍合作，於中藥市場中早已建立了科技先驅者的地位。經過不斷創新，培力無論在產品研發、生產技術及設備、市場及營銷策劃、品質控制以至基礎科學研究方面，已被譽為領先同業，並受到消費者及醫藥界的廣泛認同。"},
               {"role": "user", "content": "你有咩值得推薦嘅產品?"},
               {"role": "assistant", "content": "我只能推薦以下一個產品，不可以自己配方. 我哋嘅產品有保健養生系列同藥用治療系列。保健養生系列包括金靈芝，烏髮濃男士專用特效配方，烏髮濃女士專用特效配方，烏髮濃男士專用健髮洗髮露，烏髮濃女士專用健髮洗髮露，培力心全一通，培力鼻敏感配方，培力補肝配方，嵐國蟲草菌絲體。藥用治療系列包括農本方感冒沖劑銀翹散，農本方止咳沖劑止嗽散，農本方複方羅漢果止咳沖劑，農本方失眠沖劑酸棗仁湯和安固生雲芝膠囊，益抗適，農本方濃縮中藥配方顆粒"},
               ]
    
    # create knowledge bank for AI
    knowledge_bank = knowledge(transcript_text)
    #print("knowledge for this conversation is: ", knowledge_bank)

    # datetime of each entry
    entry_datetime = str(datetime.datetime.now())
   
    if knowledge_bank =="":
        messages.append({"role": "user", "content": transcript_text})
    else: 
        messages[0]["content"]+= knowledge_bank
        messages.append({"role": "user", "content": transcript_text})

       
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=.1,
        # top_p=1,
        frequency_penalty=0,
        presence_penalty=0     
    )

    chat_transcript = response.choices[0].message.content
    cleaned_transcript = remove_markdown(chat_transcript)
    prompt_token = response.usage.prompt_tokens

    #print("prompt for this conversation is: ", messages)
    #print(response)
    
    # push data to redis
    db_push(transcript_text, cleaned_transcript, entry_datetime, prompt_token)
    
    # Clear Knowledge bank from messages/prompt
    messages[0]["content"] = character

    return cleaned_transcript
  
def speech_synthesis(chat_transcript):
    try:
        # Clear last speech
        if os.path.exists("outputaudio.mp3"):
            with open("outputaudio.mp3", "r+") as file:
                file.truncate(0)
        
        # Remove markdown formatting
        clean_transcript = remove_markdown(chat_transcript)
        
        # Speech synthesis
        tts(clean_transcript)
        
        # Check if audio response created     
        if os.path.exists("outputaudio.mp3"):
            log("Audio synthesized successfully")
            return "outputaudio.mp3"
        else:
            log("Audio not created yet. Waiting...")
            time.sleep(30)
            if os.path.exists("outputaudio.mp3"):
                return "outputaudio.mp3"
            else:
                raise Exception("Audio file not created after waiting")
    except Exception as e:
        log(f"Error in speech synthesis: {e}")
        return None


os.environ['SPEECH_KEY'] = '6c638ef5e42242518c67c22fc62b08b4'
os.environ['SPEECH_REGION'] = 'southeastasia'
os.environ['SSL_CERT_DIR']='/etc/ssl/certs'

# This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
#audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

# Sets the synthesis output format.
speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3)

# Replace with your own audio file name.
file_name = "outputaudio.mp3"
file_config = speechsdk.audio.AudioOutputConfig(filename=file_name)

# The language of the voice that speaks.
speech_config.speech_synthesis_voice_name='zh-HK-WanLungNeural'

#speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=file_config)

# Get text from the console and synthesize to the default speaker.
def tts(text):

    speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()

    if speech_synthesis_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        pass
        #print("Speech synthesized for text [{}]".format(text))
    elif speech_synthesis_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_synthesis_result.cancellation_details
        pass
        #print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                pass
                #print("Error details: {}".format(cancellation_details.error_details))
                #print("Did you set the speech resource key and region values?")

def log(log: str):
    """
    Print and write to status.txt
    """
    #print(log)
    with open("status.txt", "w") as f:
        f.write(log)

# Initialize Redis connection
try:
    db = redis.Redis(
        host=os.environ.get('REDIS_HOST'),
        port=int(os.environ.get('REDIS_PORT', 6379)),
        username=os.environ.get('REDIS_USERNAME'),
        password=os.environ.get('REDIS_PASSWORD'),
        decode_responses=True
    )
    # Test the connection
    db.ping()
    print("Successfully connected to Redis")
    user_db = db.llen('user_message')
except redis.exceptions.ConnectionError as e:
    print(f"Failed to connect to Redis: {e}")
    print("Redis connection details:")
    print(f"Host: {os.environ.get('REDIS_HOST')}")
    print(f"Port: {os.environ.get('REDIS_PORT')}")
    print(f"Username: {os.environ.get('REDIS_USERNAME')}")
    print("Password: [REDACTED]")
    db = None
    user_db = 0
except Exception as e:
    print(f"An unexpected error occurred while connecting to Redis: {e}")
    db = None
    user_db = 0

# push data to redis
def db_push(transcript_text, chat_transcript, entry_datetime, token):

    db.rpush('user_message', transcript_text)
    db.rpush('matthew_response', chat_transcript)
    db.rpush('Datetime_HK', entry_datetime)
    db.rpush('Token_HK', token)

product_files = glob(os.path.join(".", "products", "*"))
product_dict = {product_path.split("/")[-1].split(".")[-2].strip("products\\"): product_path
    for product_path in product_files}

def get_product_info(keywords):
    """
    Retrieve product information based on keywords.
    
    Args:
    keywords (list): List of keywords to search for in product files
    
    Returns:
    list: List of product information strings
    """
    file_names = [file_name for file_name in product_files if any(keyword in file_name for keyword in keywords)]
    knowledge = []
    for file_name in file_names:
        with open(file_name, 'r', encoding='utf-8') as file:
            content = file.read()
            knowledge.append(f'推薦產品: {content}')
    return knowledge

def knowledge(transcript_text):
    """
    Generate knowledge based on the transcript text.
    
    Args:
    transcript_text (str): The transcribed user input
    
    Returns:
    str: Comma-separated string of relevant product information
    """
    keyword_map = {
        ('脫髮', '白髮', '掉頭髮', '洗髮'): ['烏髮濃專用健髮洗髮露'],
        ('失眠', '瞓唔着'): ['失眠'],
        ('感冒',): ['感冒'],
        ('咳', '嗽'): ['咳', '嗽'],
        ('心',): ['心'],
        ('病後', '調理'): ['雲芝'],
        ('免疫力', '抵抗力', '身體虛弱'): ['靈芝', '雲芝'],
        ('全面保健', '延緩老化'): ['金靈芝'],
        ('改善呼吸', '補肺益腎'): ['蟲草菌絲'],
        ('鼻敏感',): ['鼻敏感'],
        ('肝',): ['肝'],
        ('濃縮中藥',): ['濃縮中藥'],
    }

    knowledge = []
    for keywords, product_keywords in keyword_map.items():
        if any(keyword in transcript_text for keyword in keywords):
            knowledge.extend(get_product_info(product_keywords))

    return ', '.join(knowledge)


# main Gradio Body
with gr.Blocks(css=css_style) as app:

    gr.Markdown(
    """
    # First Cantonese HyperChat Hua Tuo Chatbot Built by Humania.ai 

    """)

    with gr.Row():
        with gr.Column(scale=2, min_width=200):
            video = gr.Video("https://www.purapharm.com/wp-content/uploads/2020/07/Nongs-video-Eng.mp4", autoplay=False)
            gr.TextArea(label="From Farm to Bottle",
                        value="農本方自設種子種苗繁育國家工程中心、生產及研究基地，以及連鎖中醫診所，將中藥的種植、生產製造及服用方法全面現代化，提供安全、可靠及有效的中藥產品及醫療服務。\n\n From our Chinese herbs plantation, to our state-of-the-art production facility, to Nong’s® Clinics, Nong’s® takes you to a journey to see how we modernize the way in which traditional Chinese medicine is manufactured, prepared and consumed, offering safe, reliable, and effective products and treatment for all.")
            
            gallery = gr.Gallery(image_files, preview=True, object_fit="scale-down")
            
        with gr.Column(scale=1, min_width=200):
            # img2 = gr.Video("huatuo.mp4", autoplay=True)
            # inputs=gr.Audio(source="microphone", type="filepath", interactive=True)
            img2 = gr.Video("huatuo.mp4", autoplay=True)
            inputs=gr.Audio(type="filepath", source="microphone", interactive=True)
           
            with gr.Row(visible=False) as output_col:
                submit_btn = gr.Button("Submit")
                # datetime of each entry
                entry_datetime = str(datetime.datetime.now())
                clear_btn = gr.ClearButton(value="Clear")
                
            output_1 = gr.Textbox(label="Your messages:", visible=False)
            output_2 = gr.Textbox(label="Hua Tuo:", visible=False, lines=3)
            
            def submit_1():
                return {output_col: gr.update(visible=True)}
            
            def submit_2():
                return {output_1: gr.update(visible=True),
                        output_2: gr.update(visible=True),
                        output_col: gr.update(visible=False)}
            
            inputs.stop_recording(submit_1, outputs=output_col)
            submit_btn.click(submit_2, outputs=[output_1, output_2, output_col]).then(fn=transcribe, inputs=inputs, outputs=output_1).then(fn=chat, inputs=output_1, outputs=output_2).success(fn=speech_synthesis, inputs=output_2, outputs=gr.Audio("outputaudio.mp3", autoplay=True))
            clear_btn.click(lambda: None, None, inputs, queue=False)

# app.launch(share=False, debug=False, server_name="0.0.0.0", server_port=7860)  # auth="humania","cantoai"

# Run the Gradio app directly
if __name__ == "__main__":
    app.launch(share=False, debug=False)
else:
    # For ASGI servers like Uvicorn
    app = app.launch(show_api=False)
