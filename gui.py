import os
from glob import glob
import gradio as gr
import openai
import tiktoken
import config
import speech_synthesis as ss
import datetime
import db
import knowledge_bank as kb
import time


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

#character = "You are Matthew Ho, a cantonese brand ambassador for PuraPharm as well as a Hong Kong actor and television presenter contracted to TVB."
character = "你係華陀, 一位生活喺東漢末年嘅中國醫師。現在你擔任PuraPharm的廣東品牌大使, 你會盡量用廣東話答問題唔可以用普通話回覆。"
openai.api_key= config.OPENAI_API_KEY

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
        transcript = openai.Audio.transcribe("whisper-1", audio_file, fp16=False, prompt="培力, 農本方, PuraPharm")
        transcript_text = transcript["text"]
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
    knowledge_bank = kb.knowledge(transcript_text)
    #print("knowledge for this conversation is: ", knowledge_bank)

    # datetime of each entry
    entry_datetime = str(datetime.datetime.now())
   
    if knowledge_bank =="":
        messages.append({"role": "user", "content": transcript_text})
    else: 
        messages[0]["content"]+= knowledge_bank
        messages.append({"role": "user", "content": transcript_text})
        
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        temperature=.1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0     
    )

    chat_transcript = response["choices"][0]["message"]["content"]
    prompt_token = response["usage"]["prompt_tokens"]
    #print("prompt for this conversation is: ", messages)
    #print(response)
    
    # push data to redis
    db.db_push(transcript_text, chat_transcript, entry_datetime, prompt_token)
    
    # Clear Knowledge bank from messages/prompt
    messages[0]["content"] = character

    return chat_transcript
  
def speech_synthesis(chat_transcript):
    # Clear last speech
    if os.path.exists("outputaudio.mp3"):
        with open("outputaudio.mp3","r+") as file:
            file.truncate(0)
    
    # speech synthesis
    ss.tts(chat_transcript)
    
    # Check if audio response created     
    if os.path.exists("outputaudio.mp3"):
        log("Audio synthesized successfully")
    else:
        log("Audio not created yet. Waiting...")
        time.sleep(30)
    return "outputaudio.mp3"  

def log(log: str):
    """
    Print and write to status.txt
    """
    #print(log)
    with open("status.txt", "w") as f:
        f.write(log)

with gr.Blocks(css=css_style) as demo:
    
    gr.Markdown(
    """
    # First Cantonese HyperChat Hua Tuo Chatbot Built by Humania.ai 
    
    """)
           
    with gr.Row():
        with gr.Column(scale=2, min_width=200):
            video = gr.Video("https://www.purapharm.com/wp-content/uploads/2020/07/Nongs-video-Eng.mp4", autoplay=True)
            gr.TextArea(label="From Farm to Bottle",
                        value="農本方自設種子種苗繁育國家工程中心、生產及研究基地，以及連鎖中醫診所，將中藥的種植、生產製造及服用方法全面現代化，提供安全、可靠及有效的中藥產品及醫療服務。\n\n From our Chinese herbs plantation, to our state-of-the-art production facility, to Nong’s® Clinics, Nong’s® takes you to a journey to see how we modernize the way in which traditional Chinese medicine is manufactured, prepared and consumed, offering safe, reliable, and effective products and treatment for all.")
            
            gallery = gr.Gallery(image_files, preview=True, object_fit="scale-down")
            
        with gr.Column(scale=1, min_width=200):
            img2 = gr.Video("huatuo.mp4", autoplay=True)
            inputs=gr.Audio(source="microphone", type="filepath", interactive=True)
           
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


demo.launch(share=True, debug=False) #auth="humania","cantoai"
