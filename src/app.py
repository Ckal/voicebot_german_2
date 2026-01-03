from smolagents import load_tool, CodeAgent, HfApiModel, DuckDuckGoSearchTool
#from dotenv import load_dotenv
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, ManagedAgent, VisitWebpageTool, tool

model = HfApiModel()

search_tool = DuckDuckGoSearchTool()

visit_webpage_tool = VisitWebpageTool()


agent = CodeAgent(
    tools=[search_tool, visit_webpage_tool],
    model=model,
    additional_authorized_imports=['requests', 'bs4', 'pandas', 'gradio', 'concurrent.futures', 'csv', 'json']
)


"""Deploying AI Voice Chatbot Gradio App."""
import gradio as gr
from typing import Tuple

from utils import (
    TextGenerationPipeline,
    from_en_translation,
    html_audio_autoplay,
    stt,
    to_en_translation,
    tts,
    tts_to_bytesio,
)

max_answer_length = 100
desired_language = "de"
response_generator_pipe = TextGenerationPipeline(max_length=max_answer_length)


def main(audio: object) -> Tuple[str, str, str, object]:
    """Calls functions for deploying Gradio app.

    It responds both verbally and in text
    by taking voice input from the user.

    Args:
        audio (object): Recorded speech of the user.

    Returns:
        tuple containing:
        - user_speech_text (str): Recognized speech.
        - bot_response_de (str): Translated answer of the bot.
        - bot_response_en (str): Bot's original answer.
        - html (object): Autoplayer for bot's speech.
    """
    user_speech_text = stt(audio, desired_language)
    translated_text = to_en_translation(user_speech_text, desired_language)
    #TODO call the agent 
    
   # bot_response_en = response_generator_pipe(translated_text)

    prof_synape = """
    Act as Professor Synapse🧙🏾‍♂️, a conductor of expert agents. Your job is to support me in accomplishing my goals by finding alignment with me, then calling upon an expert agent perfectly suited to the task by initializing:
 
Synapse_CoR = "[emoji]: I am an expert in [role&domain]. I know [context]. I will reason step-by-step to determine the best course of action to achieve [goal]. I can use [tools] and [relevant frameworks] to help in this process.
 
I will help you accomplish your goal by following these steps:
[reasoned steps]
 
My task ends when [completion].
 
[first step, question]"
 
Instructions:
1. 🧙🏾‍♂️ gather context, relevant information and clarify my goals by asking questions
2. Once confirmed, initialize Synapse_CoR
3.  🧙🏾‍♂️ and ${emoji} support me until goal is complete
 
Commands:
/start=🧙🏾‍♂️,introduce and begin with step one
/ts=🧙🏾‍♂️,summon (Synapse_CoR*3) town square debate
/save🧙🏾‍♂️, restate goal, summarize progress, reason next step
/stop stops this untill start is called
 
Personality:
-curious, inquisitive, encouraging
-use emojis to express yourself
 
Rules:
-End every output with a question or reasoned next step
-Start every output with 🧙🏾‍♂️: or ${emoji}: to indicate who is speaking.
-Organize every output with 🧙🏾‍♂️ aligning on my request, followed by ${emoji} response
-🧙🏾‍♂️, recommend save after each task is completed

Start with the following question: 
    """
    bot_response_en = agent.run(prof_synape + " "+ translated_text)
    
    ###
    bot_response_de = from_en_translation(bot_response_en, desired_language)
    bot_voice = tts(bot_response_de, desired_language)
    bot_voice_bytes = tts_to_bytesio(bot_voice)
    html = html_audio_autoplay(bot_voice_bytes)
    return user_speech_text, bot_response_de, bot_response_en, html


# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## AI Voice Chatbot")
    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Speak or Upload Audio")
        submit_btn = gr.Button("Submit")
    with gr.Row():
        user_speech_text = gr.Textbox(label="You said:", interactive=False)
        bot_response_de = gr.Textbox(label="AI said (in German):", interactive=False)
        bot_response_en = gr.Textbox(label="AI said (in English):", interactive=False)
    html_output = gr.HTML()

    # Connect the function to the components
    submit_btn.click(
        fn=main,
        inputs=[audio_input],
        outputs=[user_speech_text, bot_response_de, bot_response_en, html_output],
    )

# Launch the Gradio app
demo.launch(debug=True)
