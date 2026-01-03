"""Some utility functions for the app."""
from base64 import b64encode
from io import BytesIO

from gtts import gTTS
from mtranslate import translate
from speech_recognition import AudioFile, Recognizer
from transformers import (BlenderbotSmallForConditionalGeneration,
                          BlenderbotSmallTokenizer)


def stt(audio: object, language: str) -> str:
    """Converts speech to text.

    Args:
        audio: record of user speech

    Returns:
        text (str): recognized speech of user
    """
    r = Recognizer()
    # open the audio file
    with AudioFile(audio) as source:
        # listen for the data (load audio to memory)
        audio_data = r.record(source)
        # recognize (convert from speech to text)
        text = r.recognize_google(audio_data, language=language)
    return text


def to_en_translation(text: str, language: str) -> str:
    """Translates text from specified language to English.

    Args:
        text (str): input text
        language (str): desired language

    Returns:
        str: translated text
    """
    return translate(text, "en", language)


def from_en_translation(text: str, language: str) -> str:
    """Translates text from english to specified language.

    Args:
        text (str): input text
        language (str): desired language

    Returns:
        str: translated text
    """
    return translate(text, language, "en")


class TextGenerationPipeline:
    """Pipeline for text generation of blenderbot model.

    Returns:
        str: generated text
    """

    # load tokenizer and the model
    model_name = "facebook/blenderbot_small-90M"
    tokenizer = BlenderbotSmallTokenizer.from_pretrained(model_name)
    model = BlenderbotSmallForConditionalGeneration.from_pretrained(model_name)

    def __init__(self, **kwargs):
        """Specififying text generation parameters.

        For example: max_length=100 which generates text shorter than
        100 tokens. Visit:
        https://huggingface.co/docs/transformers/main_classes/text_generation
        for more parameters
        """
        self.__dict__.update(kwargs)

    def preprocess(self, text) -> str:
        """Tokenizes input text.

        Args:
            text (str): user specified text

        Returns:
            torch.Tensor (obj): text representation as tensors
        """
        return self.tokenizer(text, return_tensors="pt")

    def postprocess(self, outputs) -> str:
        """Converts tensors into text.

        Args:
            outputs (torch.Tensor obj): model text generation output

        Returns:
            str: generated text
        """
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def __call__(self, text: str) -> str:
        """Generates text from input text.

        Args:
            text (str): user specified text

        Returns:
            str: generated text
        """
        tokenized_text = self.preprocess(text)
        output = self.model.generate(**tokenized_text, **self.__dict__)
        return self.postprocess(output)


def tts(text: str, language: str) -> object:
    """Converts text into audio object.

    Args:
        text (str): generated answer of bot

    Returns:
        object: text to speech object
    """
    return gTTS(text=text, lang=language, slow=False)


def tts_to_bytesio(tts_object: object) -> bytes:
    """Converts tts object to bytes.

    Args:
        tts_object (object): audio object obtained from gtts

    Returns:
        bytes: audio bytes
    """
    bytes_object = BytesIO()
    tts_object.write_to_fp(bytes_object)
    bytes_object.seek(0)
    return bytes_object.getvalue()


def html_audio_autoplay(bytes: bytes) -> object:
    """Creates html object for autoplaying audio at gradio app.

    Args:
        bytes (bytes): audio bytes

    Returns:
        object: html object that provides audio autoplaying
    """
    b64 = b64encode(bytes).decode()
    html = f"""
    <audio controls autoplay>
    <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    """
    return html
