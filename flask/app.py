from flask import Flask, request, jsonify
from transformers import AutoModelForSeq2SeqLM
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
from ttsmms import TTS
import soundfile as sf
import shutil
import tempfile
import torch
import whisper
import random 
import os

app = Flask(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4

model = whisper.load_model("small")

def transcribe(audio):
    options = dict(beam_size=5, best_of=5)
    translate_options = dict(task="transcribe", **options)
    result = model.transcribe(audio, **translate_options)
    return result

def initialize_model_and_tokenizer(ckpt_dir, direction):
    tokenizer = IndicTransTokenizer(direction=direction)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = model.to(DEVICE).half() if DEVICE == "cuda" else model.to(DEVICE)
    model.eval()
    return tokenizer, model

def batch_translate(input_sentences, src_lang, tgt_lang, model, tokenizer, ip):
    translations = []
    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = ip.preprocess_batch(input_sentences[i: i + BATCH_SIZE], src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = tokenizer(batch, src=True, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True,).to(DEVICE)
        with torch.no_grad():
            generated_tokens = model.generate(**inputs, use_cache=True, min_length=0, max_length=256, num_beams=5, num_return_sequences=1,)
        generated_tokens = tokenizer.batch_decode(generated_tokens.cpu().tolist(), src=False)
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)
    torch.cuda.empty_cache()
    return translations

def generate_audio(text, tts):
    wav = tts.synthesis(text)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_path = audio_file.name
        sf.write(audio_path, wav["x"], wav["sampling_rate"])
    return audio_path

@app.route("/process_audio", methods=["POST"])
def process_audio():
    audio_file = request.files["file"]
    file_name = str(random.randint(0, 100000))
    audio_file.save(file_name)

    # Transcribe audio
    transcribed_text = transcribe(file_name)
    transcription = transcribed_text['text']
    
    # Translate text
    en_indic_ckpt_dir = "ai4bharat/indictrans2-en-indic-dist-200M"
    en_indic_tokenizer, en_indic_model = initialize_model_and_tokenizer(en_indic_ckpt_dir, "en-indic")
    ip = IndicProcessor(inference=True)
    en_sents = [transcribed_text['text']]
    src_lang, tgt_lang = "eng_Latn", "hin_Deva"
    translated_text = batch_translate(en_sents, src_lang, tgt_lang, en_indic_model, en_indic_tokenizer, ip)[0]
    
    # Synthesize audio
    tts = TTS("hin")
    audio_path = generate_audio(translated_text, tts)
    
    # Save audio file
    destination_file = "test/out.wav"
    shutil.copy(audio_path, destination_file)
    os.remove(file_name)
    
    return jsonify({'transcribed_text': transcription, 'translated_text': translated_text, 'audio_path': destination_file})

if __name__ == "__main__":
    app.run(debug=True)
