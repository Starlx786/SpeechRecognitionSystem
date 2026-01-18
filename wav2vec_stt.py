import torch
import soundfile as sf
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Load pre-trained model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def speech_to_text(audio_path):
    # Load audio
    speech, sample_rate = sf.read(audio_path)

    # Convert to mono if stereo
    if len(speech.shape) > 1:
        speech = speech.mean(axis=1)

    # Resample to 16kHz (required by wav2vec)
    if sample_rate != 16000:
        speech = librosa.resample(speech, orig_sr=sample_rate, target_sr=16000)

    # Process audio
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    return transcription.lower()

if __name__ == "__main__":
    print("Transcribed Text:")
    print(speech_to_text("audio.wav"))
