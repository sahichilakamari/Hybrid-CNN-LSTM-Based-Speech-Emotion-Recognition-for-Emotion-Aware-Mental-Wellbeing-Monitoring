import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

def extract_wav2vec(path):
    y, sr = librosa.load(path, sr=16000)
    
    inputs = processor(y, sampling_rate=16000, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings