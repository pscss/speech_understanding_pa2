import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import os

def load_data(dataset_path):
    # Load your dataset here
    # This function should return a list of tuples (audio_path, speaker_id)
    # For simplicity, let's assume each audio file is named as speaker_id_*.wav
    data = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.wav'):
                speaker_id = file.split('_')[0]
                data.append((os.path.join(root, file), speaker_id))
    return data

def extract_features(model, processor, data):
    features = []
    for audio_path, speaker_id in data:
        waveform, sample_rate = torchaudio.load(audio_path)
        input_values = processor(waveform, return_tensors="pt", sampling_rate=sample_rate).input_values
        with torch.no_grad():
            hidden_states = model(input_values).last_hidden_state
            embeddings = torch.mean(hidden_states, dim=1)
        features.append((embeddings, speaker_id))
    return features

def calculate_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def main():
    # Load pre-trained model
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")

    # Load VoxCeleb1-H dataset
    dataset_path = "path/to/voxceleb1-h"  # Update this path
    data = load_data(dataset_path)

    # Extract features
    features = extract_features(model, processor, data)

    # Calculate pairwise distances and labels
    distances = []
    labels = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            dist = torch.norm(features[i][0] - features[j][0]).item()
            distances.append(dist)
            labels.append(1 if features[i][1] == features[j][1] else 0)

    # Calculate EER
    eer = calculate_eer(labels, distances)
    print(f"EER: {eer * 100:.2f}%")

if __name__ == "__main__":
    main()
