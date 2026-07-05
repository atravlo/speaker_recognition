import torchaudio

waveform, sr = torchaudio.load("voices/new.wav")

print("Shape:", waveform.shape)
print("Sample rate:", sr)