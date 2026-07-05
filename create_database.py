import os
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmpdir"
)

os.makedirs("embeddings", exist_ok=True)

for file in os.listdir("voices"):

    if not file.lower().endswith(".wav"):
        continue

    path = os.path.join("voices", file)

    signal, sample_rate = torchaudio.load(path)

    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    if sample_rate != 16000:
        signal = torchaudio.functional.resample(
            signal,
            sample_rate,
            16000
        )

    signal = signal.squeeze(0).unsqueeze(0)

    with torch.no_grad():
        embedding = classifier.encode_batch(signal)

    name = os.path.splitext(file)[0]

    torch.save(
        embedding.squeeze(),
        f"embeddings/{name}.pt"
    )

    print(f"Saved {name}.pt")