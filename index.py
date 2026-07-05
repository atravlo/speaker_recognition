import os
import torch
import torchaudio
import torch.nn.functional as F

from speechbrain.inference.speaker import EncoderClassifier

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmpdir"
)


def get_embedding(path):
    signal, sample_rate = torchaudio.load(path)

    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            sample_rate,
            16000
        )
        signal = resampler(signal)

    signal = signal.squeeze(0).unsqueeze(0)

    with torch.no_grad():
        embedding = classifier.encode_batch(signal)

    return embedding.squeeze(0)


new_embedding = get_embedding("new.wav")

best_name = None
best_score = -1

for file in os.listdir("embeddings"):

    reference = torch.load(
        os.path.join("embeddings", file)
    )

    score = F.cosine_similarity(
        reference,
        new_embedding,
        dim=-1
    ).item()

    print(f"{file}: {score:.3f}")

    if score > best_score:
        best_score = score
        best_name = os.path.splitext(file)[0]

print()
print("Best match:", best_name)
print("Similarity:", best_score)

THRESHOLD = 0.75

if best_score >= THRESHOLD:
    print("Speaker identified:", best_name)
else:
    print("Unknown speaker")