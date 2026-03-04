from speechbrain.pretrained import SpeakerRecognition

verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmpdir"
)

score, prediction = verification.verify_files(
    "voices/voice1.m4a",   # эталон
    "voices/voice2.m4a"      # новая запись
)

print("Score:", score)
print("Same person:", prediction)