from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")

# send pipeline to GPU (when available)
import torch
pipeline.to(torch.device("cpu"))

# apply pretrained pipeline
diarization = pipeline(r"C:\Users\AynurV\Downloads\zapis300.mp3")

# print the result
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"[<start={turn.start:.1f} stop={turn.end:.1f} speaker={speaker}>]")