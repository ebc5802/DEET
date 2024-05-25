from transformers import AutoProcessor, BarkModel
import scipy

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

# voice_preset = "v2/en_speaker_6"
voice_preset = "v2/zh_speaker_4" # Chinese Female Voice

inputs = processor("Hello. 你好我的名字是丹尼尔.", voice_preset=voice_preset)

audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze() # Can use CPU or GPU

sample_rate = model.generation_config.sample_rate
scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=audio_array)