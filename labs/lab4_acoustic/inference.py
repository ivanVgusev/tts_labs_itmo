import json
from pathlib import Path

import soundfile as sf
import torch
from TTS.tts.configs.fastspeech2_config import Fastspeech2Config
from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.models.forward_tts import ForwardTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

CONFIG_PATH = Path("labs/lab4_acoustic/runs/baseline/baseline-October-28-2026_11+31AM-b1ba7ed/config.json")
CHECKPOINT_PATH = Path("labs/lab4_acoustic/runs/baseline/baseline-October-28-2026_11+31AM-b1ba7ed/checkpoint_250083.pth")

# Читаем исходные значения без десериализации Coqpit
with CONFIG_PATH.open(encoding="utf-8") as f:
    raw_config = json.load(f)

# Загружаем остальную конфигурацию
config = Fastspeech2Config()
config.load_json(str(CONFIG_PATH))

# Восстанавливаем исходный фонемный словарь
char_data = raw_config["characters"]

config.characters = CharactersConfig(
    **{
        **(char_data or {}),
        "characters": None,
    }
)

# Возвращаем список, который Coqpit не смог десериализовать
config.characters.characters = char_data["characters"]

# Создаём модель
ap = AudioProcessor.init_from_config(config)
tokenizer, config = TTSTokenizer.init_from_config(config)

model = ForwardTTS(config, ap, tokenizer, speaker_manager=None) # type: ignore

print("Размер словаря:", tokenizer.characters.num_chars)
print("Embedding:", model.emb.weight.shape)

# Загружаем веса
model.load_checkpoint(config, str(CHECKPOINT_PATH), eval=True)

print("Checkpoint успешно загружен!")

# Первая запись из датасета RUSLAN
metadata_path = Path("data/metadata_phones.csv")

with metadata_path.open(encoding="utf-8") as f:
    line = next(line for line in f if line.strip())

utt_id, text, phones = line.strip().split("|", maxsplit=2)

print("Текст:", text)
print("Фонемы:", phones)

# Токенизация
token_ids = tokenizer.text_to_ids(phones)
tokens = torch.tensor([token_ids], dtype=torch.long)

# Синтез
model.eval()
with torch.inference_mode():
    outputs = model.inference(tokens)

# Результат модели: [batch, frames, 80]
mel = outputs["model_outputs"][0].cpu().numpy()

print("Mel shape:", mel.shape)

# Проверяем, что спектрограмма корректна
assert mel.ndim == 2 and mel.shape[1] == 80
assert mel.shape[0] > 0, "Модель не сгенерировала акустические кадры"

# Mel -> WAV через Griffin-Lim
wav = ap.inv_melspectrogram(mel.T)

output_path = Path("inference/test_250083.wav")
sf.write(output_path, wav, ap.sample_rate) # type: ignore

print(f"Готово: {output_path.resolve()}")
print(f"Длительность: {len(wav) / ap.sample_rate:.2f} с") # type: ignore