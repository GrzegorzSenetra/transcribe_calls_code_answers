# transcribe_calls_code_answers

Telefon na głośniku obok komputera. Mikrofon nagrywa rozmowę, Whisper robi transkrypcję z etykietami JA/ROZMÓWCA,
a gdy padnie pytanie, `claude -p` (tylko do odczytu: Read/Grep/Glob) przeszukuje kod i wysyła odpowiedź na Telegram.

## Setup

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt --index-strategy unsafe-best-match
export TG_TOKEN=...  # od @BotFather
export TG_TOKEN=...  # od @BotFather
export TG_CHAT_ID=...    # napisz do bota, potem: curl https://api.telegram.org/bot$TG_TOKEN/getUpdates
.venv/bin/python call.py tg       # sprawdź, czy wiadomość dochodzi
.venv/bin/python call.py enroll   # 20 s Twojego głosu, bez telefonu w tle
.venv/bin/python call.py
```

Mikrofon to domyślne źródło PulseAudio (zmienisz w `pavucontrol`). Transkrypcja zapisuje się do `rozmowa_*.txt`.

## Kalibracja (zmienne środowiskowe)

| zmienna | domyślnie | kiedy zmienić |
|---|---|---|
| `ME_THRESHOLD` | 0.75 | Twoje wypowiedzi oznacza jako ROZMÓWCA → obniż; odwrotnie → podnieś |
| `SILENCE_RMS` | auto (2× szum z 2 s na starcie) | nie tnie po zdaniach → podnieś; ucina w środku zdania → obniż |
| `WHISPER_MODEL` | small | `medium` lepiej radzi sobie z polskim, ale na CPU jest ~3x wolniejszy |
| `CODE_DIR` | `~` | gdzie Claude ma szukać kodu |
