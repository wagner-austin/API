# Services

## Overview

| Service | Port | Description | GPU |
|---------|------|-------------|-----|
| data-bank-api | 8001 | Content-addressed file storage | No |
| Model-Trainer | 8005 | GPT-2 and Char-LSTM training | Yes |
| handwriting-ai | 8004 | MNIST digit recognition | No |
| turkic-api | 8000 | Turkic language NLP | No |
| transcript-api | 8003 | YouTube transcription | No |
| qr-api | 8002 | QR code generation | No |
| music-wrapped-api | 8006 | Music analytics | No |

---

## data-bank-api

Content-addressed file storage service with atomic writes and SHA256 hashing.

**Features:**
- Upload files via multipart form data
- Retrieve files by content hash
- Automatic deduplication
- Atomic writes (no partial files)

**Key Endpoints:**
```
POST /files          Upload file, returns hash
GET  /files/{hash}   Download file by hash
HEAD /files/{hash}   Check if file exists
```

**Start:**
```bash
make up-databank
```

**Docs:** [README](../services/data-bank-api/README.md) | [API](../services/data-bank-api/docs/api.md)

---

## Model-Trainer

GPT-2 and Char-LSTM model training with CUDA GPU support.

**Features:**
- Fine-tune GPT-2 on custom text corpora
- Train character-level LSTM models
- Background job processing via RQ
- Progress streaming via Redis pub/sub
- Model artifact packaging with manifests

**Key Endpoints:**
```
POST /jobs/train     Start training job
GET  /jobs/{id}      Get job status
GET  /jobs/{id}/logs Stream training logs
POST /generate       Generate text from trained model
```

**Requirements:**
- NVIDIA GPU with CUDA 12.4+
- NVIDIA Container Toolkit

**Start:**
```bash
make up-trainer
```

**Docs:** [README](../services/Model-Trainer/README.md) | [API](../services/Model-Trainer/docs/api.md) | [Design](../services/Model-Trainer/DESIGN.md)

---

## handwriting-ai

MNIST digit recognition with calibrated confidence scores.

**Features:**
- Recognize handwritten digits (0-9)
- Calibrated probability estimates
- Support for multiple image formats
- Batch prediction

**Key Endpoints:**
```
POST /predict        Predict digit from image
POST /predict/batch  Predict multiple images
```

**Start:**
```bash
make up-handwriting
```

**Docs:** [README](../services/handwriting-ai/README.md) | [API](../services/handwriting-ai/docs/api.md)

---

## turkic-api

Turkic language detection and IPA transliteration.

**Features:**
- Detect Turkic languages (Turkish, Kazakh, Uzbek, etc.)
- Transliterate to IPA (International Phonetic Alphabet)
- Support for multiple scripts (Latin, Cyrillic, Arabic)

**Key Endpoints:**
```
POST /detect         Detect language from text
POST /transliterate  Convert to IPA
```

**Start:**
```bash
make up-turkic
```

**Docs:** [README](../services/turkic-api/README.md) | [API](../services/turkic-api/docs/api.md) | [Design](../services/turkic-api/DESIGN.md)

---

## transcript-api

YouTube video transcription service.

**Features:**
- Extract transcripts from YouTube videos
- Support for auto-generated captions
- Multiple language support
- Timestamp extraction

**Key Endpoints:**
```
POST /transcripts    Get transcript for YouTube URL
```

**Start:**
```bash
make up-transcript
```

**Docs:** [README](../services/transcript-api/README.md) | [API](../services/transcript-api/docs/api.md)

---

## qr-api

QR code generation service.

**Features:**
- Generate QR codes from text/URLs
- Customizable size and error correction
- Multiple output formats (PNG, SVG)

**Key Endpoints:**
```
POST /generate       Generate QR code
```

**Start:**
```bash
make up-qr
```

**Docs:** [README](../services/qr-api/README.md) | [API](../services/qr-api/docs/api.md)

---

## music-wrapped-api

Music listening analytics aggregating data from streaming services.

**Features:**
- Spotify listening history analysis
- Apple Music integration
- Last.fm scrobble aggregation
- Yearly wrapped-style reports

**Key Endpoints:**
```
POST /connect/{service}  Connect streaming service
GET  /stats              Get listening statistics
GET  /wrapped/{year}     Generate yearly wrapped
```

**Start:**
```bash
make up-music
```

**Docs:** [README](../services/music-wrapped-api/README.md)

---

## Clients

### DiscordBot

Discord bot integrating all platform services.

**Features:**
- Slash commands for all services
- Real-time job progress updates
- Rich embeds for results
- Redis pub/sub event subscription

**Start:**
```bash
make up-discord
```

**Docs:** [README](../clients/DiscordBot/README.md)
