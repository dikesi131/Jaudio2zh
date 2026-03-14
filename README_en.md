# Jaudio2zh

A local Python command-line tool for:
- Transcribing Japanese audio using `openai-whisper`
- Translating subtitles to Chinese via a local `llama.cpp` OpenAI API
- Exporting subtitle files (`.lrc`, `.srt`, `.txt`)
- Writing runtime logs to a log file

## Features
- **Fully Local Inference** (Whisper local model + Local API Server), ensuring privacy and security.
- **Performance-optimized defaults for Apple Silicon CPU/MPS** (also supports Windows CUDA).
- **Parallel translation requests** to improve API throughput.
- **Detailed logging** output to both console and file.

## Requirements
- Python 3.10+
- `ffmpeg` installed and available in system PATH (recommended)
- Local model files prepared:
  - `openai-whisper` model name (e.g., `medium`) or local `.pt` file
  - `sakura` model loaded by a `llama.cpp` OpenAI-compatible server

## Whisper Model Download
Official Download Links: [whisper/whisper/__init__.py](https://github.com/openai/whisper/blob/main/whisper/__init__.py)

```python
_MODELS = {
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large-v1": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
    "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "large-v3": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
    "large-v3-turbo": "https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt",
    "turbo": "https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt",
}
```
You can choose and download the appropriate model based on your computer configuration.

## Translation Server Deployment (Llama Model Deployment)

This project relies on a local OpenAI-compatible API service provided by `llama.cpp`.

### 1. Install llama.cpp

#### macOS (Recommended)
Install using Homebrew:
```bash
brew install llama.cpp
```
Check binaries:
```bash
which llama-server
llama-server --version
```

#### Windows
Windows users can download pre-compiled binaries or install via pip (functionality may be limited):
1. Go to [llama.cpp Releases](https://github.com/ggerganov/llama.cpp/releases) and download the archive containing `llama-server.exe`.
2. Extract and add the path to the system Environment Variables PATH.
3. Alternatively, use Python bindings (requires compilation environment):
   ```bash
   pip install llama-cpp-python
   ```

### 2. Download GGUF Translation Model

You need a model file in `.gguf` format (safetensors are not supported).
It is recommended to use the `huggingface-cli` tool for downloading.

**Install Tool:**
```bash
pip install -U "huggingface_hub[cli]"
```

**Recommended Model Directory Structure:**
For ease of management, it is recommended to establish a standard model directory structure in the project root:
```text
Jaudio2zh/
├── models/
│   ├── llm/                # Store translation models
│   │   └── sakura-7b/      # Example: Sakura model directory
│   └── whisper/            # Store Whisper models (optional, defaults to auto-download)
├── outputs/                # Output directory
└── ...
```

**Download Command Example:**
The following command downloads the Sakura model to the recommended `models/llm/sakura-7b` directory:

```bash
# Create directory
mkdir -p models/llm/sakura-7b

# Download model (Example: Sakura 7B Qwen2.5)
huggingface-cli download \
  SakuraLLM/sakura-7b-qwen2.5-v1.0-GGUF \
  sakura-7b-qwen2.5-v1.0-iq4xs.gguf \
  --local-dir ./models/llm/sakura-7b
```

**Verify File Existence:**
```bash
# macOS/Linux
ls -lh ./models/llm/sakura-7b/*.gguf

# Windows (PowerShell)
Get-ChildItem ./models/llm/sakura-7b/*.gguf
```

**Manual Download Link:** [SakuraLLM](https://huggingface.co/SakuraLLM)

### 3. Start Local OpenAI API Server

Use `llama-server` to load the model, specifying host, port, and alias.

**Basic Start Command:**
```bash
llama-server \
  -m ./models/llm/sakura-7b/sakura-7b-qwen2.5-v1.0-iq4xs.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --alias sakura
```

**Apple Silicon (M1/M2/M3/M4) Optimization Flags:**
- `-ngl 999`: Offload as many layers as possible to GPU (Metal).
- `-c 8192`: Increase context length when needed.
- `-t 8`: Set CPU thread count (adjust based on your machine cores).

**Optimized Start Example:**
```bash
llama-server \
  -m ./models/llm/sakura-7b/sakura-7b-qwen2.5-v1.0-iq4xs.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --alias sakura \
  -ngl 999 \
  -c 8192 \
  -t 8
```

**Note for Windows Users:**
If running in CMD or PowerShell, backslashes `\` may need to be changed to `^` (CMD) or `` ` `` (PowerShell), or write the command on a single line.
```powershell
# PowerShell Example
llama-server -m ./models/llm/sakura-7b/sakura-7b-qwen2.5-v1.0-iq4xs.gguf --host 127.0.0.1 --port 8080 --alias sakura -ngl 999
```

### 4. Verify Server is Ready

**Check Model List Endpoint:**
```bash
curl -sS http://127.0.0.1:8080/v1/models
```

**Quick Chat Completion Test:**
```bash
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "model": "sakura",
     "messages": [
      { "role": "system", "content": "You are a translator."},
      { "role": "user", "content": "Translate to Chinese: こんにちは"}
    ],
     "temperature": 0
  }'
```
If it returns JSON containing `choices`, the server is ready for `jaudio2zh`.

## Install Jaudio2zh

It is recommended to install using a virtual environment like conda.

```bash
cd Jaudio2zh/
conda create -n Jaudio2zh python=3.11
conda activate Jaudio2zh
pip install -e .
```

## Usage

```bash
# --sakura-api-base and --sakura-model are optional
# --whisper-model can specify a local .pt model file

jaudio2zh \
  --input /path/to/audio.wav \
  --whisper-model medium \
  --sakura-api-base http://127.0.0.1:8080 \
  --sakura-model sakura \
  --formats lrc,srt
```

**Batch recursively process all audio files in a directory:**
```bash
jaudio2zh \
  --batch-input-dir /path/to/audios \
  --whisper-model medium \
  --sakura-api-base http://127.0.0.1:8080 \
  --sakura-model sakura \
  --formats lrc
```

### Common Options

| Option              | Description                                                  |
| :------------------ | :----------------------------------------------------------- |
| `--device`          | Select device: `auto`, `cpu`, `mps` (Mac), `cuda` (Windows)  |
| `--batch-size`      | Translation API concurrent request count (Recommended 2-8)   |
| `--no-translate`    | Skip the Japanese-to-Chinese translation step                |
| `--sakura-api-base` | Local translation server address (Default `http://127.0.0.1:8080`) |
| `--sakura-model`    | Model alias registered in the server (Must match `--alias`)  |
| `--request-timeout` | API request timeout (seconds)                                |
| `--language`        | Source audio language (Default `ja`)                         |
| `--log-dir`         | Log storage directory (Default `./logs`)                     |
| `--formats`         | Output formats, comma separated (e.g., `lrc,srt,txt`)        |
| `--output-dir`      | Optional output directory; defaults to input audio directory if not passed |
| `--batch-input-dir` | Recursively process all audio/video files in directory       |

## Output Files

For each input file, the following files will be created:
- Default save location is the same directory as the input file (e.g., Input `/user/audios/audio1.mp3`, Output `/user/audios/audio1.lrc`)
- If `--output-dir` is specified, outputs go to the specified directory.

- `<name>.lrc`: LRC format subtitles (with timestamps)
- `<name>.srt`: SRT format subtitles (with timestamps)
- `<name>.txt`: Plain text translation (with timestamps)
- `<name>.ja.txt`: Japanese original text only (no timestamps)
- `<name>.segments.jsonl`: Segmentation data for debugging/reference

Log files will be written to the `--log-dir` directory, e.g.:
- `logs/run_20260305_120000.log`

## Notes on Translation Quality

Currently, the tool uses sentence-by-sentence translation. However, since Japanese is a highly context-dependent language, pronoun translation effects are not ideal. For the Sakura model, **full-text translation yields better results than sentence-by-sentence**. However, full-text translation may cause errors or result in the model repeating content endlessly. To ensure stability, the tool uses sentence-by-sentence translation by default.

**If you care more about translation quality, you can use the following workflow:**

1. **Transcribe subtitles only first:**
   ```bash
   jaudio2zh \
     --input /path/to/audio.wav \
     --output-dir ./outputs \
     --whisper-model medium \
     --sakura-api-base http://127.0.0.1:8080 \
     --sakura-model sakura \
     --formats lrc,srt \
     --no-translate
   ```

2. **This will generate two main files:**
   1. Japanese original text file (without timestamps)
   2. Translated subtitle file (lrc/srt, with timestamps)

3. **Use Sakura UI for Full Text Translation:**
   Access the Sakura UI interface via browser. Upload the Japanese original text for full-text translation to get a clean Chinese text file without timestamps.

4. **Merge Text and Timeline:**
   Combine the translated text file and the timeline file to get the final subtitle file. This can be achieved using `combine_text_and_timeline.py`:
   ```bash
   python combine_text_and_timeline.py \
     --timeline-file ./outputs/05.MakingCocktails.lrc \
     --zh-text-file ./outputs/ToolTranslation.txt \
     --output-file ./outputs/ToolTranslation.lrc \
     --allow-mismatch
   ```

**Overall Accuracy:**
Translation accuracy depends on both transcription and translation. If there is no official Japanese original text, accuracy is around **70-80%**. If there is an official Japanese original text, using only the translation step can achieve accuracy around **90%**.

## Common Deployment Issues

| Issue                    | Solution                                                     |
| :----------------------- | :----------------------------------------------------------- |
| `connection refused`     | `llama-server` is not running, or host/port configuration is incorrect |
| `model not found`        | `--sakura-model` argument does not match `--alias` at server startup |
| Very slow first response | Model warm-up is normal; subsequent requests will be faster  |
| OOM / Memory Pressure    | Use a smaller GGUF quantization version (e.g., `Q4`), reduce context (`-c`), or lower GPU offload layers (`-ngl`) |
| Path Error (Windows)     | Ensure paths use forward slashes `/` or escaped backslashes `\\` |

## Actual Test Results

**Configuration:** MacOS 26.3.1, 16GB Memory, M4
**Models Used:**
- Transcription: `whisper-large-v3`
- Translation: `sakura-14b-qwen2.5`

![image-20260306210509282](https://dikkksi-wiki-pic.oss-cn-chengdu.aliyuncs.com/wiki_img_2/20260306211046636.png)