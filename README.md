# Jaudio2zh

一个本地 Python 命令行工具，用于：
- 使用 `openai-whisper` 转录日语音频
- 通过本地 `llama.cpp` OpenAI API 将字幕翻译为中文
- 导出字幕文件（`.lrc`, `.srt`, `.txt`）
- 将运行时日志写入日志文件

## 功能特性
- **完全本地推理**（Whisper 本地模型 + 本地 API 服务器），保障隐私与安全
- **针对 Apple Silicon CPU/MPS 优化的性能默认值**（同时支持 Windows CUDA）
- **并行翻译请求**，提高 API 吞吐量
- **详细的日志记录**，同时输出到控制台和文件

## 系统要求
- Python 3.10+
- `ffmpeg` 已安装并可用（建议在系统 PATH 中）
- 准备好以下本地模型文件：
  - `openai-whisper` 模型名称（例如 `medium`）或本地 `.pt` 文件
  - 由 `llama.cpp` OpenAI 兼容服务器加载的 `sakura` 模型

---

## whisper模型下载

官方下载地址：[whisper/whisper/__init__.py at main · openai/whisper](https://github.com/openai/whisper/blob/main/whisper/__init__.py)

```json
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

可自行根据电脑配置选择合适的模型下载

## 翻译服务器部署 (Llama Model Deployment)

本项目依赖由 `llama.cpp` 提供的本地 OpenAI 兼容 API 服务。

### 1. 安装 llama.cpp

#### macOS (推荐)
使用 Homebrew 安装：
```bash
brew install llama.cpp
```
检查二进制文件：
```bash
which llama-server
llama-server --version
```

#### Windows
Windows 用户可下载预编译的二进制文件，或通过 pip 安装（功能可能受限）：
1. 前往 [llama.cpp Releases](https://github.com/ggerganov/llama.cpp/releases) 下载包含 `llama-server.exe` 的压缩包。
2. 解压并将路径添加到系统环境变量 PATH 中。
3. 或者使用 Python 绑定（需编译环境）：
   ```bash
   pip install llama-cpp-python
   ```

### 2. 下载 GGUF 翻译模型

你需要一个 `.gguf` 格式的模型文件（不支持 safetensors）。
推荐使用 `huggingface-cli` 工具下载。

**安装工具：**
```bash
pip install -U "huggingface_hub[cli]"
```

**模型目录结构建议：**
为了便于管理，建议在项目根目录下建立标准的模型目录结构：
```text
Jaudio2zh/
├── models/
│   ├── llm/                # 存放翻译模型
│   │   └── sakura-7b/      # 示例：Sakura 模型目录
│   └── whisper/            # 存放 Whisper 模型 (可选，默认自动下载)
├── outputs/                # 输出目录
└── ...
```

**下载命令示例：**
以下命令将 Sakura 模型下载到推荐的 `models/llm/sakura-7b` 目录：

```bash
# 创建目录
mkdir -p models/llm/sakura-7b

# 下载模型 (以 Sakura 7B Qwen2.5 为例)
huggingface-cli download \
  SakuraLLM/sakura-7b-qwen2.5-v1.0-GGUF \
  sakura-7b-qwen2.5-v1.0-iq4xs.gguf \
  --local-dir ./models/llm/sakura-7b
```

**验证文件是否存在：**
```bash
# macOS/Linux
ls -lh ./models/llm/sakura-7b/*.gguf

# Windows (PowerShell)
Get-ChildItem ./models/llm/sakura-7b/*.gguf
```

手动下载地址：[SakuraLLM (SakuraLLM)](https://huggingface.co/SakuraLLM)

### 3. 启动本地 OpenAI API 服务器

使用 `llama-server` 加载模型，并指定主机、端口和别名。

**基本启动命令：**
```bash
llama-server \
  -m ./models/llm/sakura-7b/sakura-7b-qwen2.5-v1.0-iq4xs.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --alias sakura
```

**Apple Silicon (M1/M2/M3/M4) 优化参数：**
- `-ngl 999`: 尽可能将更多层卸载到 GPU (Metal)。
- `-c 8192`: 需要时增加上下文长度。
- `-t 8`: 设置 CPU 线程数（根据你的机器核心数调整）。

**优化后的启动示例：**
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

**Windows 用户注意：**
如果在 CMD 或 PowerShell 中运行，反斜杠 `\` 可能需要改为 `^` (CMD) 或 `` ` `` (PowerShell)，或者将命令写在一行。
```powershell
# PowerShell 示例
llama-server -m ./models/llm/sakura-7b/sakura-7b-qwen2.5-v1.0-iq4xs.gguf --host 127.0.0.1 --port 8080 --alias sakura -ngl 999
```

### 4. 验证服务器是否就绪

**检查模型列表端点：**
```bash
curl -sS http://127.0.0.1:8080/v1/models
```

**快速聊天完成测试：**
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
如果返回包含 `choices` 的 JSON，则服务器已准备好供 `jaudio2zh` 使用。

---

## 安装 Jaudio2zh

建议使用conda等虚拟环境安装

```bash
cd Jaudio2zh/
conda create -n Jaudio2zh python=3.11
conda activate Jaudio2zh
pip install -e .
```

## 使用方法

```bash
// --sakura-api-base和--sakura-model为可选项
// --whiser-model可指定本地pt模型文件

jaudio2zh \
  --input /path/to/audio.wav \
  --output-dir ./outputs \
  --whisper-model medium \
  --sakura-api-base http://127.0.0.1:8080 \
  --sakura-model sakura \
  --formats lrc,srt
```

### 常用选项

| 选项 | 说明 |
| :--- | :--- |
| `--device` | 选择设备：`auto`, `cpu`, `mps` (Mac), `cuda` (Windows) |
| `--batch-size` | 翻译 API 并发请求数 (建议 2-8) |
| `--no-translate` | 跳过日语到中文的翻译步骤 |
| `--sakura-api-base` | 本地翻译服务器地址 (默认 `http://127.0.0.1:8080`) |
| `--sakura-model` | 服务器中注册的模型别名 (需与 `--alias` 一致) |
| `--request-timeout` | API 请求超时时间 (秒) |
| `--language` | 源音频语言 (默认 `ja`) |
| `--log-dir` | 日志存储目录 (默认 `./logs`) |
| `--formats` | 输出格式，逗号分隔 (e.g., `lrc,srt,txt`) |

---

## 输出文件

对于每个输入文件，将在 `--output-dir` 目录中创建以下文件：

- `<name>.lrc` : LRC 格式字幕 (含时间轴)
- `<name>.srt` : SRT 格式字幕 (含时间轴)
- `<name>.txt` : 纯文本翻译 (含时间轴)
- `<name>.ja.txt` : 仅日语原文 (无时间轴)
- `<name>.segments.jsonl` : 调试/参考用的分段数据

日志文件将写入 `--log-dir` 目录，例如：
- `logs/run_20260305_120000.log`

---

## 关于翻译效果的说明

目前工具使用的是一句一句翻译，但由于日语是一个语境强相关的语言，所以关于代词翻译的效果并不理想，目前对于sakura模型来说，全文翻译效果要好于逐句翻译，但由于全文翻译可能会报错或出现模型一直重复输出相同内容的问题，所以为保证稳定性，工具翻译使用的是诸句翻译，如果你更在乎翻译效果，可以使用如下方案：

```sh
// 首先仅转录出字幕文件
jaudio2zh \
  --input /path/to/audio.wav \
  --output-dir ./outputs \
  --whisper-model medium \
  --sakura-api-base http://127.0.0.1:8080 \
  --sakura-model sakura \
  --formats lrc,srt
  --no-translate
  
// 这会得到两个主要文件
1. 日语原文文件(不包含时间轴)
2. 翻译后的字幕文件(lrc/srt, 包含时间轴)

// 随后浏览器访问sakura UI界面
将日语原文上传进行全文翻译, 得到不包含时间轴的纯净中文文本文件

// 将文本翻译文件和时间轴合并得到最后的字幕文件, 可以使用combine_text_and_timeline.py实现
python combine_text_and_timeline.py --timeline-file ./outputs/05.制作鸡尾酒与一起品酒.lrc --zh-text-file ./outputs/工具翻译.txt --output-file ./outputs/工具翻译.lrc --allow-mism
atch
```

整体来讲，翻译的准确性取决于转录和翻译两部分，如果不存在官方日语原文，准确率在70-80%左右，如果有官方日语原文，那么仅翻译一方面，准确性可以达到90%左右

## 常见部署问题

| 问题 | 解决方案 |
| :--- | :--- |
| `connection refused` | `llama-server` 未运行，或主机/端口配置错误 |
| `model not found` | `--sakura-model` 参数与服务器启动时的 `--alias` 不匹配 |
| 首次响应非常慢 | 模型预热是正常现象；后续请求会更快 |
| OOM / 内存压力 | 使用更小的 GGUF 量化版本 (如 `Q4`)，减小上下文 (`-c`)，或降低 GPU 卸载层数 (`-ngl`) |
| 路径错误 (Windows) | 确保路径使用正斜杠 `/` 或转义的反斜杠 `\\` |

## 实际测试效果

配置：Macos 26.3.1 16GB内存 M4

模型选用：

转录：whisper-large-v3

翻译：sakura-14b-qwen2.5

![image-20260306210509282](https://dikkksi-wiki-pic.oss-cn-chengdu.aliyuncs.com/wiki_img_2/20260306211046636.png)