# Illustrator 📚✨

A powerful LangGraph application that analyzes manuscript chapters using advanced NLP and emotional resonance detection, then generates optimized illustration prompts for DALL-E, Google Imagen, and Replicate-hosted models including Flux 1.1 Pro and Seedream 4.

[![LangGraph](https://img.shields.io/badge/Built%20with-LangGraph-00324d.svg)](https://langchain-ai.github.io/langgraph/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## ✨ Features

- **🤖 GPT-OSS Native Analysis**: Ships with OpenAI's open-weight `gpt-oss-120b` (Harmony format) via HuggingFace for high-quality reasoning
- **📖 Chapter-by-Chapter Analysis**: Process manuscripts one chapter at a time with detailed emotional and thematic breakdowns
- **🧠 Emotional Resonance Engine**: Surface the most emotionally impactful moments (defaults to 10 per chapter)
- **🎨 Multi-Provider Support**: Generate illustrations using DALL-E 3, Vertex AI Imagen (Imagen 3), Flux 1.1 Pro (remote endpoint or local diffusers pipeline), or Seedream 4 via Replicate
- **🔄 Interactive CLI**: User-friendly command-line interface with CTRL+D input handling
- **💾 Persistent Storage**: Save analysis results and generated images for future reference
- **⚙️ Configurable Styles**: Customize artistic styles, color palettes, and creative influences

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- **Database**: MongoDB 5+ (local or managed). Set `MONGODB_URI` (or `MONGO_URL`) and `MONGO_DB_NAME` to point at your instance. For local development and automated tests you can enable the embedded mock driver by setting `MONGO_USE_MOCK=1`.
- **LLM access**:
  - HuggingFace API key (default `openai/gpt-oss-120b` reasoning model)
  - Optional: Anthropic API key if you prefer Claude instead of gpt-oss
- **Image generation providers** (pick the ones you plan to use):
  - **DALL-E**: OpenAI API key
  - **Imagen (Vertex AI)**: Google Cloud credentials and project ID, _or_ a Replicate API token
  - **Flux 1.1 Pro**: HuggingFace API key, _or_ a Replicate API token
  - **Seedream 4**: Replicate API token

### Installation

1. **Clone and setup**:
```bash
git clone https://github.com/and-other-tales/illustrator.git
cd illustrator
pip install -e .
```

2. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. **Run the application**:
```bash
illustrator
```

### Environment Variables

Create a `.env` file with the following variables:

```env
# Default LLM: gpt-oss-120b on HuggingFace (Harmony format)
HUGGINGFACE_API_KEY=your-huggingface-key
DEFAULT_LLM_PROVIDER=huggingface
DEFAULT_LLM_MODEL=openai/gpt-oss-120b

# Optional Claude fallback
ANTHROPIC_API_KEY=your-anthropic-key

# For DALL-E (OpenAI)
OPENAI_API_KEY=your-openai-key

# For Vertex AI Imagen (Google Cloud)
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
GOOGLE_PROJECT_ID=your-project-id

# For Flux (HuggingFace)
HUGGINGFACE_FLUX_ENDPOINT_URL=https://api.endpoints.huggingface.cloud/your-org/flux

# Flux local pipeline (diffusers)
#FLUX_USE_PIPELINE=true
#FLUX_PIPELINE_MODEL_ID=black-forest-labs/FLUX.1-dev
#FLUX_PIPELINE_DEVICE=cuda
#FLUX_PIPELINE_DTYPE=bfloat16

# For Replicate-hosted models (Flux, Imagen 4, Seedream 4)
REPLICATE_API_TOKEN=your-replicate-token

# Optional: Customize defaults
DEFAULT_IMAGE_PROVIDER=dalle  # dalle, imagen4, flux, or seedream
DEFAULT_ILLUSTRATION_STYLE=digital_painting

# Database
MONGODB_URI=mongodb://localhost:27017
MONGO_URL=mongodb://localhost:27017
MONGO_DB_NAME=illustrator
MONGO_USE_MOCK=false
```

> The app automatically renders prompts and parses responses using OpenAI's [Harmony format](https://github.com/openai/harmony) whenever a gpt-oss model is selected.

## 💡 How It Works

### 1. **Manuscript Input**
- Interactive chapter input with title and content
- CTRL+D to finish entering chapter text
- Support for large text blocks via copy/paste

### 2. **Emotional Analysis**
- gpt-oss-120b (Harmony format) analyzes text for emotional peaks and valleys
- Pattern matching for emotional intensity
- Identification of up to 10 most resonant moments per chapter (scene-aware by default)

### 3. **Visual Scene Detection**
- Extracts visually compelling narrative moments
- Analyzes setting, atmosphere, and character emotions
- Identifies illustration opportunities

### 4. **Prompt Generation**
- Provider-specific optimization (DALL-E vs Vertex AI Imagen vs Flux)
- Style-aware prompt engineering
- Emotional tone translation to visual elements

### 5. **Image Generation** (Optional)
- Generate actual illustrations using selected provider
- Multiple style variations
- High-quality output with metadata

## 🖥️ Flux Local Pipeline (Diffusers)

- Set `FLUX_USE_PIPELINE=true` in your environment (or `.env`) to route Flux jobs through a local [diffusers](https://github.com/huggingface/diffusers) pipeline instead of a remote HuggingFace endpoint.
- Accept the [`FLUX.1 [dev]`](https://huggingface.co/black-forest-labs/FLUX.1-dev) license and run `huggingface-cli login` so the weights and README-gated metadata can be downloaded locally.
- Install the Hugging Face reference stack exactly as shown in the model card: `pip install -U diffusers accelerate`.
- The pipeline loader mirrors the documentation example, bootstrapping `FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)` and calling `enable_model_cpu_offload()` on CUDA devices to keep VRAM usage in check.
- At runtime we default to the documented sampling settings—`guidance_scale=3.5`, `num_inference_steps=50`, `height=1024`, `width=1024`, and `max_sequence_length=512`—while still letting individual prompts override them through their `technical_params` payload.
- Optional knobs:
  - `FLUX_PIPELINE_DEVICE` (e.g. `cuda`, `cuda:0`, `mps`, `cpu`)
  - `FLUX_PIPELINE_DTYPE` (`bfloat16`, `float16`, `float32`, or `auto`)
  - `FLUX_PIPELINE_VARIANT` / `FLUX_PIPELINE_REVISION` to pin a specific bundle from the model card
- The app falls back to remote HuggingFace or Replicate endpoints automatically if the pipeline is disabled or cannot be initialised.

## 🎨 Supported Image Providers

| Provider | Strengths | Best For |
|----------|-----------|----------|
| **DALL-E (gpt-image-1)** | High-quality, coherent images | Character-focused scenes, detailed illustrations |
| **Imagen (Vertex AI Imagen 3)** | Photorealistic output, fine control | Realistic settings, atmospheric scenes |
| **Flux 1.1 Pro** | Artistic styles, creative interpretation | Stylized art, concept illustrations |

> Flux 1.1 Pro can run through HuggingFace Inference Endpoints or Replicate. Configure `HUGGINGFACE_API_KEY` and `HUGGINGFACE_FLUX_ENDPOINT_URL` for self-hosted endpoints, or set `REPLICATE_API_TOKEN` to call the managed Replicate model. If you omit both, the app defaults to the hosted HuggingFace endpoint at `https://qj029p0ofvfmjxus.us-east-1.aws.endpoints.huggingface.cloud`.

## 📋 Example Usage

```bash
$ illustrator

📚 Manuscript Illustrator ✨
╭────────────────────────────────────────────╮
│                                            │
│ Analyze your manuscript chapters and       │
│ generate AI illustrations                  │
│                                            │
│ • Enter chapter content with CTRL+D       │
│ • Choose from DALL-E, Imagen, Flux, or Seedream    │
│ • Get emotional analysis and prompts      │
│                                            │
╰────────────────────────────────────────────╯

📖 Manuscript Information
Manuscript title: The Midnight Garden
Author name: Jane Smith
Genre: Fantasy

🎨 Style Preferences
Available image providers:
  1. DALL-E 3 (OpenAI)
  2. Imagen (Google Vertex AI)
  3. Flux 1.1 Pro (HuggingFace/Replicate)
  4. Seedream 4 (Replicate)

Select image provider [1]: 1
Preferred art style [digital painting]: watercolor
Color palette preference: muted pastels
Artistic influences: Studio Ghibli

📝 Chapter 1
Chapter 1 title: The Secret Door
Enter chapter content (press CTRL+D when finished):

[Paste your chapter content here]
^D

✅ Chapter captured: 2,847 words
Add another chapter? [Y/n]: n

🔄 Processing 1 chapters...
✅ Chapter 1: Analysis complete

✨ Processing complete! Analyzed 1 chapters.
```

## 📁 Output Structure

```
illustrator_output/
└── The_Midnight_Garden/
    ├── manuscript_analysis.json    # Complete analysis data
    ├── chapter_1_analysis.json     # Detailed chapter breakdown
    ├── prompts/
    │   ├── chapter_1_moment_1.txt
    │   ├── chapter_1_moment_2.txt
    │   └── chapter_1_moment_3.txt
    └── images/                     # Generated illustrations (if enabled)
        ├── chapter_1_moment_1.png
        ├── chapter_1_moment_2.png
        └── chapter_1_moment_3.png
```

## 🔧 Advanced Usage

### Batch Processing

```bash
illustrator --batch --config-file my_config.json
```

### Custom Configuration

Create a `config.json` file:

```json
{
  "image_provider": "flux",
  "art_style": "oil painting",
  "color_palette": "warm earth tones",
  "max_emotional_moments": 10,
  "min_intensity_threshold": 0.7,
  "generate_images": true,
  "save_analysis": true
}
```

## 🏗️ Architecture

Built using **LangGraph** for robust workflow orchestration:

- **State Management**: Track chapters, analyses, and user preferences
- **Node-based Processing**: Modular analysis and generation steps
- **Error Handling**: Graceful recovery and retry logic
- **Persistence**: Store intermediate results and user data
- **Streaming**: Real-time processing updates

## 📊 Analysis Output

For each chapter, you'll receive:

- **Emotional Moments**: up to 10 highest intensity passages with emotional tone analysis (scene-aware)

## 🧭 CLI Options Highlights

- `--max-moments` to set the target number of moments per chapter (default: 10)
- `--comprehensive/--standard` to enable/disable scene-aware comprehensive analysis (default: comprehensive)
- **Themes**: Dominant literary themes and motifs
- **Setting Description**: Atmospheric and visual setting details
- **Character Emotions**: Emotional arcs for key characters
- **Illustration Prompts**: Optimized prompts for your chosen AI provider

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangGraph](https://langchain-ai.github.io/langgraph/) by LangChain
- Powered by `gpt-oss-120b` (Harmony) for literary analysis, with optional Claude fallback
- Supports [DALL-E](https://openai.com/dall-e-3), [Imagen](https://cloud.google.com/vertex-ai), [Flux 1.1 Pro](https://replicate.com/black-forest-labs/flux-1.1-pro), and [Seedream 4](https://replicate.com/bytedance/seedream-4)

---

**Author**: David James Lennon
**Email**: david@othertales.co
**Project**: [And Other Tales](https://othertales.co)
