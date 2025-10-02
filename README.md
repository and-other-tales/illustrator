# Othertales Illustrator (Beta) 📚✨

[![LangGraph](https://img.shields.io/badge/Built%20with-LangGraph-00324d.svg)](https://langchain-ai.github.io/langgraph/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Othertales%20EULA-red.svg)](LICENSE.md)

Othertales Illustrator (Beta) is the manuscript-to-illustration workflow developed by PI & Other Tales, Inc. It analyzes long-form fiction with LangGraph-powered agents, extracts the most evocative scenes, and engineers provider-specific prompts for state-of-the-art image models.

Maintained by **David James Lennon** (PI & Other Tales, Inc.) · david@othertales.co · [GitHub repository](https://github.com/and-other-tales/illustrator)

---

## ✨ Feature Highlights

- **LangGraph orchestration** – resilient, node-based pipelines for manuscript ingestion, emotional analysis, prompt engineering, generation, and persistence
- **GPT-OSS & Anthropic capable** – defaults to `openai/gpt-oss-120b` via Hugging Face with automatic fallbacks to Anthropic (Claude or Vertex) when keys are present
- **Multi-provider imagery** – turnkey support for DALL·E (gpt-image-1), Google Imagen4 (Vertex AI or Replicate), Flux 1.1 Pro (Hugging Face Endpoint or Replicate), local Flux diffusers pipelines, and Seedream 4
- **Narrative-first prompts** – scene-aware emotional scoring, style preferences, and automatic quality feedback tailor prompts to each provider’s strengths
- **Persistent storage** – MongoDB-first design with automatic `mongomock` fallback for local tests, plus resumable saved manuscripts and checkpointing
- **CLI + Web UI** – rich terminal experience, batch/interactive modes, and an optional FastAPI/UVicorn experience with live progress updates over WebSockets

---

## 🚀 Quick Start

### Prerequisites

- Python **3.11+** and `pip`
- MongoDB 5.x or compatible Atlas cluster (set `MONGO_USE_MOCK=1` to fall back to in-memory `mongomock` for development/tests)
- API credentials for the providers you intend to use (see [Configuration](#configuration))
- CUDA/MPS GPU recommended for local Flux diffusers pipeline (CPU is supported but slow)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/and-other-tales/illustrator.git
cd illustrator

# 2. (Recommended) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install the CLI runtime
pip install -e .

# 4. Optional extras
# Web UI + API server
pip install -e '.[web]'
# Developer toolchain (linting, tests, type checking)
pip install -e '.[dev]'
```

> The package entry point installs an `illustrator` console script that exposes all CLI and web commands.

### Configuration

Create a `.env` file (or export the equivalent environment variables) to point the application at your services:

```env
# Core manuscript analysis LLM (defaults to gpt-oss-120b via Hugging Face)
HUGGINGFACE_API_KEY=your-huggingface-token
DEFAULT_LLM_PROVIDER=huggingface
DEFAULT_LLM_MODEL=openai/gpt-oss-120b

# Optional Claude fallback
ANTHROPIC_API_KEY=sk-ant-...

# MongoDB connection details
MONGODB_URI=mongodb://localhost:27017
MONGO_URL=mongodb://localhost:27017
MONGO_DB_NAME=illustrator
MONGO_USE_MOCK=false  # set to true/1 to force mongomock

# DALL·E (OpenAI)
OPENAI_API_KEY=sk-...

# Google Imagen4 / Anthropic Vertex
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json  # or inline JSON
GOOGLE_PROJECT_ID=your-gcp-project-id

# Flux (Hugging Face endpoint)
HUGGINGFACE_FLUX_ENDPOINT_URL=https://api.endpoints.huggingface.cloud/your-org/flux

# Flux local diffusers pipeline
FLUX_USE_PIPELINE=false           # set true to prefer local pipeline
FLUX_PIPELINE_MODEL_ID=black-forest-labs/FLUX.1-dev
FLUX_PIPELINE_DEVICE=cuda         # cuda, cuda:0, mps, or cpu
FLUX_PIPELINE_DTYPE=bfloat16

# Flux Dev Vertex (private preview)
FLUX_DEV_VERTEX_ENDPOINT_URL=https://us-central1-aiplatform.googleapis.com/v1/projects/your-project/locations/us-central1/publishers/black-forest-labs/models/flux-1.1-dev

# Replicate-hosted models (Flux, Imagen4 proxy, Seedream)
REPLICATE_API_TOKEN=r8_...

# Default illustration preferences
DEFAULT_IMAGE_PROVIDER=dalle      # dalle | imagen4 | flux | flux_dev_vertex | seedream
DEFAULT_ILLUSTRATION_STYLE=digital_painting
```

Key points:
- `GOOGLE_APPLICATION_CREDENTIALS` accepts either a file path or raw JSON service-account content.
- Imagen4 requires both `GOOGLE_APPLICATION_CREDENTIALS` and `GOOGLE_PROJECT_ID`, or a `REPLICATE_API_TOKEN` for the hosted Imagen 4 model.
- Flux requires a Hugging Face token (for endpoints or diffusers downloads) or a Replicate token.

### ▶️ Run the CLI

Interactive manuscript walkthrough:

```bash
illustrator analyze
```

Batch mode (reads configuration & chapters from JSON):

```bash
illustrator analyze --batch --config-file my_run_config.json
```

List and reload saved manuscripts:

```bash
illustrator analyze --list-saved
illustrator analyze --load saved_manuscripts/my_story.json
```

### 🌐 Run the Web Experience

Serve the FastAPI API + embedded web client:

```bash
illustrator start --host 0.0.0.0 --port 8080
```

Run just the API (useful for Cloud Run / hosted deployments):

```bash
illustrator web-server --host 0.0.0.0 --port 8080
```

Run only the web client and point it at a remote API server:

```bash
illustrator web-client --server-url https://illustrator-api.example.com
```

---

## 🖼️ Supported Image Providers

| Provider | Requirements | Strengths |
|----------|--------------|-----------|
| **DALL·E (gpt-image-1)** | `OPENAI_API_KEY` | Cohesive characters, illustrative detail |
| **Imagen4 (Vertex AI)** | `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_PROJECT_ID` | Cinematic, photoreal scenes |
| **Imagen4 (Replicate)** | `REPLICATE_API_TOKEN` | Imagen4 without managing GCP infra |
| **Flux 1.1 Pro (HF Endpoint)** | `HUGGINGFACE_API_KEY` (+ optional custom endpoint URL) | Stylized, painterly art |
| **Flux 1.1 Pro (Diffusers)** | Local GPU, `FLUX_USE_PIPELINE=true` | On-device control & privacy |
| **Flux Dev (Vertex AI)** | `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_PROJECT_ID`, optional `FLUX_DEV_VERTEX_ENDPOINT_URL` | Latest Vertex-hosted Flux Dev models |
| **Flux 1.1 Pro (Replicate)** | `REPLICATE_API_TOKEN` | Flux without infrastructure |
| **Seedream 4 (Replicate)** | `REPLICATE_API_TOKEN` | Dreamlike, imaginative renders |

The CLI automatically validates credentials for the selected provider and guides you toward any missing keys.

---

## 🧠 Workflow Overview

1. **Chapter ingestion** – paste chapters or load saved manuscripts (CTRL+D ends input)
2. **Emotional resonance scoring** – extracts up to 10 high-impact moments per chapter with intensity, characters, themes, and setting context
3. **Prompt engineering** – provider-specific prompt rewriting with stylistic preferences and technical parameters
4. **Image generation (optional)** – orchestrates async calls to DALL·E, Imagen4, Flux, or Seedream providers, including retries and safety fallbacks
5. **Persistence** – stores analyses, prompts, and image metadata under `illustrator_output/<manuscript_slug>/`

### Output Layout

```
illustrator_output/
└── The_Midnight_Garden/
    ├── manuscript_analysis.json
    ├── chapter_1_analysis.json
    ├── prompts/
    │   ├── chapter_1_moment_1.txt
    │   └── ...
    └── images/
        ├── chapter_1_moment_1.png
        └── ...
```

---

## 🖥️ Flux Local Pipeline (Diffusers)

The local pipeline mirrors Hugging Face's reference implementation and now installs `diffusers`, `accelerate`, `torch`, and `torchvision` via the project dependencies.

1. Accept the [`black-forest-labs/FLUX.1-dev`](https://huggingface.co/black-forest-labs/FLUX.1-dev) terms and `huggingface-cli login`
2. Enable the pipeline: `FLUX_USE_PIPELINE=true`
3. Optionally tune `FLUX_PIPELINE_MODEL_ID`, `FLUX_PIPELINE_DEVICE`, `FLUX_PIPELINE_DTYPE`, or pass generation overrides in the CLI/web UI
4. The app falls back to Hugging Face endpoints or Replicate automatically if the local pipeline cannot load

Default sampling parameters follow the official guidance (`guidance_scale=3.5`, `num_inference_steps=50`, `height=1024`, `width=1024`, `max_sequence_length=512`) and can be overridden per prompt.

---

## 🔧 Advanced CLI Options

- `--max-moments` – number of emotional moments to surface per chapter (default **10**)
- `--comprehensive/--standard` – toggle scene-aware pipeline vs. lightweight mode
- `--mode {basic|scene|parallel}` – choose processing topology
- `--style-config path.json` – load pre-built style presets (e.g. `eh_shepard_pencil_config.json`)
- `--interactive/--batch` – switch between guided sessions and unattended automation

See `illustrator analyze --help` for the full list.

---

## 🏗️ Architecture Highlights

- **LangGraph** orchestrates the multi-node pipeline (context building, analysis, prompt engineering, quality feedback, render orchestration)
- **Pydantic v2** models capture manuscript state, style preferences, and provider payloads
- **MongoDB** persistence layer with indexed collections for manuscripts, sessions, checkpoints, and generated assets
- **FastAPI + WebSockets** power the optional real-time dashboard and remote execution modes
- **Robust error handling** via structured retries and provider-specific fallbacks

---

## 🧪 Development & Testing

```bash
pip install -e '.[dev]'
ruff check src tests
pytest
```

Automated tests rely on `mongomock`; set `MONGO_USE_MOCK=1` when running locally.

---

## 🤝 Contributing

Pull requests, issues, and feature ideas are welcome! See the [CONTRIBUTING.md](CONTRIBUTING.md) guidelines before submitting changes.

## 📄 License

Distributed under the [PI & Other Tales, Inc. End User License Agreement](LICENSE.md). Open source third-party components remain under their respective licenses.

## 👤 Maintainer

David James Lennon · PI & Other Tales, Inc. · david@othertales.co

If you ship something with Othertales Illustrator (Beta), we would love to hear about it!
