# Illustrator 📚✨

A powerful LangGraph application that analyzes manuscript chapters using advanced NLP and emotional resonance detection, then generates optimized illustration prompts for DALL-E, Google Vertex AI Imagen (Imagen 3), and Flux 1.1 Pro.

[![LangGraph](https://img.shields.io/badge/Built%20with-LangGraph-00324d.svg)](https://langchain-ai.github.io/langgraph/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## ✨ Features

- **📖 Chapter-by-Chapter Analysis**: Process manuscripts one chapter at a time with detailed emotional and thematic analysis
- **🧠 Emotional Resonance Engine**: Advanced NLP to identify the most emotionally impactful moments (defaults to 10 per chapter)
- **🎨 Multi-Provider Support**: Generate illustrations using DALL-E 3, Vertex AI Imagen (Imagen 3), or Flux 1.1 Pro
- **🔄 Interactive CLI**: User-friendly command-line interface with CTRL+D input handling
- **💾 Persistent Storage**: Save analysis results and generated images for future reference
- **⚙️ Configurable Styles**: Customize artistic styles, color palettes, and creative influences

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- API keys for your chosen image generation provider(s):
  - **DALL-E**: OpenAI API key
  - **Imagen (Vertex AI)**: Google Cloud credentials and project ID
  - **Flux**: HuggingFace API key
- **Claude API key** (Anthropic) for text analysis

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
# Required for Claude analysis
ANTHROPIC_API_KEY=your-anthropic-key

# For DALL-E (OpenAI)
OPENAI_API_KEY=your-openai-key

# For Vertex AI Imagen (Google Cloud)
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account.json
GOOGLE_PROJECT_ID=your-project-id

# For Flux (HuggingFace)
HUGGINGFACE_API_KEY=your-huggingface-key

# Optional: Customize defaults
DEFAULT_IMAGE_PROVIDER=dalle  # dalle, imagen4 (Vertex Imagen), or flux
DEFAULT_ILLUSTRATION_STYLE=digital_painting
```

## 💡 How It Works

### 1. **Manuscript Input**
- Interactive chapter input with title and content
- CTRL+D to finish entering chapter text
- Support for large text blocks via copy/paste

### 2. **Emotional Analysis**
- Claude analyzes text for emotional peaks and valleys
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

## 🎨 Supported Image Providers

| Provider | Strengths | Best For |
|----------|-----------|----------|
| **DALL-E (gpt-image-1)** | High-quality, coherent images | Character-focused scenes, detailed illustrations |
| **Imagen (Vertex AI Imagen 3)** | Photorealistic output, fine control | Realistic settings, atmospheric scenes |
| **Flux 1.1 Pro** | Artistic styles, creative interpretation | Stylized art, concept illustrations |

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
│ • Choose from DALL-E, Vertex AI Imagen, or Flux    │
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
  3. Flux 1.1 Pro (HuggingFace)

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
- Powered by [Claude](https://anthropic.com) for literary analysis
- Supports [DALL-E](https://openai.com/dall-e-3), [Vertex AI Imagen](https://cloud.google.com/vertex-ai), and [Flux](https://huggingface.co/black-forest-labs/FLUX.1-pro)

---

**Author**: David James Lennon
**Email**: david@othertales.co
**Project**: [And Other Tales](https://othertales.co)
