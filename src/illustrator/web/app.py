"""FastAPI web application for Manuscript Illustrator."""

import os
import json
from pathlib import Path
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from rich.console import Console

from illustrator.web.routes import manuscripts, chapters
from illustrator.web.models.web_models import ConnectionManager
from illustrator.models import EmotionalTone, IllustrationPrompt

console = Console()

# Load environment variables from .env file in the project root
# Find the project root directory (where the .env file is located)
project_root = Path(__file__).parent.parent.parent.parent
env_file = project_root / '.env'
load_dotenv(env_file)


# Get the web directory path
WEB_DIR = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"
TEMPLATES_DIR = WEB_DIR / "templates"

# Create FastAPI app
app = FastAPI(
    title="Manuscript Illustrator",
    description="AI-powered manuscript analysis and illustration generation",
    version="1.0.0",
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Mount generated images directory
GENERATED_IMAGES_DIR = Path("illustrator_output") / "generated_images"
GENERATED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/generated", StaticFiles(directory=str(GENERATED_IMAGES_DIR)), name="generated")

# Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# WebSocket connection manager
connection_manager = ConnectionManager()

# Include routers
app.include_router(manuscripts.router, prefix="/api/manuscripts", tags=["manuscripts"])
app.include_router(chapters.router, prefix="/api/chapters", tags=["chapters"])

# Add the process endpoint directly here for simplicity
from illustrator.web.models.web_models import ProcessingRequest
from fastapi import BackgroundTasks
import uuid
from datetime import datetime
import asyncio

@app.post("/api/process")
async def start_processing(
    request: ProcessingRequest,
    background_tasks: BackgroundTasks
):
    """Start manuscript processing and illustration generation."""
    try:
        # Generate a session ID for tracking
        session_id = str(uuid.uuid4())

        # Start the actual processing in the background
        background_tasks.add_task(
            run_processing_workflow,
            session_id=session_id,
            manuscript_id=request.manuscript_id,
            style_config=request.style_config,
            max_emotional_moments=getattr(request, 'max_emotional_moments', 10)
        )

        return {
            "success": True,
            "session_id": session_id,
            "message": "Processing started successfully",
            "started_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error starting processing: {str(e)}"
        )

async def run_processing_workflow(
    session_id: str,
    manuscript_id: str,
    style_config: dict,
    max_emotional_moments: int = 10
):
    """Run the actual processing workflow with WebSocket updates."""
    try:
        # Import the processing logic
        import sys
        import base64
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent.parent.parent))
        from generate_scene_illustrations import ComprehensiveSceneAnalyzer, IllustrationGenerator
        from illustrator.web.routes.manuscripts import get_saved_manuscripts
        import uuid

        # Send initial status
        await connection_manager.send_personal_message(
            json.dumps({
                "type": "log",
                "level": "info",
                "message": "Starting manuscript processing..."
            }),
            session_id
        )

        # Initialize components with WebSocket-enabled analyzer
        analyzer = WebSocketComprehensiveSceneAnalyzer(connection_manager, session_id)
        from illustrator.models import ImageProvider

        # Map string to ImageProvider enum
        provider_str = style_config.get("image_provider", "imagen4").lower()
        if provider_str == "dalle":
            provider = ImageProvider.DALLE
        elif provider_str == "imagen4":
            provider = ImageProvider.IMAGEN4
        elif provider_str == "flux":
            provider = ImageProvider.FLUX
        else:
            provider = ImageProvider.DALLE  # default fallback

        # Create output directory
        output_dir = Path("illustrator_output") / "generated_images"
        output_dir.mkdir(parents=True, exist_ok=True)

        generator = WebSocketIllustrationGenerator(connection_manager, session_id, provider, output_dir)

        # Load manuscript chapters
        await connection_manager.send_personal_message(
            json.dumps({
                "type": "progress",
                "progress": 10,
                "message": "Loading manuscript chapters..."
            }),
            session_id
        )

        # Find manuscript by ID (using the same logic as manuscripts.py)
        manuscripts = get_saved_manuscripts()
        manuscript = None
        for saved_manuscript in manuscripts:
            generated_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, saved_manuscript.file_path))
            if generated_id == manuscript_id:
                manuscript = saved_manuscript
                break

        if not manuscript:
            raise Exception(f"Manuscript {manuscript_id} not found")

        chapters = manuscript.chapters
        if not chapters:
            raise Exception(f"No chapters found for manuscript {manuscript_id}")

        await connection_manager.send_personal_message(
            json.dumps({
                "type": "step",
                "step": 0,
                "status": "completed"
            }),
            session_id
        )

        # Process each chapter
        total_images = 0
        progress_per_chapter = 70 // len(chapters)

        for i, chapter in enumerate(chapters):
            await connection_manager.send_personal_message(
                json.dumps({
                    "type": "progress",
                    "progress": 20 + (i * progress_per_chapter),
                    "message": f"Analyzing Chapter {chapter.number}: {chapter.title}"
                }),
                session_id
            )

            # Analyze chapter for emotional moments
            await connection_manager.send_personal_message(
                json.dumps({
                    "type": "log",
                    "level": "info",
                    "message": f"🔍 Deep analysis of Chapter {chapter.number}: {chapter.title}"
                }),
                session_id
            )

            emotional_moments = await analyzer.analyze_chapter_comprehensive(chapter)

            await connection_manager.send_personal_message(
                json.dumps({
                    "type": "step",
                    "step": 1,
                    "status": "processing" if i == 0 else "completed"
                }),
                session_id
            )

            # Generate illustration prompts
            await connection_manager.send_personal_message(
                json.dumps({
                    "type": "progress",
                    "progress": 20 + (i * progress_per_chapter) + (progress_per_chapter // 3),
                    "message": f"Creating illustration prompts for Chapter {chapter.number}"
                }),
                session_id
            )

            prompts = []
            for idx, moment in enumerate(emotional_moments[:max_emotional_moments], 1):
                primary_tone = moment.emotional_tones[0] if moment.emotional_tones else "neutral"

                # Use advanced AI-powered prompt engineering
                prompt_text = await generator.create_advanced_prompt(
                    emotional_moment=moment,
                    provider=provider,
                    style_config=style_config,
                    chapter=chapter
                )
                prompts.append(prompt_text)

                # Log detailed prompt generation with AI analysis
                excerpt_preview = moment.text_excerpt
                prompt_preview = prompt_text
                await connection_manager.send_personal_message(
                    json.dumps({
                        "type": "log",
                        "level": "info",
                        "message": f"      🤖 AI-Enhanced Prompt {idx}: [{primary_tone}] \"{excerpt_preview}\""
                    }),
                    session_id
                )
                await connection_manager.send_personal_message(
                    json.dumps({
                        "type": "log",
                        "level": "info",
                        "message": f"         Generated: \"{prompt_preview}\""
                    }),
                    session_id
                )

            if i == 0:
                await connection_manager.send_personal_message(
                    json.dumps({
                        "type": "step",
                        "step": 1,
                        "status": "completed"
                    }),
                    session_id
                )
                await connection_manager.send_personal_message(
                    json.dumps({
                        "type": "step",
                        "step": 2,
                        "status": "processing"
                    }),
                    session_id
                )

            # Generate images
            await connection_manager.send_personal_message(
                json.dumps({
                    "type": "progress",
                    "progress": 20 + (i * progress_per_chapter) + (2 * progress_per_chapter // 3),
                    "message": f"Generating {len(prompts)} images for Chapter {chapter.number}"
                }),
                session_id
            )

            results = await generator.generate_images(prompts, chapter)
            total_images += len(results)

            # Send image updates
            for result in results:
                if result.get("success") and result.get("file_path"):
                    # Convert file path to web-accessible URL
                    image_url = f"/generated/{Path(result['file_path']).name}"
                    await connection_manager.send_personal_message(
                        json.dumps({
                            "type": "image",
                            "image_url": image_url,
                            "prompt": result.get("prompt", "")
                        }),
                        session_id
                    )

        await connection_manager.send_personal_message(
            json.dumps({
                "type": "step",
                "step": 2,
                "status": "completed"
            }),
            session_id
        )
        await connection_manager.send_personal_message(
            json.dumps({
                "type": "step",
                "step": 3,
                "status": "processing"
            }),
            session_id
        )

        # Final completion
        await connection_manager.send_personal_message(
            json.dumps({
                "type": "progress",
                "progress": 100,
                "message": "Processing completed successfully!"
            }),
            session_id
        )

        await connection_manager.send_personal_message(
            json.dumps({
                "type": "step",
                "step": 3,
                "status": "completed"
            }),
            session_id
        )

        await connection_manager.send_personal_message(
            json.dumps({
                "type": "complete",
                "images_count": total_images,
                "message": f"Successfully generated {total_images} illustrations!"
            }),
            session_id
        )

    except Exception as e:
        console.print(f"[red]Processing error: {e}[/red]")
        await connection_manager.send_personal_message(
            json.dumps({
                "type": "error",
                "error": str(e)
            }),
            session_id
        )


class WebSocketComprehensiveSceneAnalyzer:
    """Enhanced analyzer that sends progress updates via WebSocket."""

    def __init__(self, connection_manager, session_id, llm_model: str = "claude-sonnet-4-20250514"):
        # Import here to avoid circular imports
        from generate_scene_illustrations import ComprehensiveSceneAnalyzer
        self.analyzer = ComprehensiveSceneAnalyzer(llm_model)
        self.connection_manager = connection_manager
        self.session_id = session_id

    async def analyze_chapter_comprehensive(self, chapter):
        """Analyze chapter with WebSocket progress updates."""
        # Send initial analysis message
        await self.connection_manager.send_personal_message(
            json.dumps({
                "type": "log",
                "level": "info",
                "message": f"📑 Created detailed text segments for analysis"
            }),
            self.session_id
        )

        # Create segments similar to the original analyzer
        segments = self.analyzer._create_detailed_segments(chapter.content, segment_size=300, overlap=50)

        await self.connection_manager.send_personal_message(
            json.dumps({
                "type": "log",
                "level": "info",
                "message": f"   📑 Created {len(segments)} detailed text segments"
            }),
            self.session_id
        )

        # Process segments with progress updates
        all_scored_moments = []
        total_segments = len(segments)

        for i, segment in enumerate(segments):
            # Send progress update every few segments with details about high-scoring segments
            if i % max(1, total_segments // 20) == 0:
                progress = int((i / total_segments) * 100)
                await self.connection_manager.send_personal_message(
                    json.dumps({
                        "type": "log",
                        "level": "info",
                        "message": f"⠋ Analyzing segments... {progress}% ({i}/{total_segments}) - Found {len(all_scored_moments)} promising moments so far"
                    }),
                    self.session_id
                )

            # Multi-criteria scoring
            emotional_score = await self.analyzer._score_emotional_intensity(segment)
            visual_score = await self.analyzer._score_visual_potential(segment)
            narrative_score = await self.analyzer._score_narrative_significance(segment)
            dialogue_score = await self.analyzer._score_dialogue_richness(segment)

            # Combined score with weights for illustration potential
            combined_score = (
                emotional_score * 0.3 +
                visual_score * 0.4 +
                narrative_score * 0.2 +
                dialogue_score * 0.1
            )

            if combined_score >= 0.4:  # Lower threshold for more candidates
                moment = await self.analyzer._create_detailed_moment(segment, combined_score, chapter)
                all_scored_moments.append((moment, combined_score))

                # Log high-scoring moments as they're discovered (only show very high scores to avoid spam)
                if combined_score >= 0.7:
                    segment_preview = segment
                    await self.connection_manager.send_personal_message(
                        json.dumps({
                            "type": "log",
                            "level": "info",
                            "message": f"      🌟 High-impact moment found (Score: {combined_score:.2f}): \"{segment_preview}\""
                        }),
                        self.session_id
                    )

        await self.connection_manager.send_personal_message(
            json.dumps({
                "type": "log",
                "level": "info",
                "message": f"   ✅ Found {len(all_scored_moments)} high-potential illustration moments"
            }),
            self.session_id
        )

        # Log details about each high-potential moment
        for i, (moment, score) in enumerate(all_scored_moments[:10], 1):  # Show top 10
            excerpt_preview = moment.text_excerpt
            tones_str = ", ".join([tone.value for tone in moment.emotional_tones])
            await self.connection_manager.send_personal_message(
                json.dumps({
                    "type": "log",
                    "level": "info",
                    "message": f"      #{i} (Score: {score:.2f}) [{tones_str}]: \"{excerpt_preview}\""
                }),
                self.session_id
            )

        # Select diverse moments
        selected_moments = await self.analyzer._select_diverse_moments(all_scored_moments, target_count=10)

        await self.connection_manager.send_personal_message(
            json.dumps({
                "type": "log",
                "level": "info",
                "message": f"   🎨 Selected {len(selected_moments)} diverse illustration scenes"
            }),
            self.session_id
        )

        # Log details about each selected moment
        for i, moment in enumerate(selected_moments, 1):
            excerpt_preview = moment.text_excerpt
            tones_str = ", ".join([tone.value for tone in moment.emotional_tones])
            await self.connection_manager.send_personal_message(
                json.dumps({
                    "type": "log",
                    "level": "info",
                    "message": f"      Selected #{i} [{tones_str}] (Intensity: {moment.intensity_score:.2f}): \"{excerpt_preview}\""
                }),
                self.session_id
            )

        return selected_moments


class WebSocketIllustrationGenerator:
    """Enhanced image generator that sends progress updates via WebSocket."""

    def __init__(self, connection_manager, session_id, provider, output_dir):
        # Import here to avoid circular imports
        from generate_scene_illustrations import IllustrationGenerator
        from illustrator.prompt_engineering import PromptEngineer
        from langchain.chat_models import init_chat_model

        self.generator = IllustrationGenerator(provider, output_dir)
        self.connection_manager = connection_manager
        self.session_id = session_id

        # Initialize the advanced prompt engineering system
        self.llm = init_chat_model(model="claude-sonnet-4-20250514", model_provider="anthropic")
        self.prompt_engineer = PromptEngineer(self.llm)

    async def create_advanced_prompt(self, emotional_moment, provider, style_config, chapter):
        """Create an advanced AI-analyzed prompt for image generation."""
        try:
            # Use the advanced prompt engineering system
            illustration_prompt = await self.prompt_engineer.engineer_prompt(
                emotional_moment=emotional_moment,
                provider=provider,
                style_preferences=style_config,
                chapter_context=chapter,
                previous_scenes=[]  # Could be enhanced to track previous scenes
            )

            await self.connection_manager.send_personal_message(
                json.dumps({
                    "type": "log",
                    "level": "info",
                    "message": f"🧠 AI Scene Analysis: Generated {len(illustration_prompt.prompt.split())}-word prompt with visual composition analysis"
                }),
                self.session_id
            )

            # Log additional AI analysis details
            if hasattr(illustration_prompt, 'style_modifiers') and illustration_prompt.style_modifiers:
                await self.connection_manager.send_personal_message(
                    json.dumps({
                        "type": "log",
                        "level": "info",
                        "message": f"   🎨 Style modifiers: {', '.join(illustration_prompt.style_modifiers[:3])}..."
                    }),
                    self.session_id
                )

            return illustration_prompt.prompt

        except Exception as e:
            # Fallback to simple prompt generation
            await self.connection_manager.send_personal_message(
                json.dumps({
                    "type": "log",
                    "level": "warning",
                    "message": f"⚠️ AI prompt generation failed, using fallback: {str(e)}"
                }),
                self.session_id
            )

            return self._create_fallback_prompt(emotional_moment.text_excerpt,
                                              emotional_moment.emotional_tones[0] if emotional_moment.emotional_tones else EmotionalTone.NEUTRAL,
                                              chapter.title,
                                              style_config.get("art_style", "digital painting"))

    def _create_fallback_prompt(self, description, tone, chapter_title, art_style):
        """Fallback prompt generation method."""
        tone_description = {
            EmotionalTone.JOY: "bright, uplifting, warm colors",
            EmotionalTone.MELANCHOLY: "soft, muted tones, contemplative mood",
            EmotionalTone.TENSION: "dramatic lighting, sharp contrasts, dynamic composition",
            EmotionalTone.MYSTERY: "shadowy, atmospheric, intriguing lighting",
            EmotionalTone.ROMANCE: "warm, intimate lighting, soft focus",
            EmotionalTone.FEAR: "dark, ominous, high contrast",
            EmotionalTone.ANTICIPATION: "golden hour lighting, expansive composition",
            EmotionalTone.SADNESS: "soft, muted tones, melancholic mood",
            EmotionalTone.EXCITEMENT: "vibrant, energetic colors, dynamic composition",
            EmotionalTone.PEACE: "serene, calm lighting, balanced composition",
            EmotionalTone.ADVENTURE: "epic, cinematic lighting, heroic composition"
        }.get(tone, "balanced lighting and composition")

        return f"{art_style} illustration: {description[:100]}... Scene mood: {tone_description}, high-quality {art_style}"

    async def generate_images(self, prompts, chapter):
        """Generate images with WebSocket progress updates."""
        generated_images = []
        total_prompts = len(prompts)

        await self.connection_manager.send_personal_message(
            json.dumps({
                "type": "log",
                "level": "info",
                "message": f"🎨 Starting image generation for {total_prompts} prompts..."
            }),
            self.session_id
        )

        for i, prompt in enumerate(prompts):
            try:
                # Send progress update with more detail
                progress = int(((i + 1) / total_prompts) * 100)
                # Extract scene description from prompt for logging
                prompt_preview = str(prompt)
                await self.connection_manager.send_personal_message(
                    json.dumps({
                        "type": "log",
                        "level": "info",
                        "message": f"🖼️ Generating image {i+1}/{total_prompts} ({progress}%): {prompt_preview}"
                    }),
                    self.session_id
                )

                # Convert string prompt to IllustrationPrompt object if necessary
                if isinstance(prompt, str):
                    prompt_obj = IllustrationPrompt(
                        provider=self.generator.provider,
                        prompt=prompt,
                        style_modifiers=[],
                        negative_prompt="",
                        technical_params={}
                    )
                else:
                    prompt_obj = prompt

                result = await self.generator.image_provider.generate_image(prompt_obj)

                if result.get('success'):
                    # Save the image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"chapter_{chapter.number}_scene_{i+1}_{timestamp}.png"
                    file_path = self.generator.output_dir / filename

                    # Ensure output directory exists
                    self.generator.output_dir.mkdir(parents=True, exist_ok=True)

                    # Save image based on result type
                    if 'base64_image' in result:
                        image_data = base64.b64decode(result['base64_image'])
                        with open(file_path, 'wb') as f:
                            f.write(image_data)
                    elif 'url' in result:
                        # For URL-based results, we'd need to download
                        # For now, just log the URL
                        await self.connection_manager.send_personal_message(
                            json.dumps({
                                "type": "log",
                                "level": "info",
                                "message": f"✅ Generated image {i+1}: {result['url']}"
                            }),
                            self.session_id
                        )

                    result_info = {
                        'success': True,
                        'file_path': str(file_path),
                        'prompt': str(prompt),
                        'chapter_number': chapter.number,
                        'scene_number': i + 1,
                        'provider': str(self.generator.provider),
                        'generated_at': timestamp
                    }

                    generated_images.append(result_info)

                    await self.connection_manager.send_personal_message(
                        json.dumps({
                            "type": "log",
                            "level": "success",
                            "message": f"✅ Successfully generated image {i+1}/{total_prompts}"
                        }),
                        self.session_id
                    )
                else:
                    await self.connection_manager.send_personal_message(
                        json.dumps({
                            "type": "log",
                            "level": "warning",
                            "message": f"❌ Failed to generate image {i+1}/{total_prompts}: {result.get('error', 'Unknown error')}"
                        }),
                        self.session_id
                    )

            except Exception as e:
                await self.connection_manager.send_personal_message(
                    json.dumps({
                        "type": "log",
                        "level": "error",
                        "message": f"❌ Error generating image {i+1}/{total_prompts}: {str(e)}"
                    }),
                    self.session_id
                )

        await self.connection_manager.send_personal_message(
            json.dumps({
                "type": "log",
                "level": "success",
                "message": f"🎉 Image generation complete! Generated {len(generated_images)} out of {total_prompts} images."
            }),
            self.session_id
        )

        return generated_images


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "Dashboard"}
    )


@app.get("/manuscript/new", response_class=HTMLResponse)
async def new_manuscript(request: Request):
    """New manuscript creation page."""
    return templates.TemplateResponse(
        "manuscript_form.html",
        {"request": request, "title": "New Manuscript", "manuscript": None}
    )


@app.get("/manuscript/{manuscript_id}", response_class=HTMLResponse)
async def manuscript_detail(request: Request, manuscript_id: str):
    """Manuscript detail and chapter management page."""
    return templates.TemplateResponse(
        "manuscript_detail.html",
        {"request": request, "title": "Manuscript", "manuscript_id": manuscript_id}
    )


@app.get("/manuscript/{manuscript_id}/edit", response_class=HTMLResponse)
async def edit_manuscript(request: Request, manuscript_id: str):
    """Edit manuscript page."""
    return templates.TemplateResponse(
        "manuscript_form.html",
        {"request": request, "title": "Edit Manuscript", "manuscript_id": manuscript_id}
    )


@app.get("/manuscript/{manuscript_id}/chapter/new", response_class=HTMLResponse)
async def new_chapter(request: Request, manuscript_id: str):
    """New chapter creation page."""
    return templates.TemplateResponse(
        "chapter_form.html",
        {"request": request, "title": "New Chapter", "manuscript_id": manuscript_id, "chapter": None}
    )


@app.get("/chapter/{chapter_id}/edit", response_class=HTMLResponse)
async def edit_chapter(request: Request, chapter_id: str):
    """Edit chapter page."""
    return templates.TemplateResponse(
        "chapter_form.html",
        {"request": request, "title": "Edit Chapter", "chapter_id": chapter_id}
    )


@app.get("/manuscript/{manuscript_id}/style", response_class=HTMLResponse)
async def style_config(request: Request, manuscript_id: str):
    """Style configuration page."""
    return templates.TemplateResponse(
        "style_config.html",
        {"request": request, "title": "Style Configuration", "manuscript_id": manuscript_id}
    )


@app.get("/manuscript/{manuscript_id}/process", response_class=HTMLResponse)
async def processing_page(request: Request, manuscript_id: str):
    """Processing and analysis page."""
    return templates.TemplateResponse(
        "processing.html",
        {"request": request, "title": "Processing", "manuscript_id": manuscript_id}
    )


@app.get("/chapter/{chapter_id}/headers", response_class=HTMLResponse)
async def chapter_headers_page(request: Request, chapter_id: str):
    """Chapter header options page."""
    return templates.TemplateResponse(
        "chapter_headers.html",
        {"request": request, "title": "Chapter Headers", "chapter_id": chapter_id}
    )


@app.get("/manuscript/{manuscript_id}/gallery", response_class=HTMLResponse)
async def gallery_page(request: Request, manuscript_id: str):
    """Image gallery page."""
    return templates.TemplateResponse(
        "gallery.html",
        {"request": request, "title": "Gallery", "manuscript_id": manuscript_id}
    )


@app.websocket("/ws/processing/{session_id}")
async def processing_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time processing updates."""
    await connection_manager.connect(websocket, session_id)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            # Echo back for now - in practice this would trigger processing
            await connection_manager.send_personal_message(f"Echo: {data}", session_id)
    except WebSocketDisconnect:
        connection_manager.disconnect(session_id)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "manuscript-illustrator"}


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app


def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    uvicorn.run(
        "illustrator.web.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_server()