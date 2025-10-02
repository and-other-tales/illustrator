#!/usr/bin/env python3
"""
Comprehensive scene illustration generator for manuscript analysis.

This script performs deep analysis of manuscript chapters and generates
10 high-quality illustration prompts per chapter, then creates PNG images
using OpenAI DALL-E, Google Vertex AI, or Hugging Face APIs.
"""

import asyncio
import base64
import json
import os
import sys
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv
from rich.console import Console

# Configure logging
logger = logging.getLogger(__name__)
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# Import the illustrator modules
from illustrator.models import (
    Chapter, EmotionalMoment, EmotionalTone, ImageProvider,
    IllustrationPrompt, ChapterAnalysis
)
from illustrator.analysis import EmotionalAnalyzer
from illustrator.providers import ProviderFactory
from illustrator.parallel_processor import ParallelProcessor, parallel_processor_decorator
from illustrator.context import ManuscriptContext, get_default_context
from illustrator.llm_factory import create_chat_model_from_context

console = Console()

class ComprehensiveSceneAnalyzer:
    """Performs comprehensive analysis to extract 10 illustration-worthy scenes per chapter."""

    def __init__(
        self,
        context: ManuscriptContext | None = None,
        enable_parallel: bool = True,
    ):
        """Initialize with enhanced analysis parameters and optional parallel processing."""

        if context is None:
            context = get_default_context()

        self.context = context

        try:
            self.llm = create_chat_model_from_context(self.context)
        except Exception as exc:  # pragma: no cover - surfaced to CLI
            raise RuntimeError(
                "Failed to initialize analysis language model."
            ) from exc

        self.emotional_analyzer = EmotionalAnalyzer(self.llm)

        # Initialize parallel processor
        if enable_parallel:
            self.parallel_processor = ParallelProcessor(
                max_concurrent_llm=8,
                max_concurrent_image=3,
                enable_rate_limiting=True,
                enable_circuit_breaker=True
            )
        else:
            self.parallel_processor = None

    async def analyze_chapter_comprehensive(self, chapter: Chapter) -> List[EmotionalMoment]:
        """
        Perform comprehensive analysis to extract exactly 10 illustration-worthy moments.
        Uses multiple analysis passes to ensure rich, diverse scene selection.
        """
        console.print(f"🔍 Deep analysis of Chapter {chapter.number}: {chapter.title}")

        # Pass 1: Segment text into smaller, overlapping chunks for detailed analysis
        segments = self._create_detailed_segments(chapter.content, segment_size=300, overlap=50)
        console.print(f"   📑 Created {len(segments)} detailed text segments")

        # Pass 2: Score ALL segments for multiple criteria
        all_scored_moments = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:

            task = progress.add_task(f"Analyzing segments...", total=len(segments))

            for i, segment in enumerate(segments):
                # Multi-criteria scoring
                emotional_score = await self._score_emotional_intensity(segment)
                visual_score = await self._score_visual_potential(segment)
                narrative_score = await self._score_narrative_significance(segment)
                dialogue_score = await self._score_dialogue_richness(segment)

                # Combined score with weights for illustration potential
                combined_score = (
                    emotional_score * 0.3 +
                    visual_score * 0.4 +
                    narrative_score * 0.2 +
                    dialogue_score * 0.1
                )

                if combined_score >= 0.4:  # Lower threshold for more candidates
                    moment = await self._create_detailed_moment(segment, combined_score, chapter)
                    all_scored_moments.append((moment, combined_score))

                progress.update(task, advance=1)

        console.print(f"   ✅ Found {len(all_scored_moments)} high-potential illustration moments")

        # Pass 3: Select diverse set of exactly 10 moments
        selected_moments = await self._select_diverse_moments(all_scored_moments, target_count=10)

        console.print(f"   🎨 Selected {len(selected_moments)} diverse illustration scenes")
        return selected_moments

    async def analyze_chapter_comprehensive_parallel(self, chapter: Chapter) -> List[EmotionalMoment]:
        """
        Parallel version of comprehensive chapter analysis with optimized performance.
        Uses the scene-aware analysis for better emotional moment detection.
        """
        if not self.parallel_processor:
            # Fallback to regular analysis
            return await self.analyze_chapter_comprehensive(chapter)

        console.print(f"🚀 Parallel deep analysis of Chapter {chapter.number}: {chapter.title}")

        # Use the new scene-aware analysis from the enhanced EmotionalAnalyzer
        try:
            # This uses the scene boundary detection and character tracking
            selected_moments = await self.emotional_analyzer.analyze_chapter_with_scenes(
                chapter,
                max_moments=10,
                min_intensity=0.5,
                scene_awareness=True
            )

            console.print(f"   ✅ Scene-aware analysis found {len(selected_moments)} high-quality illustration moments")
            return selected_moments

        except Exception as e:
            console.print(f"   ⚠️ Scene-aware analysis failed ({e}), falling back to parallel segment analysis")

            # Fallback to parallel segment analysis
            return await self._analyze_segments_parallel(chapter)

    async def _analyze_segments_parallel(self, chapter: Chapter) -> List[EmotionalMoment]:
        """Parallel analysis of text segments using the parallel processor."""

        # Create segments
        segments = self._create_detailed_segments(chapter.content, segment_size=300, overlap=50)
        console.print(f"   📑 Created {len(segments)} segments for parallel analysis")

        # Create scoring tasks for parallel execution
        @parallel_processor_decorator(provider='llm_analysis', max_retries=2, timeout=60.0)
        async def score_segment_parallel(segment_data):
            segment, chapter_ref = segment_data

            # Multi-criteria scoring
            emotional_score = await self._score_emotional_intensity(segment)
            visual_score = await self._score_visual_potential(segment)
            narrative_score = await self._score_narrative_significance(segment)
            dialogue_score = await self._score_dialogue_richness(segment)

            combined_score = (
                emotional_score * 0.3 +
                visual_score * 0.4 +
                narrative_score * 0.2 +
                dialogue_score * 0.1
            )

            return {
                'segment': segment,
                'emotional_score': emotional_score,
                'visual_score': visual_score,
                'narrative_score': narrative_score,
                'dialogue_score': dialogue_score,
                'combined_score': combined_score,
                'chapter': chapter_ref
            }

        # Prepare segment data for parallel processing
        segment_data = [(segment, chapter) for segment in segments]

        # Process segments in batches for better performance
        batch_size = 10
        all_scored_results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:

            task = progress.add_task(f"Parallel segment analysis...", total=len(segments))

            for i in range(0, len(segment_data), batch_size):
                batch = segment_data[i:i + batch_size]

                # Process batch in parallel
                batch_tasks = [
                    score_segment_parallel(data) for data in batch
                ]

                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                # Process results and handle exceptions
                for result in batch_results:
                    if isinstance(result, Exception):
                        console.print(f"   ⚠️ Segment analysis failed: {result}")
                        continue

                    if result['combined_score'] >= 0.4:
                        all_scored_results.append(result)

                progress.update(task, advance=len(batch))

        console.print(f"   ✅ Parallel analysis found {len(all_scored_results)} high-potential moments")

        # Convert scored results to EmotionalMoments
        all_scored_moments = []
        for result in all_scored_results:
            try:
                moment = await self._create_detailed_moment(
                    result['segment'],
                    result['combined_score'],
                    result['chapter']
                )
                all_scored_moments.append((moment, result['combined_score']))
            except Exception as e:
                console.print(f"   ⚠️ Failed to create moment: {e}")

        # Select diverse moments
        selected_moments = await self._select_diverse_moments(all_scored_moments, target_count=10)

        console.print(f"   🎨 Selected {len(selected_moments)} diverse moments from parallel analysis")
        return selected_moments

    def _create_detailed_segments(self, text: str, segment_size: int = 300, overlap: int = 50):
        """Create overlapping segments optimized for scene detection."""
        words = text.split()
        segments = []

        # Create overlapping segments
        for i in range(0, len(words), segment_size - overlap):
            start_idx = i
            end_idx = min(i + segment_size, len(words))

            segment_text = ' '.join(words[start_idx:end_idx])

            # Skip very short segments
            if len(segment_text.split()) < 50:
                continue

            # Calculate character positions
            start_pos = len(' '.join(words[:start_idx])) if start_idx > 0 else 0
            end_pos = start_pos + len(segment_text)

            # Enhanced context
            context_before = ' '.join(words[max(0, start_idx-100):start_idx])
            context_after = ' '.join(words[end_idx:min(len(words), end_idx+100)])

            segments.append({
                'text': segment_text,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'context_before': context_before,
                'context_after': context_after,
                'word_count': end_idx - start_idx
            })

        return segments

    async def _score_visual_potential(self, segment: dict) -> float:
        """Score a segment for its visual illustration potential."""

        visual_prompt = """Rate this text passage's potential for creating a compelling visual illustration on a scale from 0.0 to 1.0.

Consider:
- Vivid sensory details (colors, lighting, textures, sounds)
- Clear spatial arrangements and composition potential
- Distinctive character appearances or actions
- Atmospheric elements (weather, time of day, mood)
- Symbolic or metaphorical visual elements
- Action or dramatic moments that would translate well to a static image

Rate ONLY for visual potential, not emotional impact. Return only a decimal number 0.0-1.0."""

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content=visual_prompt),
                HumanMessage(content=f"Text: {segment['text']}")
            ]

            response = await self.llm.ainvoke(messages)
            return max(0.0, min(1.0, float(response.content.strip())))

        except Exception:
            # Fallback: pattern-based visual scoring
            visual_keywords = [
                'light', 'shadow', 'color', 'bright', 'dark', 'glow', 'shimmer',
                'face', 'eyes', 'hair', 'hands', 'smile', 'expression',
                'room', 'window', 'door', 'street', 'sky', 'tree', 'building',
                'movement', 'gesture', 'pose', 'standing', 'sitting', 'walking'
            ]
            text_lower = segment['text'].lower()
            matches = sum(1 for keyword in visual_keywords if keyword in text_lower)
            return min(1.0, matches / 20.0)

    async def _score_narrative_significance(self, segment: dict) -> float:
        """Score narrative importance and plot significance."""

        narrative_prompt = """Rate this passage's narrative significance and plot importance on a scale 0.0 to 1.0.

Consider:
- Key plot developments or turning points
- Character revelations or important dialogue
- Introduction of significant elements
- Climactic or resolution moments
- Themes being established or developed

Return only a decimal number 0.0-1.0."""

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content=narrative_prompt),
                HumanMessage(content=f"Text: {segment['text']}")
            ]

            response = await self.llm.ainvoke(messages)
            return max(0.0, min(1.0, float(response.content.strip())))

        except Exception:
            return 0.5  # Default moderate significance

    async def _score_dialogue_richness(self, segment: dict) -> float:
        """Score the richness and illustration potential of dialogue."""
        text = segment['text']

        # Count dialogue markers
        dialogue_markers = text.count('"') + text.count("'") + text.count('"') + text.count('"')

        # Look for dialogue tags and character interactions
        dialogue_tags = ['said', 'asked', 'replied', 'whispered', 'shouted', 'murmured']
        tag_count = sum(text.lower().count(tag) for tag in dialogue_tags)

        # Normalize by text length
        score = min(1.0, (dialogue_markers + tag_count * 2) / (len(text.split()) / 10))
        return score

    async def _score_emotional_intensity(self, segment: dict) -> float:
        """Enhanced emotional intensity scoring."""
        # Use the existing emotional analyzer with the segment structure
        segment_obj = type('Segment', (), {
            'text': segment['text'],
            'start_pos': segment['start_pos'],
            'end_pos': segment['end_pos'],
            'context_before': segment['context_before'],
            'context_after': segment['context_after']
        })()

        return await self.emotional_analyzer._score_emotional_intensity(segment_obj)

    async def _create_detailed_moment(self, segment: dict, score: float, chapter: Chapter) -> EmotionalMoment:
        """Create a detailed EmotionalMoment from a high-scoring segment."""

        # Enhanced analysis prompt for illustration context
        analysis_prompt = """Analyze this text passage for illustration purposes. Provide:

1. The dominant emotional tones (up to 3 from: joy, sadness, anger, fear, surprise, disgust, anticipation, trust, melancholy, excitement, tension, peace, mystery, romance, adventure)
2. A detailed description of the visual scene for illustration
3. Key visual elements that should be emphasized
4. The emotional/atmospheric context
5. Characters present in the scene (list their names only)
6. A brief description of the setting (location, time, environment)
7. A brief narrative context (what's happening in the story at this point)

Respond in JSON format:
{
    "emotional_tones": ["emotion1", "emotion2", "emotion3"],
    "visual_description": "Detailed description of the scene for illustration",
    "key_visual_elements": ["element1", "element2", "element3"],
    "context": "Emotional and atmospheric context for illustration",
    "characters_present": ["character1", "character2"],
    "setting_description": "Brief description of the physical setting",
    "narrative_context": "Brief description of what's happening in the story"
}"""

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [
                SystemMessage(content=analysis_prompt),
                HumanMessage(content=f"Chapter: {chapter.title}\n\nText: {segment['text']}")
            ]

            response = await self.llm.ainvoke(messages)
            analysis = json.loads(response.content.strip())

            # More robust handling of emotional tones with validation
            valid_tones = []
            for tone in analysis.get('emotional_tones', ['anticipation']):
                try:
                    valid_tones.append(EmotionalTone(tone.lower()))
                except ValueError:
                    logger.warning(f"Invalid emotional tone '{tone}' detected - using default")
                    # Don't add invalid tones, we'll use a default if none are valid
            
            # Use a default if no valid tones were found
            emotional_tones = valid_tones if valid_tones else [EmotionalTone.ANTICIPATION]
            context = analysis.get('context', 'Significant narrative moment with visual potential')
            
            # Extract additional information
            characters_present = analysis.get('characters_present', [])
            setting_description = analysis.get('setting_description', 'Scene setting')
            narrative_context = analysis.get('narrative_context', 'Story progression')

        except Exception as e:
            # Fallback with more detailed error logging
            logger.error(f"Error creating detailed moment: {str(e)}")
            emotional_tones = [EmotionalTone.ANTICIPATION]
            context = "Visually compelling scene with narrative significance"
            characters_present = []
            setting_description = "Unknown setting"
            narrative_context = "Narrative progression"

        # Extract most visually rich excerpt
        excerpt = self._extract_visual_excerpt(segment['text'])

        return EmotionalMoment(
            text_excerpt=excerpt,
            start_position=segment['start_pos'],
            end_position=segment['end_pos'],
            emotional_tones=emotional_tones,
            intensity_score=score,
            context=context,
            characters_present=characters_present,
            setting_description=setting_description,
            narrative_context=narrative_context
        )

    def _extract_visual_excerpt(self, text: str, max_length: int = 250) -> str:
        """Extract the most visually descriptive excerpt."""
        sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]

        if not sentences:
            return text[:max_length]

        # Score sentences for visual content
        visual_keywords = [
            'light', 'shadow', 'color', 'bright', 'dark', 'glow', 'face', 'eyes',
            'room', 'window', 'street', 'stood', 'walked', 'looked', 'smiled'
        ]

        scored_sentences = []
        for sentence in sentences:
            if len(sentence) > 15:  # Skip very short sentences
                score = sum(1 for keyword in visual_keywords if keyword.lower() in sentence.lower())
                scored_sentences.append((sentence, score))

        if not scored_sentences:
            return text[:max_length]

        # Sort by visual richness and build excerpt
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        excerpt_parts = []
        current_length = 0

        for sentence, score in scored_sentences:
            if current_length + len(sentence) <= max_length:
                excerpt_parts.append(sentence)
                current_length += len(sentence) + 2  # +2 for ". "
            else:
                break

        if excerpt_parts:
            return '. '.join(excerpt_parts) + '.'
        else:
            return scored_sentences[0][0][:max_length]

    async def _select_diverse_moments(self, scored_moments: List[tuple], target_count: int = 10) -> List[EmotionalMoment]:
        """Select diverse, high-quality moments for illustration."""
        if len(scored_moments) <= target_count:
            return [moment for moment, score in scored_moments]

        # Sort by score
        scored_moments.sort(key=lambda x: x[1], reverse=True)

        selected = []
        used_emotions = set()
        text_positions = []

        # First pass: select highest scoring moments with diverse emotions
        for moment, score in scored_moments:
            if len(selected) >= target_count:
                break

            # Check for emotion diversity
            moment_emotions = set(tone.value for tone in moment.emotional_tones)

            # Check for position diversity (avoid clustering)
            too_close = any(
                abs(moment.start_position - pos) < 5000  # 5000 characters apart
                for pos in text_positions
            )

            # Select if: high score, diverse emotions, good spacing
            if (score > 0.6 or len(selected) < target_count // 2) and not too_close:
                selected.append(moment)
                used_emotions.update(moment_emotions)
                text_positions.append(moment.start_position)
            elif len(selected) < target_count and not any(
                emotion in used_emotions for emotion in moment_emotions
            ):
                # Add for emotional diversity even if score is lower
                selected.append(moment)
                used_emotions.update(moment_emotions)
                text_positions.append(moment.start_position)

        # Fill remaining slots with highest scoring moments
        while len(selected) < target_count and len(selected) < len(scored_moments):
            for moment, score in scored_moments:
                if moment not in selected and len(selected) < target_count:
                    selected.append(moment)

        return selected[:target_count]


class IllustrationGenerator:
    """Generates high-quality illustrations from analyzed scenes."""

    def __init__(
        self,
        provider: ImageProvider,
        output_dir: Path,
        context: ManuscriptContext,
    ):
        """Initialize illustration generator."""
        self.provider = provider
        self.output_dir = output_dir
        self.context = context
        self.image_provider = self._setup_provider()

    def _setup_provider(self):
        """Setup the image generation provider."""
        return ProviderFactory.create_provider(
            self.provider,
            openai_api_key=self.context.openai_api_key,
            google_credentials=self.context.google_credentials,
            google_project_id=self.context.google_project_id,
            gcp_project_id=getattr(self.context, 'gcp_project_id', None) or self.context.google_project_id,
            huggingface_api_key=self.context.huggingface_api_key,
            anthropic_api_key=self.context.anthropic_api_key,
            llm_provider=self.context.llm_provider,
            llm_model=self.context.model,
            huggingface_task=self.context.huggingface_task,
            huggingface_device=self.context.huggingface_device,
            huggingface_max_new_tokens=self.context.huggingface_max_new_tokens,
            huggingface_temperature=self.context.huggingface_temperature,
            huggingface_model_kwargs=self.context.huggingface_model_kwargs,
            huggingface_endpoint_url=self.context.huggingface_endpoint_url,
            huggingface_flux_endpoint_url=getattr(self.context, 'huggingface_flux_endpoint_url', None),
            huggingface_timeout=self.context.huggingface_timeout,
            flux_dev_vertex_endpoint_url=getattr(self.context, 'flux_dev_vertex_endpoint_url', None),
        )

    async def generate_illustration_prompts(
        self,
        moments: List[EmotionalMoment],
        chapter: Chapter,
        style_preferences: Dict[str, str]
    ) -> List[IllustrationPrompt]:
        """Generate optimized illustration prompts for each moment with robust logging and timeout."""

        import logging, asyncio
        logger = logging.getLogger(__name__)
        prompts = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating illustration prompts...", total=len(moments))

            for i, moment in enumerate(moments):
                logger.info(f"[PromptGen] Starting prompt for moment {i+1}/{len(moments)} in Chapter {chapter.number}: {chapter.title}")
                try:
                    # Timeout for LLM calls (30s)
                    prompt = await asyncio.wait_for(
                        self.image_provider.generate_prompt(
                            emotional_moment=moment,
                            style_preferences=style_preferences,
                            context=f"Chapter {chapter.number}: {chapter.title}"
                        ),
                        timeout=30.0
                    )
                    prompts.append(prompt)
                    logger.info(f"[PromptGen] Success for moment {i+1}: {getattr(prompt, 'prompt', str(prompt))[:120]}")
                except asyncio.TimeoutError:
                    logger.error(f"[PromptGen] Timeout for moment {i+1} in Chapter {chapter.number}")
                    console.print(f"[red]Timeout: Could not generate prompt for moment {i+1} in Chapter {chapter.number}[/red]")
                except Exception as e:
                    logger.error(f"[PromptGen] Error for moment {i+1}: {e}")
                    console.print(f"[yellow]Warning: Could not generate prompt for moment {i+1}: {e}[/yellow]")
                    # Fallback prompt (instructive, style-aware)
                    art_style = (style_preferences.get('art_style') or 'digital painting').lower()
                    if any(k in art_style for k in ["pencil", "shepard"]):
                        lead = "A natural pencil sketch illustration in the classic E.H. Shepard style."
                        technique = (
                            "The sketch should balance subtle emotion and charm, with fine crosshatching, "
                            "gentle graphite shading, and expressive, characterful linework."
                        )
                    elif "watercolor" in art_style:
                        lead = "A delicate watercolor illustration with soft edges and layered washes."
                        technique = "Use gentle color transitions, reserved whites, and restrained detail for a book-illustration feel."
                    elif "oil" in art_style:
                        lead = "A traditional oil painting with classic book-illustration sensibility."
                        technique = "Employ controlled brushwork, clear focal hierarchy, and warm, cohesive tones."
                    elif "digital" in art_style:
                        lead = "A cinematic digital painting with classic book-illustration clarity."
                        technique = "Use clean rendering, atmospheric lighting, and a readable focal point with restrained detailing."
                    else:
                        lead = f"A detailed {art_style} illustration in a classic book-illustration style."
                        technique = "Maintain clear focal hierarchy, readable forms, and tasteful, story-forward detailing."

                    tone_key = (moment.emotional_tones[0].name if moment.emotional_tones else "NEUTRAL").upper()
                    tone_cues = {
                        "JOY": "light, open posture; soft, warm lighting; relaxed expressions",
                        "SADNESS": "slumped shoulders; downcast eyes; muted tones; gentle shadows",
                        "FEAR": "stiff smile, wide eyes, tense shoulders; close, intimate framing; subtle unease",
                        "ANGER": "tight jaw; furrowed brow; energetic angle; bold contrasts",
                        "TENSION": "held breath; careful spacing between figures; controlled, taut poses",
                        "MYSTERY": "soft shadows; partially obscured details; suggestive, not explicit cues",
                        "ANTICIPATION": "forward lean; alert eyes; restrained motion; poised tension",
                        "SUSPENSE": "stillness; withheld action; compressed spacing; shadow accents",
                        "MELANCHOLY": "soft posture; thoughtful gaze; cool, quiet atmosphere",
                        "PEACE": "relaxed stance; gentle light; uncluttered forms",
                        "ROMANCE": "gentle posture; softened expressions; warm, intimate spacing",
                        "NEUTRAL": "balanced posture; natural lighting; calm, observational framing"
                    }
                    tone_dir = tone_cues.get(tone_key, tone_cues["NEUTRAL"])            

                    comp_by_tone = {
                        "FEAR": "Use intimate, slightly off-center framing to heighten unease.",
                        "TENSION": "Favor a medium, close, or three-quarter view that emphasizes body language.",
                        "SADNESS": "Choose a quiet, balanced composition; leave gentle breathing space around the subject.",
                        "ANGER": "Consider a dynamic angle with directional lines leading into the focal point.",
                        "MYSTERY": "Let soft shadow shapes and partial occlusion guide the eye to the focal area.",
                        "JOY": "Use an open composition with soft, welcoming shapes.",
                        "ANTICIPATION": "Place the subject slightly off-center; leave room in the direction of attention.",
                        "SUSPENSE": "Keep a controlled, symmetrical base with a small destabilizing element.",
                        "MELANCHOLY": "Widen the framing slightly; allow negative space to carry feeling.",
                        "PEACE": "Use an even, centered composition with gentle overlaps and clear separation.",
                        "ROMANCE": "Favor a two-shot with soft triangulation and gentle overlap for intimacy.",
                        "NEUTRAL": "Keep a balanced medium shot with clear focal hierarchy."
                    }
                    comp = comp_by_tone.get(tone_key, comp_by_tone["NEUTRAL"])          

                    excerpt = (moment.text_excerpt or "").strip()
                    if excerpt:
                        scene_line = (
                            f"Depict the specific moment described — \"{excerpt[:220]}\" — as a clear, readable scene."
                        )
                    else:
                        scene_line = f"Depict a key moment from '{chapter.title}', focusing on a clear action and reaction."

                    # Optional environment hints
                    env_hints = []
                    lower_all = f"{chapter.title} {excerpt}".lower()
                    if any(k in lower_all for k in ["room", "hall", "hallway", "stair", "kitchen", "living room", "house", "home", "bedroom"]):
                        env_hints.append("Interior domestic setting; suggest doorframes, window light, wall textures, and floor patterns.")
                    if any(k in lower_all for k in ["phone", "call", "receiver", "hang up", "dial"]):
                        env_hints.append("Include a phone or receiver as a subtle prop if appropriate.")
                    if any(k in lower_all for k in ["night", "dark", "shadow", "dim", "lamp"]):
                        env_hints.append("Low ambient light; soft lamp glow; readable shadow shapes.")

                    parts = [
                        lead,
                        scene_line,
                        f"Focus cues: {tone_dir}.",
                        comp,
                        "Include specific environmental cues suggested by the text (doors, windows, light sources, textures).",
                        technique
                    ]
                    if env_hints:
                        parts.insert(4, " ".join(env_hints))

                    prompt_text = " ".join(parts)

                    fallback_prompt = IllustrationPrompt(
                        provider=self.provider,
                        prompt=prompt_text,
                        style_modifiers=[style_preferences.get('art_style', 'book illustration')],
                        negative_prompt="text, watermark, low quality, blurry",
                        technical_params=self._get_default_params()
                    )
                    prompts.append(fallback_prompt)

                progress.update(task, advance=1)
                logger.info(f"[PromptGen] Finished moment {i+1}/{len(moments)}")

        return prompts

    def _get_default_params(self) -> Dict[str, Any]:
        """Get default technical parameters for the provider."""
        if self.provider == ImageProvider.DALLE:
            return {
                "model": "gpt-image-1",
                "size": "1024x1024",
                "quality": "hd",
                "style": "natural"
            }
        elif self.provider == ImageProvider.IMAGEN4:
            return {
                "aspect_ratio": "1:1",
                "safety_filter_level": "block_some",
                "person_generation": "allow_adult"
            }
        elif self.provider == ImageProvider.FLUX:
            return {
                "width": 1024,
                "height": 1024,
                "guidance_scale": 7.5,
                "num_inference_steps": 50
            }
        elif self.provider == ImageProvider.FLUX_DEV_VERTEX:
            return {
                "width": 1024,
                "height": 1024,
                "guidance_scale": 7.5,
                "num_inference_steps": 50
            }
        elif self.provider == ImageProvider.SEEDREAM:
            return {
                "width": 1024,
                "height": 1024,
                "cfg_scale": 6.5,
                "steps": 30
            }
        elif self.provider == ImageProvider.HUGGINGFACE:
            return {
                "model_id": getattr(self.context, 'huggingface_image_model', None),
                "width": 1024,
                "height": 1024,
                "guidance_scale": 7.0,
                "num_inference_steps": 40,
            }
        return {}

    async def generate_images(
        self,
        prompts: List[IllustrationPrompt],
        chapter: Chapter
    ) -> List[Dict[str, Any]]:
        """Generate actual images from prompts."""

        generated_images = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:

            task = progress.add_task(
                f"Generating images for Chapter {chapter.number}...",
                total=len(prompts)
            )

            for i, prompt in enumerate(prompts):
                try:
                    console.print(f"[dim]Generating image {i+1}/{len(prompts)}...[/dim]")

                    result = await self.image_provider.generate_image(prompt)

                    if result.get('success'):
                        # Save the image
                        filename = f"chapter_{chapter.number:02d}_scene_{i+1:02d}.png"
                        image_path = self.output_dir / filename

                        # Decode and save base64 image
                        image_data = result['image_data']
                        with open(image_path, 'wb') as f:
                            f.write(base64.b64decode(image_data))

                        # Save metadata
                        metadata = {
                            'chapter_number': chapter.number,
                            'chapter_title': chapter.title,
                            'scene_number': i + 1,
                            'prompt': prompt.prompt,
                            'style_modifiers': prompt.style_modifiers,
                            'negative_prompt': prompt.negative_prompt,
                            'technical_params': prompt.technical_params,
                            'generated_at': datetime.now().isoformat(),
                            'provider': self.provider.value,
                            'file_path': str(image_path)
                        }

                        metadata_path = self.output_dir / f"chapter_{chapter.number:02d}_scene_{i+1:02d}_metadata.json"
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=2, ensure_ascii=False)

                        generated_images.append({
                            'scene_number': i + 1,
                            'image_path': image_path,
                            'metadata': metadata,
                            'success': True
                        })

                        console.print(f"[green]✅ Scene {i+1}: {filename}[/green]")

                    else:
                        error_msg = result.get('error', 'Unknown error')
                        console.print(f"[red]❌ Scene {i+1} failed: {error_msg}[/red]")
                        generated_images.append({
                            'scene_number': i + 1,
                            'error': error_msg,
                            'success': False
                        })

                except Exception as e:
                    console.print(f"[red]❌ Scene {i+1} error: {e}[/red]")
                    generated_images.append({
                        'scene_number': i + 1,
                        'error': str(e),
                        'success': False
                    })

                progress.update(task, advance=1)

        return generated_images


async def main():
    """Main function to run comprehensive scene illustration generation."""

    # Load environment variables
    load_dotenv()

    console.print("""
[bold blue]🎨 Comprehensive Scene Illustration Generator[/bold blue]

This script will:
• Perform deep analysis of your manuscript chapters
• Extract exactly 10 illustration-worthy scenes per chapter
• Generate high-quality AI illustrations using your preferred provider
• Save PNG files and comprehensive metadata
""")

    # Find and load manuscript analysis
    analysis_path = Path("illustrator_output/Fortunes_Told_(A_Voyager's_Guide_To_Life_Between_Worlds)/manuscript_analysis.json")

    if not analysis_path.exists():
        console.print(f"[red]Error: Could not find manuscript analysis at {analysis_path}[/red]")
        sys.exit(1)

    # Load manuscript data
    with open(analysis_path, 'r') as f:
        manuscript_data = json.load(f)

    chapters = [Chapter(**ch) for ch in manuscript_data['chapters']]
    console.print(f"[green]📚 Loaded {len(chapters)} chapters ({sum(ch.word_count for ch in chapters):,} total words)[/green]")

    # Setup image provider
    provider_name = os.getenv('DEFAULT_IMAGE_PROVIDER', 'dalle').lower()
    if provider_name == 'dalle':
        provider = ImageProvider.DALLE
        console.print("[blue]🎨 Using DALL-E 3 for image generation[/blue]")
    elif provider_name == 'imagen4':
        provider = ImageProvider.IMAGEN4
        console.print("[blue]🎨 Using Google Imagen4 for image generation[/blue]")
    elif provider_name == 'flux':
        provider = ImageProvider.FLUX
        console.print("[blue]🎨 Using Flux 1.1 Pro for image generation[/blue]")
    elif provider_name == 'huggingface':
        provider = ImageProvider.HUGGINGFACE
        console.print("[blue]🎨 Using HuggingFace Endpoint for image generation[/blue]")
    elif provider_name in {'seedream', 'seedream4'}:
        provider = ImageProvider.SEEDREAM
        console.print("[blue]🎨 Using Seedream 4 for image generation[/blue]")
    else:
        console.print(f"[red]Unknown provider: {provider_name}. Using DALL-E.[/red]")
        provider = ImageProvider.DALLE

    # Create output directory
    output_dir = Path("scene_illustrations") / manuscript_data['metadata']['title'].replace(" ", "_").replace("(", "").replace(")", "")
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]📁 Output directory: {output_dir}[/green]")

    # Style preferences from environment
    default_style = os.getenv('DEFAULT_ILLUSTRATION_STYLE', 'hand-drawn pencil sketch in the style of E.H. Shepard and Tove Jansson, classic children\'s book illustration, delicate line work, cross-hatching, whimsical characters')

    style_preferences = {
        'art_style': 'pencil sketch',
        'style_name': 'E.H. Shepard',
        'base_prompt_modifiers': [
            'hand-drawn pencil sketch',
            'in the style of E.H. Shepard and Tove Jansson',
            'classic children\'s book illustration',
            'delicate line work',
            'cross-hatching',
            'whimsical characters',
            'gentle shading',
            'expressive facial features',
            'soft pencil textures'
        ],
        'color_palette': 'monochrome pencil with subtle shading',
        'artistic_influences': 'E.H. Shepard, Tove Jansson, classic book illustration',
        'technical_params': {
            'style': 'artistic',
            'quality': 'high'
        }
    }

    # Initialize components
    context = get_default_context()
    context.image_provider = provider

    analyzer = ComprehensiveSceneAnalyzer(context=context)
    generator = IllustrationGenerator(provider, output_dir, context)

    # Process each chapter
    total_scenes_generated = 0
    total_scenes_attempted = 0

    for chapter in chapters:
        console.print(f"\n[bold cyan]📖 Processing Chapter {chapter.number}: {chapter.title}[/bold cyan]")

        try:
            # Comprehensive scene analysis
            emotional_moments = await analyzer.analyze_chapter_comprehensive(chapter)
            console.print(f"[green]✅ Extracted {len(emotional_moments)} illustration scenes[/green]")

            # Generate illustration prompts
            prompts = await generator.generate_illustration_prompts(
                emotional_moments, chapter, style_preferences
            )
            console.print(f"[green]✅ Generated {len(prompts)} illustration prompts[/green]")

            # Generate images
            results = await generator.generate_images(prompts, chapter)

            successful = sum(1 for r in results if r.get('success', False))
            total_scenes_generated += successful
            total_scenes_attempted += len(results)

            console.print(f"[green]✅ Chapter {chapter.number}: {successful}/{len(results)} images generated successfully[/green]")

            # Save chapter analysis results
            chapter_data = {
                'chapter': chapter.model_dump(),
                'emotional_moments': [moment.model_dump() for moment in emotional_moments],
                'illustration_prompts': [prompt.model_dump() for prompt in prompts],
                'generation_results': [{
                    **r,
                    'image_path': str(r['image_path']) if r.get('image_path') else None
                } for r in results],
                'analysis_timestamp': datetime.now().isoformat()
            }

            chapter_file = output_dir / f"chapter_{chapter.number:02d}_analysis.json"
            with open(chapter_file, 'w') as f:
                json.dump(chapter_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            console.print(f"[red]❌ Error processing Chapter {chapter.number}: {e}[/red]")

    # Final summary
    console.print(f"""
[bold green]🎉 Scene Illustration Generation Complete![/bold green]

📊 [bold]Results Summary:[/bold]
• Total chapters processed: {len(chapters)}
• Total scenes analyzed: {total_scenes_attempted}
• Total images generated: {total_scenes_generated}
• Success rate: {(total_scenes_generated/total_scenes_attempted)*100:.1f}%

📁 [bold]Output Location:[/bold] {output_dir}

All images are saved as high-resolution PNG files with comprehensive metadata.
Each chapter has exactly 10 illustration scenes (when analysis is successful).
""")


if __name__ == "__main__":
    asyncio.run(main())
