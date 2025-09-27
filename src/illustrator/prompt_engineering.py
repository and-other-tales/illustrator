"""Advanced prompt engineering system for optimal text-to-image generation."""

import json
import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from illustrator.utils import parse_llm_json

logger = logging.getLogger(__name__)

from illustrator.models import (
    EmotionalMoment,
    EmotionalTone,
    IllustrationPrompt,
    ImageProvider,
    Chapter,
)
from illustrator.character_tracking import CharacterTracker


class CompositionType(str, Enum):
    """Visual composition types."""
    CLOSE_UP = "close_up"
    MEDIUM_SHOT = "medium_shot"
    WIDE_SHOT = "wide_shot"
    ESTABLISHING = "establishing"
    DRAMATIC = "dramatic"
    INTIMATE = "intimate"


class LightingMood(str, Enum):
    """Lighting and mood categories."""
    GOLDEN_HOUR = "golden_hour"
    DRAMATIC = "dramatic"
    SOFT = "soft"
    MYSTERIOUS = "mysterious"
    HARSH = "harsh"
    ETHEREAL = "ethereal"
    NATURAL = "natural"


@dataclass
class VisualElement:
    """Represents a visual element extracted from text."""
    element_type: str  # "character", "object", "environment", "atmosphere"
    description: str
    importance: float  # 0.0 to 1.0
    attributes: Dict[str, Any]


@dataclass
class SceneComposition:
    """Detailed scene composition analysis."""
    composition_type: CompositionType
    focal_point: str
    background_elements: List[str]
    foreground_elements: List[str]
    lighting_mood: LightingMood
    atmosphere: str
    color_palette_suggestion: str
    emotional_weight: float


@dataclass
class CharacterProfile:
    """Character appearance and emotional state tracking."""
    name: str
    physical_description: str
    emotional_state: List[EmotionalTone]
    current_action: str
    relationship_context: str
    consistency_notes: List[str]


@dataclass
class ChapterHeaderOption:
    """Represents a chapter header illustration option."""
    option_number: int
    title: str
    description: str
    visual_focus: str
    artistic_style: str
    composition_notes: str
    prompt: IllustrationPrompt


class SceneAnalyzer:
    """Analyzes text scenes for visual elements and composition."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

        # Visual element patterns
        self.character_patterns = [
            r'\b(he|she|they|[A-Z][a-z]+)\s+(looked|appeared|seemed|stood|sat|walked|ran|moved)',
            r'\b(eyes?|face|hair|hands?|expression|smile|frown)',
            r'\b(tall|short|young|old|beautiful|handsome|pale|dark)'
        ]

        self.environment_patterns = [
            r'\b(room|house|forest|field|mountain|ocean|sky|clouds?)',
            r'\b(light|shadow|sun|moon|stars?|fire|candle)',
            r'\b(tree|flower|grass|stone|rock|path|road|bridge)'
        ]

        self.object_patterns = [
            r'\b(table|chair|book|sword|door|window|mirror|painting)',
            r'\b(dress|coat|cloak|jewelry|crown|ring|necklace)',
            r'\b(food|drink|wine|bread|fruit|meal)'
        ]

    async def analyze_scene(
        self,
        emotional_moment: EmotionalMoment,
        chapter_context: Chapter,
        character_profiles: Dict[str, CharacterProfile] = None
    ) -> Tuple[SceneComposition, List[VisualElement]]:
        """Analyze a scene for visual composition and elements."""

        # Extract visual elements using pattern matching and LLM analysis
        visual_elements = await self._extract_visual_elements(
            emotional_moment.text_excerpt,
            emotional_moment.context,
            chapter_context
        )

        # Analyze scene composition
        composition = await self._analyze_composition(
            emotional_moment,
            visual_elements,
            character_profiles or {}
        )

        return composition, visual_elements

    async def _extract_visual_elements(
        self,
        text: str,
        context: str,
        chapter: Chapter
    ) -> List[VisualElement]:
        """Extract detailed visual elements from text."""

        system_prompt = """You are a master visual scene analyst specializing in E.H. Shepard-style book illustrations. Extract exceptionally detailed visual elements essential for creating emotionally rich, artistically compelling illustrations. Focus intensively on:

1. CHARACTERS WITH PRECISE EMOTIONAL DETAILS:
- Exact facial expressions (wide eyes, frozen smile, startled look, raised eyebrows, tense mouth)
- Specific body language (leaning back, shoulders hunched, hands positioned defensively, rigid posture)
- Detailed positioning (behind counter, at angle, slightly turned away, facing toward)
- Emotional state manifestations (fear showing through stiff smile, curiosity in tilted head, tension in stance)
- Clothing and appearance that enhances character portrayal

2. ENVIRONMENT WITH RICH SPECIFICITY:
- Precise setting details (cozy coffee shop interior, wooden counter, ceramic displays)
- Architectural elements (doorways, windows, ceiling details, floor patterns)
- Furniture and fixtures (specific café equipment, seating arrangements, lighting sources)
- Textural details (rough wood grain, smooth ceramic, soft fabric, metal finishes)
- Atmospheric conditions (warm lighting, shadows, spatial depth)

3. OBJECTS WITH NARRATIVE SIGNIFICANCE:
- Symbolic items that enhance storytelling (half-empty coffee cup, open book, mysterious package)
- Props that reveal character state (trembling hands holding cup, nervously arranged items)
- Environmental objects that create mood (flickering candle, old photograph, handwritten note)

4. COMPOSITION WITH EMOTIONAL IMPACT:
- Spatial relationships that create tension (close proximity yet emotional distance)
- Perspective elements (viewer's eye level, intimate framing, dramatic angles)
- Visual flow directing attention to emotional focal points

Return a JSON array with this enhanced structure:
[
    {
        "element_type": "character|environment|object|atmosphere",
        "description": "richly detailed description with specific visual and emotional elements",
        "importance": 0.9,
        "attributes": {
            "facial_expression": "precise emotional details (if character)",
            "body_language": "specific posture and gesture details (if character)",
            "spatial_position": "exact positioning and relationships",
            "emotional_significance": "detailed explanation of narrative and emotional importance",
            "artistic_potential": "how this element enhances E.H. Shepard-style illustration",
            "texture_details": "specific material and surface characteristics"
        }
    }
]

Prioritize elements that create emotional resonance and visual storytelling depth, perfect for detailed pencil illustration techniques."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""Scene text: {text}

Context: {context}

Chapter setting: Chapter {chapter.number} - {chapter.title}

Extract the most important visual elements for illustration.""")
            ]

            response = await self.llm.ainvoke(messages)

            try:
                elements_data = parse_llm_json(response.content)
            except ValueError as json_error:
                logger.warning(f"JSON parsing failed: {json_error}. Using fallback visual element extraction.")
                # Fallback: create basic visual elements from the text content
                return self._create_fallback_visual_elements(text, context, chapter)

            visual_elements = []
            for elem_data in elements_data:
                element = VisualElement(
                    element_type=elem_data.get('element_type', 'object'),
                    description=elem_data.get('description', ''),
                    importance=float(elem_data.get('importance', 0.5)),
                    attributes=elem_data.get('attributes', {})
                )
                visual_elements.append(element)

            return visual_elements

        except Exception as e:
            logger.error(f"Visual element extraction failed: {e}")
            logger.info("Attempting fallback visual element extraction...")
            try:
                return self._create_fallback_visual_elements(text, context, chapter)
            except Exception as fallback_error:
                logger.error(f"Fallback extraction also failed: {fallback_error}")
                raise ValueError(f"Failed to extract visual elements from scene: {str(e)}")

    def _create_fallback_visual_elements(self, text: str, context: str, chapter: Chapter) -> List[VisualElement]:
        """Create fallback visual elements using pattern matching when LLM parsing fails."""
        visual_elements = []

        # Basic pattern matching for characters
        character_indicators = ['he ', 'she ', 'they ', 'character', 'protagonist', 'person', 'man', 'woman']
        for indicator in character_indicators:
            if indicator.lower() in text.lower():
                visual_elements.append(VisualElement(
                    element_type="character",
                    description=f"Character from scene: {context}",
                    importance=0.8,
                    attributes={"context": context, "fallback": True}
                ))
                break

        # Basic pattern matching for objects/props
        object_patterns = ['sword', 'book', 'door', 'window', 'table', 'chair', 'lamp', 'fire', 'candle', 'mirror']
        found_objects = [obj for obj in object_patterns if obj.lower() in text.lower()]
        for obj in found_objects[:3]:  # Limit to 3 objects
            visual_elements.append(VisualElement(
                element_type="object",
                description=f"{obj} mentioned in scene",
                importance=0.6,
                attributes={"object_type": obj, "fallback": True}
            ))

        # Basic environment detection
        environment_keywords = ['room', 'house', 'forest', 'street', 'garden', 'castle', 'shop', 'inn', 'tavern']
        for env in environment_keywords:
            if env.lower() in text.lower():
                visual_elements.append(VisualElement(
                    element_type="environment",
                    description=f"Scene set in {env}",
                    importance=0.7,
                    attributes={"setting": env, "fallback": True}
                ))
                break

        # If no elements found, create a generic scene element
        if not visual_elements:
            visual_elements.append(VisualElement(
                element_type="atmosphere",
                description=f"Scene from chapter {chapter.number}: {chapter.title}",
                importance=0.5,
                attributes={"context": context, "chapter": chapter.number, "fallback": True}
            ))

        logger.info(f"Created {len(visual_elements)} fallback visual elements")
        return visual_elements

    def _create_fallback_scene_composition(
        self,
        emotional_moment: EmotionalMoment,
        visual_elements: List[VisualElement],
    ) -> SceneComposition:
        """Provide a resilient fallback scene composition when LLM analysis fails."""

        focal_point_description = (
            visual_elements[0].description if visual_elements else emotional_moment.text_excerpt[:120]
        )

        background_elements = [
            element.description
            for element in visual_elements
            if element.element_type in {"environment", "atmosphere"}
        ][:3]

        foreground_elements = [
            element.description
            for element in visual_elements
            if element.element_type == "character"
        ][:3]

        emotional_weight = 0.5 + 0.1 * len(foreground_elements)

        return SceneComposition(
            composition_type=CompositionType.MEDIUM_SHOT,
            focal_point=focal_point_description or "primary subject",
            background_elements=background_elements,
            foreground_elements=foreground_elements,
            lighting_mood=LightingMood.NATURAL,
            atmosphere="Fallback composition emphasizing core scene elements.",
            color_palette_suggestion="balanced natural tones",
            emotional_weight=min(1.0, emotional_weight),
        )

    async def _analyze_composition(
        self,
        emotional_moment: EmotionalMoment,
        visual_elements: List[VisualElement],
        character_profiles: Dict[str, CharacterProfile]
    ) -> SceneComposition:
        """Analyze scene composition and visual structure."""

        system_prompt = """You are a master cinematographer and composition expert. Analyze the scene and recommend the optimal visual composition for illustration.

Consider:
1. Emotional impact of the scene
2. Narrative significance
3. Visual hierarchy and focal points
4. Lighting and atmosphere
5. Color psychology

Return JSON with this structure:
{
    "composition_type": "close_up|medium_shot|wide_shot|establishing|dramatic|intimate",
    "focal_point": "what should be the visual center of attention",
    "background_elements": ["list", "of", "background", "elements"],
    "foreground_elements": ["list", "of", "foreground", "elements"],
    "lighting_mood": "golden_hour|dramatic|soft|mysterious|harsh|ethereal|natural",
    "atmosphere": "description of the overall mood and atmosphere",
    "color_palette_suggestion": "suggested color palette with emotional reasoning",
    "emotional_weight": 0.8
}"""

        try:
            # Prepare visual elements summary
            elements_summary = []
            for element in visual_elements:
                elements_summary.append(f"{element.element_type}: {element.description}")

            emotional_tones_str = ", ".join([tone.value for tone in emotional_moment.emotional_tones])

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""Scene: {emotional_moment.text_excerpt}

Context: {emotional_moment.context}

Visual elements identified:
{chr(10).join(elements_summary)}

Emotional tones: {emotional_tones_str}
Intensity: {emotional_moment.intensity_score}

Recommend the optimal composition for maximum visual and emotional impact.""")
            ]

            response = await self.llm.ainvoke(messages)
            composition_data = parse_llm_json(response.content)

            return SceneComposition(
                composition_type=CompositionType(composition_data.get('composition_type', 'medium_shot')),
                focal_point=composition_data.get('focal_point', 'central character'),
                background_elements=composition_data.get('background_elements', []),
                foreground_elements=composition_data.get('foreground_elements', []),
                lighting_mood=LightingMood(composition_data.get('lighting_mood', 'natural')),
                atmosphere=composition_data.get('atmosphere', 'emotionally resonant scene'),
                color_palette_suggestion=composition_data.get('color_palette_suggestion', 'balanced natural tones'),
                emotional_weight=float(composition_data.get('emotional_weight', 0.5))
            )

        except ValueError as parsing_error:
            logger.error(f"Scene composition analysis failed to parse JSON: {parsing_error}")
            return self._create_fallback_scene_composition(emotional_moment, visual_elements)
        except Exception as e:
            logger.error(f"Scene composition analysis failed: {e}")
            return self._create_fallback_scene_composition(emotional_moment, visual_elements)



class StyleTranslator:
    """Translates artistic styles and preferences into model-specific prompts with rich configuration support."""

    def __init__(self):
        # Load rich style configurations
        self.rich_style_configs = self._load_rich_style_configs()

        # Model-specific style vocabularies (fallback)
        self.dalle_vocabulary = {
            "artistic_styles": {
                "digital_painting": "digital painting, concept art, trending on artstation",
                "oil_painting": "oil painting, fine art, classical technique",
                "watercolor": "watercolor painting, soft edges, translucent colors",
                "pencil_sketch": "pencil sketch, graphite drawing, detailed line work",
                "photography": "photograph, professional photography, realistic"
            },
            "quality_modifiers": [
                "highly detailed", "masterpiece", "professional", "award-winning",
                "dramatic lighting", "perfect composition", "8k resolution"
            ],
            "negative_defaults": [
                "blurry", "low quality", "amateur", "distorted", "ugly"
            ]
        }

        self.imagen4_vocabulary = {
            "artistic_styles": {
                "digital_painting": "cinematic digital art, concept art, matte painting",
                "oil_painting": "classical oil painting, Renaissance technique",
                "watercolor": "delicate watercolor, flowing pigments, artistic brushstrokes",
                "pencil_sketch": "architectural sketch, technical drawing, fine line work",
                "photography": "cinematic photography, film photography, professional lighting"
            },
            "quality_modifiers": [
                "masterpiece quality", "ultra high resolution", "cinematic composition",
                "professional photography lighting", "dramatic atmosphere",
                "photorealistic", "stunning visual"
            ],
            "negative_defaults": [
                "blurry", "low quality", "distorted", "amateur", "text", "watermark"
            ]
        }

        self.flux_vocabulary = {
            "artistic_styles": {
                "digital_painting": "detailed digital illustration, concept art style",
                "oil_painting": "traditional oil painting, impasto technique",
                "watercolor": "watercolor illustration, soft organic shapes",
                "pencil_sketch": "detailed pencil drawing, crosshatching technique",
                "photography": "photorealistic rendering, studio lighting"
            },
            "quality_modifiers": [
                "highly detailed", "intricate artwork", "trending on artstation",
                "concept art", "professional illustration", "dynamic composition",
                "rich textures", "masterful technique"
            ],
            "negative_defaults": [
                "low quality", "blurred", "pixelated", "distorted", "amateur",
                "simple", "basic", "ugly", "text", "watermark"
            ]
        }

    def _load_rich_style_configs(self) -> Dict[str, Any]:
        """Load rich style configuration files like E.H. Shepard configs."""
        import json
        from pathlib import Path

        configs = {}

        # Load available rich style configurations
        config_files = [
            ("advanced_eh_shepard", "advanced_eh_shepard_config.json"),
            ("eh_shepard", "eh_shepard_pencil_config.json"),
        ]

        for config_name, file_path in config_files:
            config_path = Path(file_path)
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        configs[config_name] = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load config {file_path}: {e}")

        return configs

    def _get_emotional_style_modifiers(
        self,
        style_config: Dict[str, Any],
        emotional_tones: List[str]
    ) -> tuple[List[str], List[str]]:
        """Extract emotional style modifiers and atmosphere adjustments from rich configuration."""
        modifiers: List[str] = []
        atmosphere_notes: List[str] = []

        # Check if we have emotional adaptations in the config
        if "emotional_adaptations" in style_config:
            adaptations = style_config["emotional_adaptations"]

            for tone in emotional_tones:
                tone_lower = tone.lower()
                if tone_lower in adaptations:
                    adaptation = adaptations[tone_lower]
                    modifiers.extend(adaptation.get("style_modifiers", []))
                    atmosphere = adaptation.get("atmosphere_adjustments")
                    if atmosphere:
                        if isinstance(atmosphere, str):
                            atmosphere_notes.append(atmosphere)
                        else:
                            try:
                                atmosphere_notes.extend(list(atmosphere))
                            except TypeError:
                                atmosphere_notes.append(str(atmosphere))

        return modifiers, atmosphere_notes

    def _get_provider_optimizations(self, style_config: Dict[str, Any], provider: ImageProvider) -> Dict[str, Any]:
        """Get provider-specific optimizations from rich configuration."""
        provider_opts = {}

        if "provider_optimizations" in style_config:
            provider_key = provider.value.lower()
            if provider_key in style_config["provider_optimizations"]:
                provider_opts = style_config["provider_optimizations"][provider_key]

        return provider_opts

    def translate_style_config(
        self,
        style_config: Dict[str, Any],
        provider: ImageProvider,
        scene_composition: SceneComposition
    ) -> Dict[str, Any]:
        """Translate style configuration for specific provider with rich config support."""

        # Check if this is a rich configuration (has detailed style data)
        rich_config = self._detect_rich_configuration(style_config)

        if rich_config:
            return self._translate_rich_config(rich_config, style_config, provider, scene_composition)
        else:
            # Use standard translation
            if provider == ImageProvider.DALLE:
                return self._translate_for_dalle(style_config, scene_composition)
            elif provider == ImageProvider.IMAGEN4:
                return self._translate_for_imagen4(style_config, scene_composition)
            elif provider in (ImageProvider.FLUX, ImageProvider.SEEDREAM):
                return self._translate_for_flux(style_config, scene_composition)
            else:
                return self._generic_translation(style_config)

    def _detect_rich_configuration(self, style_config: Dict[str, Any]) -> Dict[str, Any]:
        """Detect if style config refers to a rich configuration."""

        # Check for explicit rich config references
        art_style = style_config.get("art_style", "").lower()
        style_name = style_config.get("style_name", "").lower()

        # Look for E.H. Shepard references
        if "shepard" in art_style or "shepard" in style_name or "pencil sketch" in art_style:
            if "advanced_eh_shepard" in self.rich_style_configs:
                return self.rich_style_configs["advanced_eh_shepard"]
            elif "eh_shepard" in self.rich_style_configs:
                return self.rich_style_configs["eh_shepard"]

        # Check if the style_config itself is already a rich configuration
        if "emotional_adaptations" in style_config and "base_prompt_modifiers" in style_config:
            return style_config

        return None

    def _translate_rich_config(
        self,
        rich_config: Dict[str, Any],
        style_config: Dict[str, Any],
        provider: ImageProvider,
        scene_composition: SceneComposition
    ) -> Dict[str, Any]:
        """Translate using rich configuration data."""

        # Start with base modifiers from rich config
        style_modifiers = list(rich_config.get("base_prompt_modifiers", []))
        atmosphere_guidance: List[str] = []

        # Add emotional adaptations if available
        if hasattr(scene_composition, 'emotional_tones'):
            emotional_modifiers, atmosphere_notes = self._get_emotional_style_modifiers(
                rich_config,
                scene_composition.emotional_tones
            )
            style_modifiers.extend(emotional_modifiers)
            atmosphere_guidance.extend(atmosphere_notes)

        # Get provider-specific optimizations
        provider_opts = self._get_provider_optimizations(rich_config, provider)

        # Merge technical parameters with provider overrides
        technical_params = dict(rich_config.get("technical_params", {}))
        if provider_opts.get("technical_adjustments"):
            technical_params.update(provider_opts["technical_adjustments"])

        # Deduplicate modifiers while preserving order
        seen_modifiers: set[str] = set()
        ordered_modifiers: List[str] = []
        for modifier in style_modifiers:
            if not modifier:
                continue
            normalized = modifier.strip()
            if normalized and normalized not in seen_modifiers:
                ordered_modifiers.append(normalized)
                seen_modifiers.add(normalized)

        # Build translation result
        translation = {
            "style_modifiers": ordered_modifiers,
            "negative_prompt": rich_config.get("negative_prompt", []),
            "technical_params": technical_params,
            "provider_optimizations": provider_opts,
            "atmosphere_guidance": atmosphere_guidance,
        }

        return translation

    def _translate_for_dalle(
        self,
        style_config: Dict[str, Any],
        scene_composition: SceneComposition
    ) -> Dict[str, Any]:
        """Translate style for DALL-E specific optimization."""
        vocabulary = self.dalle_vocabulary

        # Build style modifiers
        style_modifiers = []

        # Base style
        base_style = style_config.get('style_name', 'digital painting')
        if base_style.lower() in vocabulary['artistic_styles']:
            style_modifiers.append(vocabulary['artistic_styles'][base_style.lower()])
        else:
            style_modifiers.extend(style_config.get('base_prompt_modifiers', [base_style]))

        # Add composition-specific modifiers
        if scene_composition.composition_type == CompositionType.DRAMATIC:
            style_modifiers.extend(["dramatic angle", "dynamic perspective"])
        elif scene_composition.composition_type == CompositionType.INTIMATE:
            style_modifiers.extend(["intimate framing", "personal perspective"])
        elif scene_composition.composition_type == CompositionType.CLOSE_UP:
            style_modifiers.extend(["detailed portrait", "expressive close-up"])

        # Lighting modifiers
        lighting_map = {
            LightingMood.GOLDEN_HOUR: "warm golden hour lighting",
            LightingMood.DRAMATIC: "dramatic chiaroscuro lighting",
            LightingMood.SOFT: "soft diffused lighting",
            LightingMood.MYSTERIOUS: "atmospheric mood lighting",
            LightingMood.ETHEREAL: "ethereal glowing light"
        }

        if scene_composition.lighting_mood in lighting_map:
            style_modifiers.append(lighting_map[scene_composition.lighting_mood])

        # Add quality modifiers (limited for DALL-E)
        style_modifiers.extend(vocabulary['quality_modifiers'][:3])

        # Technical parameters optimized for DALL-E (OpenAI Images API)
        technical_params = style_config.get('technical_params', {})
        technical_params.update({
            "model": "gpt-image-1",
            "quality": "hd",
            "size": "1024x1024",
            "style": "vivid" if scene_composition.emotional_weight > 0.7 else "natural"
        })

        # Negative prompt (DALL-E doesn't use them, but stored for consistency)
        negative_prompt = style_config.get('negative_prompt', [])
        if not negative_prompt:
            negative_prompt = vocabulary['negative_defaults']

        return {
            "style_modifiers": style_modifiers,
            "technical_params": technical_params,
            "negative_prompt": negative_prompt,
            "provider_optimizations": {
                "narrative_focus": True,
                "emotional_clarity": True,
                "artistic_coherence": True
            }
        }

    def _translate_for_imagen4(
        self,
        style_config: Dict[str, Any],
        scene_composition: SceneComposition
    ) -> Dict[str, Any]:
        """Translate style for Imagen4 specific optimization."""
        vocabulary = self.imagen4_vocabulary

        style_modifiers = []

        # Base style with cinematic emphasis
        base_style = style_config.get('style_name', 'digital painting')
        if base_style.lower() in vocabulary['artistic_styles']:
            style_modifiers.append(vocabulary['artistic_styles'][base_style.lower()])
        else:
            style_modifiers.extend(style_config.get('base_prompt_modifiers', [base_style]))

        # Imagen4 excels at photorealistic and cinematic styles
        style_modifiers.extend([
            "cinematic composition",
            "professional cinematography",
            scene_composition.color_palette_suggestion
        ])

        # Advanced lighting for Imagen4
        lighting_modifiers = {
            LightingMood.GOLDEN_HOUR: "warm golden hour cinematography, rim lighting",
            LightingMood.DRAMATIC: "dramatic film lighting, strong shadows, chiaroscuro",
            LightingMood.SOFT: "soft box lighting, even illumination, gentle shadows",
            LightingMood.MYSTERIOUS: "atmospheric lighting, volumetric fog, mysterious ambiance",
            LightingMood.ETHEREAL: "ethereal backlighting, glowing atmosphere, dreamy quality"
        }

        if scene_composition.lighting_mood in lighting_modifiers:
            style_modifiers.append(lighting_modifiers[scene_composition.lighting_mood])

        # High-quality modifiers
        style_modifiers.extend(vocabulary['quality_modifiers'])

        # Technical parameters for Imagen4
        technical_params = {
            "aspect_ratio": "1:1",
            "safety_filter_level": "block_most",
            "seed": None,
            "guidance_scale": 15 if scene_composition.emotional_weight > 0.7 else 10
        }

        # Enhanced negative prompt
        negative_prompt = list(vocabulary['negative_defaults'])
        if style_config.get('negative_prompt'):
            negative_prompt.extend(style_config['negative_prompt'])

        # Add composition-specific negative prompts
        if scene_composition.composition_type == CompositionType.INTIMATE:
            negative_prompt.extend(["crowded", "busy background", "distracting elements"])

        return {
            "style_modifiers": style_modifiers,
            "technical_params": technical_params,
            "negative_prompt": negative_prompt,
            "provider_optimizations": {
                "cinematic_quality": True,
                "photorealistic_detail": True,
                "atmospheric_depth": True
            }
        }

    def _translate_for_flux(
        self,
        style_config: Dict[str, Any],
        scene_composition: SceneComposition
    ) -> Dict[str, Any]:
        """Translate style for Flux specific optimization."""
        vocabulary = self.flux_vocabulary

        style_modifiers = []

        # Base artistic style
        base_style = style_config.get('style_name', 'digital painting')
        if base_style.lower() in vocabulary['artistic_styles']:
            style_modifiers.append(vocabulary['artistic_styles'][base_style.lower()])
        else:
            style_modifiers.extend(style_config.get('base_prompt_modifiers', [base_style]))

        # Flux excels at artistic detail and style flexibility
        style_modifiers.extend([
            "masterful artistic technique",
            "rich artistic detail",
            "expressive brushwork" if "painting" in base_style.lower() else "precise technique"
        ])

        # Composition-based artistic modifiers
        composition_modifiers = {
            CompositionType.DRAMATIC: ["dynamic composition", "bold artistic statement"],
            CompositionType.INTIMATE: ["intimate artistic portrayal", "personal artistic vision"],
            CompositionType.WIDE_SHOT: ["expansive artistic landscape", "sweeping composition"],
            CompositionType.CLOSE_UP: ["detailed artistic study", "focused artistic expression"]
        }

        if scene_composition.composition_type in composition_modifiers:
            style_modifiers.extend(composition_modifiers[scene_composition.composition_type])

        # Emotional artistic interpretation
        emotion_modifiers = {
            "joy": "vibrant artistic energy, uplifting color harmony",
            "sadness": "melancholic artistic mood, somber tonal palette",
            "fear": "dark artistic atmosphere, unsettling visual elements",
            "anger": "intense artistic expression, bold contrasting elements",
            "mystery": "atmospheric artistic ambiance, subtle artistic details"
        }

        # Add emotional artistic guidance based on scene
        for emotion_key, modifier in emotion_modifiers.items():
            if emotion_key in scene_composition.atmosphere.lower():
                style_modifiers.append(modifier)
                break

        # Quality and technique modifiers
        style_modifiers.extend(vocabulary['quality_modifiers'])

        # Technical parameters optimized for Flux
        technical_params = {
            "guidance_scale": 8.0 if scene_composition.emotional_weight > 0.8 else 7.5,
            "num_inference_steps": 32 if scene_composition.composition_type == CompositionType.DRAMATIC else 28,
            "width": 1024,
            "height": 1024,
        }

        # Comprehensive negative prompt
        negative_prompt = list(vocabulary['negative_defaults'])
        if style_config.get('negative_prompt'):
            negative_prompt.extend(style_config['negative_prompt'])

        # Add artistic quality negatives
        negative_prompt.extend([
            "poor artistic technique",
            "amateur artistic attempt",
            "inconsistent artistic style"
        ])

        return {
            "style_modifiers": style_modifiers,
            "technical_params": technical_params,
            "negative_prompt": negative_prompt,
            "provider_optimizations": {
                "artistic_flexibility": True,
                "detailed_technique": True,
                "style_consistency": True,
                "creative_interpretation": True
            }
        }

    def _generic_translation(self, style_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generic fallback translation."""
        return {
            "style_modifiers": style_config.get('base_prompt_modifiers', ['digital art']),
            "technical_params": style_config.get('technical_params', {}),
            "negative_prompt": style_config.get('negative_prompt', ['low quality']),
            "provider_optimizations": {}
        }




class PromptEngineer:
    """Master prompt engineering system orchestrating all components."""

    def __init__(self, llm: BaseChatModel, character_tracker: Optional[CharacterTracker] = None):
        self.llm = llm
        self.scene_analyzer = SceneAnalyzer(llm)
        self.style_translator = StyleTranslator()
        self.character_tracker = character_tracker or CharacterTracker(llm)

        # Context tracking
        self.character_profiles: Dict[str, CharacterProfile] = {}
        self.setting_memory: Dict[str, str] = {}
        self.narrative_context: List[str] = []

    async def engineer_prompt(
        self,
        emotional_moment: EmotionalMoment,
        provider: ImageProvider,
        style_preferences: Dict[str, Any],
        chapter_context: Chapter,
        previous_scenes: List[Dict] = None
    ) -> IllustrationPrompt:
        """Engineer an optimal prompt using all available techniques with character consistency."""

        # Update character tracking for this chapter
        await self.character_tracker.extract_characters_from_chapter(chapter_context, update_profiles=True)

        # Store current chapter number for character context
        self._current_chapter_number = chapter_context.number

        # Update context tracking
        await self._update_context_tracking(emotional_moment, chapter_context, previous_scenes or [])

        # Analyze scene composition and visual elements
        scene_composition, visual_elements = await self.scene_analyzer.analyze_scene(
            emotional_moment,
            chapter_context,
            self.character_profiles
        )

        # Translate style preferences for the specific provider
        style_translation = self.style_translator.translate_style_config(
            style_preferences.get('style_config', style_preferences),
            provider,
            scene_composition
        )

        # Build the comprehensive prompt
        prompt_text = await self._build_comprehensive_prompt(
            emotional_moment,
            scene_composition,
            visual_elements,
            style_translation,
            provider
        )

        return IllustrationPrompt(
            provider=provider,
            prompt=prompt_text,
            style_modifiers=style_translation['style_modifiers'],
            negative_prompt=", ".join(style_translation['negative_prompt']) if style_translation['negative_prompt'] else None,
            technical_params=style_translation['technical_params']
        )

    async def _update_context_tracking(
        self,
        emotional_moment: EmotionalMoment,
        chapter: Chapter,
        previous_scenes: List[Dict]
    ):
        """Update character and setting continuity tracking."""

        # Extract character information
        character_system_prompt = """Extract character information from this scene. Focus on:
1. Character names mentioned
2. Physical descriptions
3. Emotional states
4. Actions and poses
5. Clothing or appearance details

Return JSON: {"characters": [{"name": "character_name", "description": "physical and emotional details", "action": "what they're doing"}]}"""

        try:
            messages = [
                SystemMessage(content=character_system_prompt),
                HumanMessage(content=f"Scene: {emotional_moment.text_excerpt}\nContext: {emotional_moment.context}")
            ]

            response = await self.llm.ainvoke(messages)
            character_data = json.loads(response.content.strip())

            # Update character profiles
            for char_info in character_data.get('characters', []):
                name = char_info.get('name', '').lower()
                if name and name not in ['he', 'she', 'they', 'it']:
                    if name not in self.character_profiles:
                        self.character_profiles[name] = CharacterProfile(
                            name=name,
                            physical_description=char_info.get('description', ''),
                            emotional_state=emotional_moment.emotional_tones,
                            current_action=char_info.get('action', ''),
                            relationship_context='',
                            consistency_notes=[]
                        )
                    else:
                        # Update existing profile
                        profile = self.character_profiles[name]
                        profile.emotional_state = emotional_moment.emotional_tones
                        profile.current_action = char_info.get('action', profile.current_action)

                        # Add consistency notes if description differs significantly
                        new_desc = char_info.get('description', '')
                        if new_desc and new_desc != profile.physical_description:
                            profile.consistency_notes.append(f"Ch{chapter.number}: {new_desc}")

        except Exception:
            pass  # Continue with existing context if extraction fails

        # Update setting memory
        setting_key = f"chapter_{chapter.number}"
        if setting_key not in self.setting_memory:
            self.setting_memory[setting_key] = emotional_moment.context

    async def _build_comprehensive_prompt(
        self,
        emotional_moment: EmotionalMoment,
        scene_composition: SceneComposition,
        visual_elements: List[VisualElement],
        style_translation: Dict[str, Any],
        provider: ImageProvider
    ) -> str:
        """Build the final comprehensive prompt."""

        prompt_parts = []

        # Scene description with visual focus
        scene_desc = await self._enhance_scene_description(
            emotional_moment.text_excerpt,
            visual_elements,
            scene_composition
        )
        prompt_parts.append(scene_desc)

        # Composition guidance
        composition_guidance = self._build_composition_guidance(scene_composition)
        prompt_parts.append(composition_guidance)

        # Character consistency
        character_guidance = self._build_character_guidance(visual_elements)
        if character_guidance:
            prompt_parts.append(character_guidance)

        # Atmospheric and emotional guidance
        atmospheric_guidance = self._build_atmospheric_guidance(
            emotional_moment,
            scene_composition
        )
        prompt_parts.append(atmospheric_guidance)

        # Style modifiers (handle tuples properly)
        style_modifiers_formatted = []
        for m in style_translation['style_modifiers']:
            if isinstance(m, tuple):
                # For tuples, join the elements with spaces
                style_modifiers_formatted.append(" ".join(str(elem) for elem in m))
            else:
                style_modifiers_formatted.append(str(m))
        style_modifiers_text = ", ".join(style_modifiers_formatted)
        prompt_parts.append(style_modifiers_text)

        # Provider-specific stylistic emphasis
        provider_opts = style_translation.get('provider_optimizations') or {}
        style_emphasis = provider_opts.get('style_emphasis')
        if style_emphasis and style_emphasis not in style_modifiers_formatted:
            prompt_parts.append(style_emphasis)

        quality_modifiers = provider_opts.get('quality_modifiers') or []
        if quality_modifiers:
            prompt_parts.append(
                "Quality focus: " + ", ".join(str(mod) for mod in quality_modifiers if mod)
            )

        # Emotional atmosphere guidance from rich config
        atmosphere_guidance = style_translation.get('atmosphere_guidance') or []
        if atmosphere_guidance:
            prompt_parts.append(
                "Atmospheric adjustments: " + "; ".join(atmosphere_guidance)
            )

        # Technical guidance expressed textually for providers lacking direct parameters
        technical_params = style_translation.get('technical_params') or {}
        if technical_params:
            formatted_params: List[str] = []
            for key, value in technical_params.items():
                if value in (None, ""):
                    continue
                human_key = key.replace('_', ' ')
                if isinstance(value, (int, float)):
                    formatted_params.append(f"{human_key}: {value}")
                else:
                    formatted_params.append(f"{human_key}: {value}")
            if formatted_params:
                prompt_parts.append(
                    "Technical focus: " + "; ".join(formatted_params)
                )

        # Join all parts
        comprehensive_prompt = ". ".join(prompt_parts)

        # Final optimization pass
        optimized_prompt = await self._optimize_prompt_for_provider(
            comprehensive_prompt,
            provider,
            style_translation.get('provider_optimizations', {})
        )

        # Enforce conservative provider prompt length limits
        try:
            from illustrator.utils import enforce_prompt_length
            optimized_prompt = enforce_prompt_length(provider.value, optimized_prompt)
        except Exception:
            pass

        return optimized_prompt

    async def generate_chapter_header_options(
        self,
        chapter: Chapter,
        style_preferences: Dict[str, Any],
        provider: ImageProvider = ImageProvider.DALLE
    ) -> List[ChapterHeaderOption]:
        """Generate 4 chapter header illustration options using comprehensive content analysis."""

        # Analyze the full chapter content for visual themes and elements
        analysis_prompt = f"""
        Analyze this complete chapter and create 4 distinct header illustration options that capture different visual aspects of the chapter's content, themes, and narrative elements.

        Chapter Title: {chapter.title}
        Chapter Content: {chapter.content}

        For each header option, deeply analyze the chapter content to identify:
        1. Key visual scenes and moments
        2. Thematic symbols and metaphors present in the text
        3. Character descriptions and emotional states
        4. Environmental settings and atmospheric details
        5. Narrative conflicts and dramatic moments

        Create exactly 4 options with these focuses:
        Option 1: Symbolic/Metaphorical - Extract symbolic elements and themes from the text
        Option 2: Character-focused - Based on character descriptions and emotional moments in the chapter
        Option 3: Environmental/Atmospheric - Based on setting descriptions and environmental details
        Option 4: Dramatic moment - Based on a key conflict or dramatic scene from the chapter

        For each option, provide a detailed analysis of the specific textual elements that inform the visual design.

        Return as JSON with this structure:
        {{
            "options": [
                {{
                    "option_number": 1,
                    "title": "Specific title based on chapter content",
                    "description": "Detailed description referencing specific text passages",
                    "visual_focus": "Specific visual element extracted from chapter text",
                    "artistic_style": "Style appropriate to the content",
                    "composition_notes": "Specific composition based on textual analysis",
                    "key_textual_references": ["specific quotes or passages that inform this design"],
                    "emotional_tone": "emotion derived from chapter analysis",
                    "color_palette": "colors suggested by chapter content and mood",
                    "detailed_scene_elements": "specific environmental and character details from text"
                }},
                ...
            ]
        }}
        """

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="You are an expert visual artist and literary analyst who creates detailed illustration concepts based on deep textual analysis."),
                HumanMessage(content=analysis_prompt)
            ])

            analysis_data = parse_llm_json(response.content)
            header_options = []

            for i, option_data in enumerate(analysis_data.get("options", [])):
                # Create a mock emotional moment for each header option based on the analysis
                header_emotional_moment = EmotionalMoment(
                    text_excerpt=option_data.get('detailed_scene_elements', option_data['visual_focus']),
                    context=f"Chapter header for '{chapter.title}' - {option_data['description']}",
                    emotional_tones=[EmotionalTone(option_data.get('emotional_tone', 'anticipation').upper()) if option_data.get('emotional_tone', 'anticipation').upper() in [e.name for e in EmotionalTone] else EmotionalTone.ANTICIPATION],
                    intensity_score=0.7
                )

                # Create enhanced style preferences for this specific header
                header_style_preferences = {
                    **style_preferences,
                    'style_config': {
                        'style_name': option_data['artistic_style'],
                        'base_prompt_modifiers': [
                            option_data['artistic_style'],
                            "chapter header illustration",
                            "horizontal composition",
                            "book illustration style"
                        ],
                        'technical_params': {
                            "aspect_ratio": "16:9",
                            "style": "artistic",
                            "quality": "high"
                        }
                    }
                }

                # Generate the sophisticated prompt using the main prompt engineering system
                illustration_prompt = await self.engineer_prompt(
                    header_emotional_moment,
                    provider,
                    header_style_preferences,
                    chapter,
                    []  # No previous scenes for headers
                )

                # Create the header option with the advanced prompt
                header_option = ChapterHeaderOption(
                    option_number=i + 1,
                    title=option_data['title'],
                    description=option_data['description'],
                    visual_focus=option_data['visual_focus'],
                    artistic_style=option_data['artistic_style'],
                    composition_notes=option_data['composition_notes'],
                    prompt=illustration_prompt
                )
                header_options.append(header_option)

            return header_options

        except Exception as e:
            print(f"Error generating advanced chapter headers: {e}")
            # Create informed fallback options using chapter content analysis
            return await self._create_content_aware_header_options(chapter, style_preferences, provider)

    async def _create_content_aware_header_options(self, chapter: Chapter, style_preferences: Dict[str, Any], provider: ImageProvider) -> List[ChapterHeaderOption]:
        """Create content-aware header options when advanced analysis fails."""
        # Extract key elements from chapter content
        content_sample = chapter.content[:1000]  # Use more content than the old fallback

        # Simple content analysis to inform the options
        content_themes = {
            "symbolic": "abstract representation of chapter themes",
            "character": f"character elements from the chapter content: {content_sample[:200]}",
            "environmental": f"setting and atmosphere based on: {content_sample[:200]}",
            "dramatic": f"key dramatic moment from chapter: {content_sample[:200]}"
        }

        styles = ["symbolic watercolor", "character portrait", "atmospheric landscape", "dramatic scene"]
        focuses = ["symbolic", "character-driven", "environmental", "dramatic"]

        header_options = []
        for i in range(4):
            # Create a more informed emotional moment
            header_emotional_moment = EmotionalMoment(
                text_excerpt=content_themes[focuses[i]],
                context=f"Chapter header for '{chapter.title}' with {focuses[i]} focus",
                emotional_tones=[EmotionalTone.ANTICIPATION],
                intensity_score=0.6
            )

            # Use the main prompt engineering system even for fallback
            try:
                illustration_prompt = await self.engineer_prompt(
                    header_emotional_moment,
                    provider,
                    {
                        **style_preferences,
                        'style_config': {
                            'style_name': styles[i],
                            'base_prompt_modifiers': [styles[i], "chapter header", "horizontal composition"]
                        }
                    },
                    chapter,
                    []
                )
            except Exception:
                # Last resort basic prompt if everything fails
                illustration_prompt = IllustrationPrompt(
                    provider=provider,
                    prompt=f"Chapter header illustration for '{chapter.title}', {styles[i]} style, {focuses[i]} approach",
                    style_modifiers=[styles[i], "chapter header", "artistic"],
                    negative_prompt="text, words, low quality",
                    technical_params={"aspect_ratio": "16:9", "quality": "high"}
                )

            option = ChapterHeaderOption(
                option_number=i + 1,
                title=f"{focuses[i].title()} Header",
                description=f"A {focuses[i]} representation based on chapter content",
                visual_focus=content_themes[focuses[i]],
                artistic_style=styles[i],
                composition_notes="Horizontal header composition with chapter-specific elements",
                prompt=illustration_prompt
            )
            header_options.append(option)

        return header_options

    async def _enhance_scene_description(
        self,
        original_text: str,
        visual_elements: List[VisualElement],
        scene_composition: SceneComposition
    ) -> str:
        """Enhance scene description with comprehensive visual and artistic details."""
        enhancement_prompt = f"""
        Transform this literary text into a richly detailed visual scene description for E.H. Shepard-style classic book illustration generation.

        Original text: "{original_text}"
        Visual elements: {[f"{elem.element_type}: {elem.description}" for elem in visual_elements]}
        Scene focus: {scene_composition.focal_point}
        Composition: {scene_composition.composition_type.value}
        Atmosphere: {getattr(scene_composition, 'atmosphere', 'natural')}

        Create an exceptionally detailed scene description following this structure:

        1. PRECISE SETTING & ENVIRONMENT:
        - Exact location with architectural specifics (cozy coffee shop, intimate café interior, small friendly establishment)
        - Detailed background elements (espresso machine, displayed pastries, counter details, café furniture, menu boards)
        - Environmental textures and materials (wooden counter, ceramic cups, glass display cases, warm lighting fixtures)
        - Spatial relationships and scene layout with depth

        2. HIGHLY SPECIFIC CHARACTER DETAILS:
        - Exact positioning and body posture (behind counter, leaning slightly forward, turned at angle)
        - Precise facial expressions with emotional specificity (wide eyes, startled expression, frozen smile, stiff posture)
        - Detailed body language conveying psychological state (recoiling slightly, tense shoulders, hands positioned defensively)
        - Character interactions and spatial dynamics (customer unaware, barista's fearful gaze, intimate yet tense moment)
        - Clothing and appearance details that enhance character portrayal

        3. DETAILED ARTISTIC COMPOSITION:
        - Specific viewpoint and framing (intimate scene, everyday moment interrupted by tension)
        - Foreground elements (main characters, counter interaction) vs background elements (café atmosphere, environmental details)
        - Visual flow guiding viewer attention to emotional focal points
        - Compositional elements that enhance the narrative tension

        4. RICH EMOTIONAL ATMOSPHERE:
        - Specific emotional descriptors (startled fear, subtle tension, everyday atmosphere with underlying unease)
        - Environmental reinforcement of mood (warm café setting contrasting with psychological tension)
        - Psychological elements visible through visual storytelling
        - Emotional contrast between characters (casual customer vs frightened barista)

        5. ARTISTIC TECHNIQUE INTEGRATION:
        - References to pencil sketch qualities (fine linework, gentle shading, crosshatching opportunities)
        - E.H. Shepard-style elements (expressive faces, detailed environmental textures, whimsical yet realistic approach)
        - Classic book illustration aesthetics (intimate scenes, character-focused compositions)

        EXAMPLE QUALITY LEVEL:
        "Inside a cozy coffee shop interior, a young female barista stands behind the wooden counter, her body language betraying startled fear as she gazes at the customer she just served. Her smile is frozen and artificial, eyes wide with unmistakable alarm, shoulders tensed as she leans slightly back from the counter. The customer, a casual figure positioned to the side, remains unaware of her frightened expression, creating a moment of psychological tension within the intimate café setting. The scene captures everyday café details - espresso machine humming softly, ceramic cups arranged on shelves, pastries displayed in glass cases, small wooden tables with simple chairs - all rendered with the gentle, detailed approach characteristic of classic book illustrations. The atmosphere balances warmth and comfort with subtle underlying unease, perfect for expressive pencil work with fine crosshatching and delicate shading."

        Generate a description of this caliber with equivalent detail density and emotional specificity.
        """

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="You are an expert visual artist who creates detailed scene descriptions for classic book illustrations, specializing in capturing both visual elements and emotional nuance."),
                HumanMessage(content=enhancement_prompt)
            ])
            enhanced_description = response.content.strip()

            # Ensure we have a substantive description
            if len(enhanced_description) < 50:
                raise ValueError("AI generated description is too brief - insufficient content analysis")

            return enhanced_description

        except Exception as e:
            logger.error(f"Scene description enhancement failed: {e}")
            raise ValueError(f"Failed to enhance scene description: {str(e)}")


    def _build_composition_guidance(self, scene_composition: SceneComposition) -> str:
        """Build composition and framing guidance."""
        guidance_parts = []

        # Composition type
        comp_guidance = {
            CompositionType.CLOSE_UP: "intimate close-up composition focusing on emotional expression",
            CompositionType.MEDIUM_SHOT: "medium shot composition showing character and immediate environment",
            CompositionType.WIDE_SHOT: "wide establishing shot showing full scene and environment",
            CompositionType.DRAMATIC: "dramatic composition with dynamic angles and visual tension",
            CompositionType.INTIMATE: "intimate composition creating personal connection with viewer"
        }

        guidance_parts.append(comp_guidance.get(
            scene_composition.composition_type,
            "balanced composition with clear visual hierarchy"
        ))

        # Background/foreground structure
        if scene_composition.background_elements:
            guidance_parts.append(f"Background: {', '.join(scene_composition.background_elements)}")

        if scene_composition.foreground_elements:
            guidance_parts.append(f"Foreground: {', '.join(scene_composition.foreground_elements)}")

        return ". ".join(guidance_parts)

    def _build_character_guidance(self, visual_elements: List[VisualElement]) -> str:
        """Build advanced character consistency guidance using the character tracker."""
        character_elements = [elem for elem in visual_elements if elem.element_type == "character"]

        if not character_elements:
            return ""

        guidance_parts = []

        for char_elem in character_elements:
            char_name = char_elem.attributes.get('name', '')

            if char_name:
                # Get character description from the advanced tracker
                char_description = self.character_tracker.get_character_for_illustration(
                    char_name,
                    context_chapter=getattr(self, '_current_chapter_number', None)
                )

                if char_description:
                    # Build comprehensive character guidance
                    char_guidance_parts = []

                    # Physical consistency
                    if char_description.get('physical_summary'):
                        char_guidance_parts.append(f"Character {char_name}: {char_description['physical_summary']}")

                    # Distinctive features
                    if char_description.get('distinctive_features'):
                        char_guidance_parts.append(f"distinctive features: {char_description['distinctive_features']}")

                    # Clothing consistency
                    if char_description.get('typical_clothing'):
                        char_guidance_parts.append(f"typically wearing: {char_description['typical_clothing']}")

                    # Context-specific appearance
                    if char_description.get('context_appearance'):
                        char_guidance_parts.append(f"context appearance: {char_description['context_appearance']}")

                    # Emotional default
                    if char_description.get('emotional_default'):
                        char_guidance_parts.append(f"emotional state: {char_description['emotional_default']}")

                    # Illustration notes
                    if char_description.get('illustration_notes'):
                        char_guidance_parts.append(f"illustration notes: {char_description['illustration_notes']}")

                    # Add consistency warning if score is low
                    consistency_score = float(char_description.get('consistency_score', '1.0'))
                    if consistency_score < 0.8:
                        char_guidance_parts.append("(maintain visual consistency with previous appearances)")

                    if char_guidance_parts:
                        guidance_parts.append("; ".join(char_guidance_parts))

            # Fallback to basic character description from visual elements
            elif not char_name and char_elem.description:
                guidance_parts.append(f"Character appearance: {char_elem.description}")

        if guidance_parts:
            return "Character consistency guidance: " + ". ".join(guidance_parts)

        return ""

    def _build_atmospheric_guidance(
        self,
        emotional_moment: EmotionalMoment,
        scene_composition: SceneComposition
    ) -> str:
        """Build atmospheric and emotional guidance."""
        guidance_parts = []

        # Lighting guidance
        lighting_guidance = {
            LightingMood.GOLDEN_HOUR: "warm golden hour lighting with soft shadows",
            LightingMood.DRAMATIC: "dramatic lighting with strong contrasts and deep shadows",
            LightingMood.SOFT: "soft diffused lighting creating gentle atmosphere",
            LightingMood.MYSTERIOUS: "atmospheric lighting with mystery and depth",
            LightingMood.ETHEREAL: "ethereal lighting with glowing, dreamlike quality"
        }

        guidance_parts.append(lighting_guidance.get(
            scene_composition.lighting_mood,
            "natural lighting appropriate to scene"
        ))

        # Atmospheric description
        guidance_parts.append(scene_composition.atmosphere)

        # Color guidance
        guidance_parts.append(scene_composition.color_palette_suggestion)

        # Emotional intensity guidance
        intensity_guidance = {
            (0.8, 1.0): "highly emotional and impactful scene",
            (0.6, 0.8): "emotionally resonant scene",
            (0.4, 0.6): "moderate emotional content",
            (0.0, 0.4): "subtle emotional undertones"
        }

        for (min_val, max_val), description in intensity_guidance.items():
            if min_val <= scene_composition.emotional_weight <= max_val:
                guidance_parts.append(description)
                break

        return ". ".join(guidance_parts)

    async def _optimize_prompt_for_provider(
        self,
        prompt: str,
        provider: ImageProvider,
        optimizations: Dict[str, Any]
    ) -> str:
        """Final provider-specific prompt optimization with rich artistic detail."""

        # Load artistic technique specifications
        artistic_techniques = {
            "pencil_sketch": {
                "techniques": ["fine crosshatching", "soft graphite shading", "delicate line work", "expressive facial features"],
                "quality_terms": ["detailed linework", "classic book illustration aesthetic", "gentle flowing lines", "soft pencil textures"]
            },
            "digital_painting": {
                "techniques": ["rich brush strokes", "atmospheric lighting", "detailed texturing", "cinematic composition"],
                "quality_terms": ["high resolution", "professional digital art", "masterful technique", "artistic excellence"]
            }
        }

        # Provider-specific enhancements with rich E.H. Shepard artistic detail integration
        if provider == ImageProvider.DALLE:
            enhanced_parts = []

            # Check for E.H. Shepard style configuration
            if any(keyword in prompt.lower() for keyword in ["pencil sketch", "shepard", "crosshatching", "hand-drawn", "line work"]):
                style_prefix = "A black-and-white pencil sketch in the style of E.H. Shepard."
                enhanced_parts.append(style_prefix)
                enhanced_parts.append(prompt)
                enhanced_parts.extend([
                    "The atmosphere is intimate and everyday, with charming sketchy detail rendered in fine pencil linework.",
                    "Expressive and whimsical style with delicate crosshatching, gentle shading, capturing subtle emotional tension in the scene.",
                    "Classic children's book illustration aesthetic with detailed environmental textures and characterful expression."
                ])
            else:
                enhanced_parts = [f"A masterfully crafted illustration: {prompt}"]
                enhanced_parts.append("Professional artistic illustration with detailed environmental and emotional elements")

        elif provider == ImageProvider.IMAGEN4:
            enhanced_parts = []

            if any(keyword in prompt.lower() for keyword in ["pencil sketch", "shepard", "crosshatching", "hand-drawn", "line work"]):
                style_prefix = "Pencil illustration, drawn in the expressive and whimsical style of E.H. Shepard."
                enhanced_parts.append(style_prefix)
                enhanced_parts.append(prompt)
                enhanced_parts.extend([
                    "The scene should feel like an ordinary moment with charming sketchy detail — rendered in fine pencil linework.",
                    "Classic book illustration with detailed crosshatching, soft graphite shading, expressive character faces.",
                    "Traditional British illustration aesthetic balancing whimsy with emotional depth and environmental detail."
                ])
            else:
                enhanced_parts = [f"Cinematic artistic scene with detailed environmental and emotional elements: {prompt}"]
                enhanced_parts.append("Professional artistic rendering with rich atmospheric detail")

        elif provider in (ImageProvider.FLUX, ImageProvider.SEEDREAM):
            enhanced_parts = []

            if any(keyword in prompt.lower() for keyword in ["pencil sketch", "shepard", "crosshatching", "hand-drawn", "line work"]):
                style_prefix = "A natural pencil sketch illustration in the classic E.H. Shepard style."
                enhanced_parts.append(style_prefix)
                enhanced_parts.append(prompt)
                enhanced_parts.extend([
                    "The sketch should balance whimsy and emotional depth, with fine crosshatching, detailed environmental textures, and expressive, characterful linework.",
                    "Classic children's book illustration technique with delicate pencil work, gentle shading, and intimate scene composition.",
                    "Traditional book illustration aesthetic with detailed character expressions and rich environmental storytelling."
                ])
            else:
                enhanced_parts = [f"Masterful artistic interpretation with detailed environmental and character elements: {prompt}"]
                enhanced_parts.append("Superior artistic technique with rich visual storytelling")

        else:
            enhanced_parts = [prompt]

        # Combine all parts
        enhanced_prompt = " ".join(enhanced_parts)

        # Ensure optimal length while preserving detail
        max_length = {
            ImageProvider.DALLE: 400,
            ImageProvider.IMAGEN4: 500,
            ImageProvider.FLUX: 600,
            ImageProvider.SEEDREAM: 600,
        }.get(provider, 400)

        if len(enhanced_prompt) > max_length:
            # Truncate while preserving essential artistic elements
            essential_parts = enhanced_parts[0]  # Keep the style prefix and main description
            if len(essential_parts) <= max_length:
                enhanced_prompt = essential_parts
            else:
                enhanced_prompt = essential_parts[:max_length-3] + "..."

        return enhanced_prompt
