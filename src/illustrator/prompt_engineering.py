"""Advanced prompt engineering system for optimal text-to-image generation."""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from illustrator.models import (
    EmotionalMoment,
    EmotionalTone,
    IllustrationPrompt,
    ImageProvider,
    Chapter,
)


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

        system_prompt = """You are a visual scene analyst. Extract detailed visual elements from the provided text that would be essential for creating an illustration. Focus on:

1. Characters: Physical appearance, expressions, poses, clothing, emotional state
2. Environment: Setting details, lighting conditions, atmosphere, weather
3. Objects: Important items, props, symbolic elements
4. Composition: Spatial relationships, perspective, focal points

Return a JSON array of visual elements with this structure:
[
    {
        "element_type": "character|environment|object|atmosphere",
        "description": "detailed description",
        "importance": 0.9,
        "attributes": {
            "specific_details": "any specific visual details",
            "emotional_significance": "why this element matters to the scene"
        }
    }
]

Focus on elements that would be visually compelling and narratively significant."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""Scene text: {text}

Context: {context}

Chapter setting: Chapter {chapter.number} - {chapter.title}

Extract the most important visual elements for illustration.""")
            ]

            response = await self.llm.ainvoke(messages)
            elements_data = json.loads(response.content.strip())

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
            # Fallback to pattern-based extraction
            return self._pattern_based_extraction(text, context)

    def _pattern_based_extraction(self, text: str, context: str) -> List[VisualElement]:
        """Fallback pattern-based visual element extraction."""
        elements = []
        text_lower = text.lower()

        # Extract characters
        character_matches = []
        for pattern in self.character_patterns:
            matches = re.findall(pattern, text_lower)
            character_matches.extend(matches)

        if character_matches:
            elements.append(VisualElement(
                element_type="character",
                description=f"Character elements: {', '.join(set(character_matches))}",
                importance=0.8,
                attributes={"detected_features": character_matches}
            ))

        # Extract environment
        env_matches = []
        for pattern in self.environment_patterns:
            matches = re.findall(pattern, text_lower)
            env_matches.extend(matches)

        if env_matches:
            elements.append(VisualElement(
                element_type="environment",
                description=f"Environmental elements: {', '.join(set(env_matches))}",
                importance=0.7,
                attributes={"detected_elements": env_matches}
            ))

        return elements

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
            composition_data = json.loads(response.content.strip())

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

        except Exception:
            # Fallback composition based on emotional tones
            return self._fallback_composition(emotional_moment)

    def _fallback_composition(self, emotional_moment: EmotionalMoment) -> SceneComposition:
        """Generate fallback composition based on emotional analysis."""
        primary_emotion = emotional_moment.emotional_tones[0] if emotional_moment.emotional_tones else EmotionalTone.ANTICIPATION

        # Emotion-based composition defaults
        composition_map = {
            EmotionalTone.JOY: (CompositionType.MEDIUM_SHOT, LightingMood.GOLDEN_HOUR, "warm and uplifting"),
            EmotionalTone.SADNESS: (CompositionType.CLOSE_UP, LightingMood.SOFT, "melancholic and introspective"),
            EmotionalTone.FEAR: (CompositionType.WIDE_SHOT, LightingMood.DRAMATIC, "tense and foreboding"),
            EmotionalTone.ANGER: (CompositionType.DRAMATIC, LightingMood.HARSH, "intense and confrontational"),
            EmotionalTone.MYSTERY: (CompositionType.ESTABLISHING, LightingMood.MYSTERIOUS, "atmospheric and enigmatic"),
        }

        comp_type, lighting, atmosphere = composition_map.get(
            primary_emotion,
            (CompositionType.MEDIUM_SHOT, LightingMood.NATURAL, "balanced emotional scene")
        )

        return SceneComposition(
            composition_type=comp_type,
            focal_point="central narrative element",
            background_elements=["atmospheric setting"],
            foreground_elements=["key story elements"],
            lighting_mood=lighting,
            atmosphere=atmosphere,
            color_palette_suggestion="emotionally appropriate palette",
            emotional_weight=emotional_moment.intensity_score
        )


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

    def _get_emotional_style_modifiers(self, style_config: Dict[str, Any], emotional_tones: List[str]) -> List[str]:
        """Extract emotional style modifiers from rich configuration."""
        modifiers = []

        # Check if we have emotional adaptations in the config
        if "emotional_adaptations" in style_config:
            adaptations = style_config["emotional_adaptations"]

            for tone in emotional_tones:
                tone_lower = tone.lower()
                if tone_lower in adaptations:
                    adaptation = adaptations[tone_lower]
                    modifiers.extend(adaptation.get("style_modifiers", []))

        return modifiers

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
            elif provider == ImageProvider.FLUX:
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
        style_modifiers = rich_config.get("base_prompt_modifiers", []).copy()

        # Add emotional adaptations if available
        if hasattr(scene_composition, 'emotional_tones'):
            emotional_modifiers = self._get_emotional_style_modifiers(
                rich_config,
                scene_composition.emotional_tones
            )
            style_modifiers.extend(emotional_modifiers)

        # Get provider-specific optimizations
        provider_opts = self._get_provider_optimizations(rich_config, provider)

        # Build translation result
        translation = {
            "style_modifiers": style_modifiers,
            "negative_prompt": rich_config.get("negative_prompt", []),
            "technical_params": rich_config.get("technical_params", {}),
            "provider_optimizations": provider_opts
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

        # Technical parameters optimized for DALL-E
        technical_params = style_config.get('technical_params', {})
        technical_params.update({
            "model": "dall-e-3",
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

    async def analyze_chapter_for_headers(self, chapter: Chapter) -> List[ChapterHeaderOption]:
        """Analyze a chapter to generate 4 header illustration options."""
        analysis_prompt = f"""
        Analyze the following chapter and create 4 distinct header illustration options.
        Each option should capture a different aspect of the chapter's essence.

        Chapter Title: {chapter.title}
        Chapter Content: {chapter.content[:2000]}...

        For each header option, consider:
        1. THEMATIC FOCUS - Key themes, symbols, or motifs
        2. VISUAL STYLE - Different artistic approaches (realistic, symbolic, atmospheric, dramatic)
        3. COMPOSITION - How the elements should be arranged
        4. EMOTIONAL TONE - The feeling the header should evoke

        Create exactly 4 options with these focuses:
        Option 1: Symbolic/Metaphorical representation
        Option 2: Character-focused dramatic scene
        Option 3: Environmental/atmospheric setting
        Option 4: Action/conflict moment

        Return as JSON with this structure:
        {{
            "options": [
                {{
                    "option_number": 1,
                    "title": "Symbolic Focus",
                    "description": "Brief description of the concept",
                    "visual_focus": "Main visual element",
                    "artistic_style": "Recommended style",
                    "composition_notes": "Layout and framing suggestions",
                    "key_elements": ["element1", "element2", "element3"],
                    "emotional_tone": "primary emotion",
                    "color_palette": "color suggestions"
                }},
                ...
            ]
        }}
        """

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="You are an expert visual artist and book designer specializing in chapter header illustrations."),
                HumanMessage(content=analysis_prompt)
            ])

            # Parse the JSON response
            response_text = response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:-3].strip()

            analysis_data = json.loads(response_text)
            header_options = []

            for option_data in analysis_data.get("options", []):
                # Create basic illustration prompt for each option
                base_prompt = f"{option_data['visual_focus']}, {option_data['artistic_style']}, {option_data['composition_notes']}"

                illustration_prompt = IllustrationPrompt(
                    provider=ImageProvider.DALLE,  # Default provider
                    prompt=base_prompt,
                    style_modifiers=[
                        option_data['artistic_style'],
                        f"chapter header illustration",
                        f"{option_data['color_palette']} color palette"
                    ],
                    negative_prompt="text, letters, words, typography, low quality, blurry, distorted",
                    technical_params={
                        "aspect_ratio": "16:9",
                        "style": "artistic",
                        "quality": "high"
                    }
                )

                header_option = ChapterHeaderOption(
                    option_number=option_data['option_number'],
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
            print(f"Error analyzing chapter for headers: {e}")
            # Return default header options as fallback
            return self._create_default_header_options(chapter)

    def _create_default_header_options(self, chapter: Chapter) -> List[ChapterHeaderOption]:
        """Create default header options when analysis fails."""
        default_options = []

        # Extract first few sentences for context
        first_sentences = chapter.content[:500]

        for i in range(4):
            styles = ["watercolor painting", "digital art", "pencil sketch", "oil painting"]
            focuses = ["symbolic", "character-driven", "environmental", "dramatic"]

            base_prompt = f"Chapter header illustration, {focuses[i]} style, based on: {chapter.title}"

            illustration_prompt = IllustrationPrompt(
                provider=ImageProvider.DALLE,  # Default provider
                prompt=base_prompt,
                style_modifiers=[styles[i], "chapter header", "artistic"],
                negative_prompt="text, words, low quality",
                technical_params={"aspect_ratio": "16:9", "quality": "high"}
            )

            option = ChapterHeaderOption(
                option_number=i + 1,
                title=f"{focuses[i].title()} Header",
                description=f"A {focuses[i]} representation of chapter themes",
                visual_focus=f"{focuses[i]} elements from the chapter",
                artistic_style=styles[i],
                composition_notes="Horizontal header layout",
                prompt=illustration_prompt
            )
            default_options.append(option)

        return default_options


class PromptEngineer:
    """Master prompt engineering system orchestrating all components."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.scene_analyzer = SceneAnalyzer(llm)
        self.style_translator = StyleTranslator()

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
        """Engineer an optimal prompt using all available techniques."""

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

        # Join all parts
        comprehensive_prompt = ". ".join(prompt_parts)

        # Final optimization pass
        optimized_prompt = await self._optimize_prompt_for_provider(
            comprehensive_prompt,
            provider,
            style_translation.get('provider_optimizations', {})
        )

        return optimized_prompt

    async def generate_chapter_header_options(
        self,
        chapter: Chapter,
        style_preferences: Dict[str, Any],
        provider: ImageProvider = ImageProvider.DALLE
    ) -> List[ChapterHeaderOption]:
        """Generate 4 chapter header illustration options."""

        # Create a StyleTranslator instance with LLM for header analysis
        style_translator_with_llm = StyleTranslator()
        style_translator_with_llm.llm = self.llm
        header_options = await style_translator_with_llm.analyze_chapter_for_headers(chapter)

        # Enhance each option's prompt using the full prompt engineering system
        enhanced_options = []
        for option in header_options:
            try:
                # Get style translation for this specific style
                header_style_config = {
                    **style_preferences,
                    'style_name': option.artistic_style,
                    'base_prompt_modifiers': [
                        option.artistic_style,
                        "chapter header illustration",
                        "horizontal composition"
                    ]
                }

                # Create a mock scene composition for header style
                header_composition = SceneComposition(
                    composition_type=CompositionType.ESTABLISHING,
                    focal_point=option.visual_focus,
                    background_elements=[],
                    foreground_elements=[option.visual_focus],
                    lighting_mood=LightingMood.NATURAL,
                    atmosphere="chapter header style",
                    color_palette_suggestion="harmonious",
                    emotional_weight=0.7
                )

                # Get enhanced style translation
                style_translation = self.style_translator.translate_style_config(
                    header_style_config,
                    provider,
                    header_composition
                )

                # Create enhanced prompt
                enhanced_prompt = IllustrationPrompt(
                    provider=provider,
                    prompt=f"{option.prompt.base_prompt}, {option.composition_notes}",
                    style_modifiers=style_translation['style_modifiers'],
                    negative_prompt=", ".join(style_translation['negative_prompt']) if style_translation['negative_prompt'] else option.prompt.negative_prompt,
                    technical_params={
                        **style_translation['technical_params'],
                        "aspect_ratio": "16:9",  # Header aspect ratio
                        "style": "artistic"
                    }
                )

                # Update the option with enhanced prompt
                enhanced_option = ChapterHeaderOption(
                    option_number=option.option_number,
                    title=option.title,
                    description=option.description,
                    visual_focus=option.visual_focus,
                    artistic_style=option.artistic_style,
                    composition_notes=option.composition_notes,
                    prompt=enhanced_prompt
                )

                enhanced_options.append(enhanced_option)

            except Exception as e:
                print(f"Error enhancing header option {option.option_number}: {e}")
                # Fall back to original option
                enhanced_options.append(option)

        return enhanced_options

    async def _enhance_scene_description(
        self,
        original_text: str,
        visual_elements: List[VisualElement],
        scene_composition: SceneComposition
    ) -> str:
        """Enhance scene description with comprehensive visual and artistic details."""
        enhancement_prompt = f"""
        Transform this literary text into a detailed visual scene description for classic book illustration generation.

        Original text: "{original_text}"
        Visual elements: {[f"{elem.element_type}: {elem.description}" for elem in visual_elements]}
        Scene focus: {scene_composition.focal_point}
        Composition: {scene_composition.composition_type.value}
        Atmosphere: {getattr(scene_composition, 'atmosphere', 'natural')}

        Create a comprehensive scene description that includes:

        1. SETTING & ENVIRONMENT:
        - Specific location details (interior/exterior, architectural elements, furniture, landscape)
        - Background elements and spatial relationships
        - Environmental atmosphere and mood

        2. CHARACTER DETAILS:
        - Precise positioning and postures of all characters
        - Facial expressions with specific emotional details (wide eyes, frozen smile, etc.)
        - Body language that conveys psychological state
        - How characters relate spatially to each other and the environment

        3. ARTISTIC COMPOSITION:
        - Camera angle/viewpoint (close-up, medium shot, wide shot)
        - What's in foreground vs background
        - Visual flow and focal points
        - How elements are arranged for maximum emotional impact

        4. EMOTIONAL ATMOSPHERE:
        - Specific mood descriptors (tension, unease, mystery, etc.)
        - How the environment reinforces the emotional state
        - Psychological elements visible through visual cues

        Example output: "Interior coffee shop scene with young man at ordering counter while female barista behind register displays shocked, fearful expression with widened eyes and artificial smile frozen on her face. The barista's body language suggests she's recoiling slightly from the customer. Coffee shop ambiance with espresso machine, displayed pastries, hanging menu boards. Captures moment of psychological tension - customer appears relaxed and normal, barista visibly disturbed and frightened."

        Return only the enhanced scene description with specific visual and emotional details.
        """

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="You are an expert visual artist who creates detailed scene descriptions for classic book illustrations, specializing in capturing both visual elements and emotional nuance."),
                HumanMessage(content=enhancement_prompt)
            ])
            enhanced_description = response.content.strip()

            # Ensure we have a substantive description
            if len(enhanced_description) < 50:
                # Use enhanced fallback if AI response is too brief
                return self._create_detailed_fallback_description(original_text, visual_elements, scene_composition)

            return enhanced_description

        except Exception as e:
            logger.warning(f"Scene description enhancement failed: {e}")
            return self._create_detailed_fallback_description(original_text, visual_elements, scene_composition)

    def _create_detailed_fallback_description(
        self,
        original_text: str,
        visual_elements: List[VisualElement],
        scene_composition: SceneComposition
    ) -> str:
        """Create a detailed fallback description when AI enhancement fails."""
        parts = []

        # Determine scene type and setting
        character_elements = [elem for elem in visual_elements if elem.element_type == "character"]
        environment_elements = [elem for elem in visual_elements if elem.element_type == "environment"]
        atmosphere_elements = [elem for elem in visual_elements if elem.element_type == "atmosphere"]

        # Build detailed description
        if environment_elements:
            setting = environment_elements[0].description
            parts.append(f"Scene set in {setting}")

        if character_elements:
            char_descriptions = []
            for char in character_elements:
                # Add emotional context from attributes if available
                emotion_details = char.attributes.get('emotional_significance', '') if hasattr(char, 'attributes') and char.attributes else ''
                if emotion_details:
                    char_descriptions.append(f"{char.description} ({emotion_details})")
                else:
                    char_descriptions.append(char.description)
            parts.append(f"featuring {', '.join(char_descriptions)}")

        # Add composition details
        parts.append(f"{scene_composition.composition_type.value} composition focusing on {scene_composition.focal_point}")

        # Add atmospheric details
        if atmosphere_elements:
            parts.append(f"with {atmosphere_elements[0].description}")
        elif hasattr(scene_composition, 'atmosphere') and scene_composition.atmosphere:
            parts.append(f"creating {scene_composition.atmosphere} atmosphere")

        # Add the original emotional context
        parts.append(f"Captures the moment: {original_text}")

        return ". ".join(parts)

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
        """Build character consistency guidance."""
        character_elements = [elem for elem in visual_elements if elem.element_type == "character"]

        if not character_elements:
            return ""

        guidance_parts = []
        for char_elem in character_elements:
            # Check if we have existing character profile
            char_name = char_elem.attributes.get('name', '').lower()
            if char_name in self.character_profiles:
                profile = self.character_profiles[char_name]
                if profile.physical_description:
                    guidance_parts.append(f"Character consistency: {profile.physical_description}")

        return ". ".join(guidance_parts) if guidance_parts else ""

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

        # Provider-specific enhancements with artistic detail
        if provider == ImageProvider.DALLE:
            enhanced_parts = []

            # Add artistic style prefix based on configuration
            if "pencil sketch" in prompt.lower() or "shepard" in prompt.lower():
                style_prefix = "A pencil sketch illustration in the style of E.H. Shepard showing"
                enhanced_parts.append(style_prefix)
                enhanced_parts.append(prompt)
                enhanced_parts.extend([
                    "Delicate line work, expressive faces, gentle shading, whimsical yet unsettling atmosphere",
                    "Classic book illustration style with fine crosshatching and soft pencil textures"
                ])
            else:
                enhanced_parts = [f"A masterfully crafted illustration: {prompt}"]
                enhanced_parts.append("Professional illustration quality")

        elif provider == ImageProvider.IMAGEN4:
            enhanced_parts = []

            if "pencil sketch" in prompt.lower() or "shepard" in prompt.lower():
                style_prefix = "E.H. Shepard style pencil drawing:"
                enhanced_parts.append(style_prefix)
                enhanced_parts.append(prompt)
                enhanced_parts.extend([
                    "Detailed linework showing environmental elements and character interactions",
                    "Soft graphite shading, crosshatching techniques, illustration book quality",
                    "Classic British illustration aesthetic with gentle, flowing lines"
                ])
            else:
                enhanced_parts = [f"Cinematic artistic scene: {prompt}"]
                enhanced_parts.append("Professional artistic rendering")

        elif provider == ImageProvider.FLUX:
            enhanced_parts = []

            if "pencil sketch" in prompt.lower() or "shepard" in prompt.lower():
                style_prefix = "Detailed pencil sketch in E.H. Shepard's illustration style:"
                enhanced_parts.append(style_prefix)
                enhanced_parts.append(prompt)
                enhanced_parts.extend([
                    "Fine line art with crosshatching, soft graphite shading, expressive character work",
                    "Classic children's book illustration aesthetic, delicate linework, emotive facial features"
                ])
            else:
                enhanced_parts = [f"Masterful artistic interpretation: {prompt}"]
                enhanced_parts.append("Superior artistic technique")

        else:
            enhanced_parts = [prompt]

        # Combine all parts
        enhanced_prompt = " ".join(enhanced_parts)

        # Ensure optimal length while preserving detail
        max_length = {
            ImageProvider.DALLE: 400,
            ImageProvider.IMAGEN4: 500,
            ImageProvider.FLUX: 600
        }.get(provider, 400)

        if len(enhanced_prompt) > max_length:
            # Truncate while preserving essential artistic elements
            essential_parts = enhanced_parts[0]  # Keep the style prefix and main description
            if len(essential_parts) <= max_length:
                enhanced_prompt = essential_parts
            else:
                enhanced_prompt = essential_parts[:max_length-3] + "..."

        return enhanced_prompt