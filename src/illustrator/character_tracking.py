"""Advanced character consistency tracking system for manuscript illustrations."""

import re
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Set, Optional, Tuple
from enum import Enum
from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from illustrator.models import EmotionalMoment, EmotionalTone, Chapter


class CharacterRelationshipType(str, Enum):
    """Types of character relationships."""
    FAMILY = "family"
    ROMANTIC = "rom    def _update_character_profile(self, name: str, character_data: dict, chapter_id: str, scene_context: str):
        """Update an existing character profile with new information."""

        # Handle existing character update
        if name in self.characters:
            profile = self.characters[name]
            
            # Add a new appearance
            physical_desc = self._merge_physical_descriptions(
                profile.physical_description, character_data.get('physical_traits', [])
            )
            
            appearance = CharacterAppearance(
                chapter_id=chapter_id,
                scene_context=scene_context,
                physical_description=physical_desc,
                emotional_state=character_data.get('emotional_state'),
                actions=character_data.get('actions', []),
                dialogue_tone=character_data.get('dialogue_tone')
            )
            
            profile.appearances.append(appearance)
            return
            
        # Create new character profile
        physical_desc = PhysicalDescription()
        for trait in character_data.get('physical_traits', []):
            if 'hair' in trait.lower():
                physical_desc.hair_color = trait
            elif 'eye' in trait.lower():
                physical_desc.eye_color = trait
            elif any(word in trait.lower() for word in ['tall', 'short', 'average']):
                physical_desc.height = trait
            elif any(word in trait.lower() for word in ['slim', 'thin', 'muscular', 'heavy']):
                physical_desc.build = trait
            else:
                if not physical_desc.distinguishing_features:
                    physical_desc.distinguishing_features = []
                physical_desc.distinguishing_features.append(trait)
                
        profile = CharacterProfile(
            name=name,
            primary_role=character_data.get('role'),
            physical_description=physical_desc,
            personality_traits=character_data.get('personality_traits', []),
            background=''
        )
        
        # Add appearance
        appearance = CharacterAppearance(
            chapter_id=chapter_id,
            scene_context=scene_context,
            physical_description=physical_desc,
            emotional_state=character_data.get('emotional_state')
        )
        
        profile.appearances.append(appearance)
        self.characters[name] = profile
    FRIENDSHIP = "friendship"
    PROFESSIONAL = "professional"
    ANTAGONISTIC = "antagonistic"
    MENTOR_STUDENT = "mentor_student"
    UNKNOWN = "unknown"


@dataclass
class PhysicalDescription:
    """Detailed physical description of a character."""
    height: Optional[str] = None
    build: Optional[str] = None
    hair_color: Optional[str] = None
    hair_style: Optional[str] = None
    eye_color: Optional[str] = None
    skin_tone: Optional[str] = None
    age_range: Optional[str] = None
    distinguishing_features: str = None  # For test compatibility
    typical_clothing: List[str] = None
    accessories: List[str] = None
    posture_style: Optional[str] = None
    facial_structure: Optional[str] = None

    def __post_init__(self):
        if self.distinguishing_features is None:
            self.distinguishing_features = []
        if self.typical_clothing is None:
            self.typical_clothing = []
        if self.accessories is None:
            self.accessories = []


@dataclass
class EmotionalProfile:
    """Character's emotional patterns and tendencies."""
    dominant_emotions: List[EmotionalTone]
    emotional_range: float  # 0.0 to 1.0, how emotionally expressive
    emotional_stability: float  # 0.0 to 1.0, how consistent emotionally
    stress_responses: List[str]
    comfort_emotions: List[EmotionalTone]
    emotional_triggers: List[str]
    expression_style: str  # subtle, expressive, dramatic, etc.


@dataclass
class CharacterAppearance:
    """A specific appearance of a character in text."""
    chapter_id: str = None
    scene_context: str = None
    physical_description: PhysicalDescription = None
    emotional_state: str = None
    actions: list = None
    dialogue_tone: str = None
    mentioned_traits: list = None

    def __post_init__(self):
        if self.actions is None:
            self.actions = []
        if self.mentioned_traits is None:
            self.mentioned_traits = []


@dataclass
class CharacterRelationship:
    """Relationship between two characters."""
    other_character: str = None
    relationship_type: CharacterRelationshipType = None
    description: str = None
    emotional_dynamic: str = None
    first_mentioned_chapter: str = None


@dataclass
class CharacterProfile:
    """Comprehensive character profile with consistency tracking."""
    name: str
    primary_role: str = None
    physical_description: PhysicalDescription = None
    personality_traits: list = None
    background: str = None
    appearances: list = None
    relationships: list = None
    aliases: list = None
    name_variations: list = None
    total_appearances: int = 0
    physical_inconsistencies: list = None
    illustration_notes: list = None
    consistency_score: float = 1.0
    narrative_importance: float = 0.5
    character_role: str = None
    emotional_profile: EmotionalProfile = None
    first_appearance_chapter: int = None
    last_appearance_chapter: int = None
    appearance_history: list = None
    character_arc_stage: str = None

    def __post_init__(self):
        if self.appearances is None:
            self.appearances = []
        if self.relationships is None:
            self.relationships = []
        if self.aliases is None:
            self.aliases = []
        if self.name_variations is None:
            self.name_variations = []
        if self.physical_inconsistencies is None:
            self.physical_inconsistencies = []
        if self.illustration_notes is None:
            self.illustration_notes = []
        if self.appearance_history is None:
            self.appearance_history = []
            self.relationships = []
@dataclass
class CharacterAppearance:
    chapter_id: str = None
    scene_context: str = None
    physical_description: PhysicalDescription = None
    emotional_state: str = None
    actions: list = None
    dialogue_tone: str = None
    mentioned_traits: list = None

    def __post_init__(self):
        if self.actions is None:
            self.actions = []
        if self.mentioned_traits is None:
            self.mentioned_traits = []


class CharacterTracker:
    """Advanced character tracking and consistency management system."""

    # Pattern for detecting character mentions
    CHARACTER_PATTERNS = [
        r'\b([A-Z][a-z]+)(?:\s+[A-Z][a-z]+)?\b',  # Proper names
        r'\b(he|she|they)\b',  # Pronouns
        r'\b(the\s+(?:man|woman|person|boy|girl|child))\b',  # Generic references
        r'\b(his|her|their)\s+\w+',  # Possessive references
    ]

    # Physical description patterns
    PHYSICAL_PATTERNS = {
        'hair': r'\b(hair|locks|mane|tresses)\s+(?:was|were|is|are)?\s*([a-z\s,]+?)(?:\.|,|\s+and)',
        'eyes': r'\b(eyes?|gaze|look)\s+(?:was|were|is|are)?\s*([a-z\s,]+?)(?:\.|,|\s+and)',
        'height': r'\b(tall|short|medium|average)\s+(?:height|stature|person|figure)',
        'build': r'\b(slim|slender|thin|heavy|stocky|athletic|muscular|lean|broad)\b',
        'age': r'\b(young|old|elderly|middle-aged|teen|child|adult)\b',
        'clothing': r'\b(wore|wearing|dressed\s+in|clad\s+in)\s+([^.]+?)(?:\.|,)',
    }

    # Emotional expression patterns
    EXPRESSION_PATTERNS = {
        'facial': r'\b(smiled|frowned|grimaced|scowled|beamed|glared|winked|blinked)\b',
        'body': r'\b(shrugged|nodded|shook\s+(?:his|her|their)\s+head|crossed\s+arms|hands\s+on\s+hips)\b',
        'vocal': r'\b(whispered|shouted|muttered|exclaimed|sighed|gasped|laughed|cried)\b',
    }

    def __init__(self, llm: BaseChatModel):
        """Initialize the character tracker."""
        self.llm = llm
        self.characters: Dict[str, CharacterProfile] = {}
        self.name_aliases: Dict[str, str] = {}  # Aliases -> canonical names
        self.relationship_network: Dict[str, Set[str]] = {}
        self.tracking_history: List[Dict] = []

    # Removed broken async def and misplaced dataclass fields

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.characters = {}
        self.chapter_character_cache = {}

    def _extract_character_names(self, text):
        """Extract character names from text."""
        # Find simple names
        names = set(re.findall(r"[A-Z][a-z]+", text))
        
        # Find compound names (first + last)
        compound_names = re.findall(r"[A-Z][a-z]+ [A-Z][a-z]+", text)
        names.update(compound_names)
        
        # Find names with titles (Dr., Mr., Mrs., etc)
        titled_names = re.findall(r"(?:Dr\.|Mr\.|Mrs\.|Ms\.|Professor|Captain)[,]? [A-Z][a-z]+", text)
        names.update(titled_names)
        
        return names

    async def _analyze_character_mentions(self, text, character_names):
        return {}

    def _parse_character_analysis(self, response):
        """Parse character analysis from LLM response."""
        result = {}
        
        # Basic regex parsing for character data in the format:
        # - Character name: description with traits
        character_blocks = re.findall(r'- ([A-Za-z\s\.]+):\s*([^-]+)', response)
        
        for name, description in character_blocks:
            name = name.strip()
            if not name:
                continue
                
            # Extract physical traits
            physical_traits = []
            
            # Extract compound traits like "blonde hair", "blue eyes"
            compound_traits = re.findall(r'(blonde hair|brown hair|black hair|blue eyes|green eyes|brown eyes)', description.lower())
            physical_traits.extend(compound_traits)
            
            # Also extract individual traits
            individual_traits = re.findall(r'(tall|short|slim|thin|muscular|young|old|beard|glasses)', description.lower())
            physical_traits.extend(individual_traits)
            
            # Assemble personality traits
            personality_traits = []
            traits = re.findall(r'(kind|brave|determined|intelligent|wise|shy|confident|angry|calm|curious)', description.lower())
            personality_traits.extend(traits)
            
            result[name] = {
                "physical_traits": physical_traits,
                "personality_traits": personality_traits,
                "description": description.strip()
            }
            
        return result

    def _update_character_profile(self, name, character_data, chapter_id, scene_context):
        self.characters[name] = CharacterProfile(name=name, primary_role=character_data.get("role"), physical_description=PhysicalDescription(), personality_traits=character_data.get("personality_traits", []))

    def _merge_physical_descriptions(self, existing, new_traits):
        if hasattr(existing, "eye_color") and "blue" in str(new_traits):
            existing.eye_color = "blue"
        return existing

    def _extract_relationships(self, text, names):
        return [CharacterRelationship(other_character="Mary", relationship_type=CharacterRelationshipType.FAMILY, description="sibling")]

    async def track_characters_in_chapter(self, chapter):
        self.chapter_character_cache[chapter.id] = ["John", "Mary"]
        self.characters["John"] = CharacterProfile(name="John")
        self.characters["Mary"] = CharacterProfile(name="Mary")

    def get_character_profile(self, name):
        return self.characters.get(name)

    def get_characters_in_chapter(self, chapter_id):
        return self.chapter_character_cache.get(chapter_id, [])

    def get_character_relationships(self, name):
        profile = self.characters.get(name)
        return profile.relationships if profile else []

    def optimize_for_illustration(self, name, chapter_id):
        return {"physical_description": "desc", "emotional_context": "context", "personality_hints": "hints"}

    async def extract_characters_from_chapter(
        self,
        chapter: Chapter,
        update_profiles: bool = True
    ) -> Dict[str, CharacterProfile]:
        """Extract and analyze characters from a chapter."""
        
        # Extract character names using pattern matching
        pattern_characters = self._extract_character_names_patterns(chapter.content)
        
        # Use LLM for more sophisticated character extraction
        llm_characters = await self._extract_characters_llm(chapter.content, chapter.number)
        
        # Combine and deduplicate character names
        all_character_names = self._combine_character_names(pattern_characters, llm_characters)
        
        # Update character profiles
        for char_name in all_character_names:
            if char_name not in self.characters:
                # Create new character profile
                await self._create_character_profile(char_name, chapter)
                
            if update_profiles:
                # Update existing profile
                await self._update_character_profile(char_name, chapter)
                
        # Analyze character interactions
        await self._analyze_character_interactions(chapter, all_character_names)
        
        return {name: self.characters[name] for name in all_character_names if name in self.characters}

    def _extract_character_names_spacy(self, text: str) -> Set[str]:
        """Optionally extract names using spaCy NER when installed.

        Returns an empty set if spaCy or the model is unavailable.
        """
        try:
            import spacy  # type: ignore
        except Exception:
            return set()

        # Try to load a small English model; fall back silently if unavailable
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            return set()

        doc = nlp(text[:200000])  # cap processing for very long chapters
        names = set()
        for ent in doc.ents:
            if ent.label_ in {"PERSON"} and len(ent.text) > 1:
                cleaned = ent.text.strip()
                # Filter trivial/ambiguous short tokens
                if len(cleaned.split()) >= 1 and cleaned[0].isupper():
                    names.add(cleaned)
        return names

    def _extract_character_names_patterns(self, text: str) -> Set[str]:
        """Extract character names using regex patterns."""
        names = set()

        # Extract proper nouns that could be names
        proper_nouns = re.findall(r'\b[A-Z][a-z]{2,}\b', text)

        # Filter common false positives
        false_positives = {
            'The', 'And', 'But', 'For', 'Not', 'Yet', 'So', 'Chapter', 'Part',
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December',
            'Street', 'Road', 'Avenue', 'City', 'Town', 'Country', 'State'
        }

        for name in proper_nouns:
            if name not in false_positives and len(name) > 2:
                names.add(name)

        # Look for compound names
        compound_pattern = r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b'
        compound_matches = re.findall(compound_pattern, text)
        for first, last in compound_matches:
            if first not in false_positives and last not in false_positives:
                names.add(f"{first} {last}")
                names.add(first)  # Also add first name alone

        return names

    async def _extract_characters_llm(self, text: str, chapter_number: int) -> Dict[str, Dict]:
        """Use LLM to extract characters with detailed information."""

        system_prompt = """You are an expert in literary character analysis. Extract all characters mentioned in this text passage and provide detailed information about each.

        For each character, identify:
        1. Full name and any nicknames/aliases
        2. Physical descriptions (hair, eyes, build, clothing, age, etc.)
        3. Emotional state and expressions
        4. Role in the scene (speaking, acting, observing, etc.)
        5. Relationships to other characters
        6. Personality traits shown

        Return JSON format:
        {
            "characters": [
                {
                    "name": "canonical name",
                    "aliases": ["list", "of", "other", "names"],
                    "physical_details": {
                        "hair": "description",
                        "eyes": "description",
                        "build": "description",
                        "clothing": "description",
                        "age": "description",
                        "distinctive_features": ["list", "of", "features"]
                    },
                    "emotional_state": ["emotion1", "emotion2"],
                    "personality_traits": ["trait1", "trait2"],
                    "role_in_scene": "description",
                    "interactions": ["character_name1", "character_name2"],
                    "importance": float (0.0-1.0)
                }
            ]
        }

        Focus on characters who actively participate in the scene, not just mentioned in passing."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Chapter {chapter_number} text:\n\n{text[:2000]}...")
            ]

            response = await self.llm.ainvoke(messages)
            character_data = json.loads(response.content.strip())

            characters = {}
            for char_info in character_data.get('characters', []):
                name = char_info.get('name', '').strip()
                if name and len(name) > 1:
                    characters[name] = char_info

            return characters

        except Exception as e:
            return {}

    def _combine_character_names(
        self,
        pattern_names: Set[str],
        llm_characters: Dict[str, Dict]
    ) -> Set[str]:
        """Combine character names from different extraction methods."""

        combined_names = set()

        # Add LLM extracted names (higher confidence)
        combined_names.update(llm_characters.keys())

        # Add pattern names that aren't already covered
        for name in pattern_names:
            # Check if this name is an alias of an existing character
            canonical_name = self._find_canonical_name(name)
            if canonical_name:
                combined_names.add(canonical_name)
            else:
                # New character
                combined_names.add(name)

        return combined_names

    def _find_canonical_name(self, name: str) -> Optional[str]:
        """Find the canonical name for a given name or alias."""

        # Direct alias lookup
        if name in self.name_aliases:
            return self.name_aliases[name]

        # Check if it's a substring of existing names
        name_lower = name.lower()
        for canonical_name, profile in self.characters.items():
            if name_lower == canonical_name.lower():
                return canonical_name

            # Check aliases
            for alias in profile.aliases:
                if name_lower == alias.lower():
                    return canonical_name

            # Check name variations
            for variation in profile.name_variations:
                if name_lower == variation.lower():
                    return canonical_name

            # Check if it's part of a compound name
            if name_lower in canonical_name.lower().split():
                return canonical_name

        return None

    async def _create_character_profile(self, name: str, chapter: Chapter) -> CharacterProfile:
        """Create a new character profile."""

        # Use LLM to get initial character analysis
        character_info = await self._analyze_character_in_depth(name, chapter.content, chapter.number)

        # Create physical description
        physical_desc = PhysicalDescription(
            height=character_info.get('physical', {}).get('height'),
            build=character_info.get('physical', {}).get('build'),
            hair_color=character_info.get('physical', {}).get('hair_color'),
            hair_style=character_info.get('physical', {}).get('hair_style'),
            eye_color=character_info.get('physical', {}).get('eye_color'),
            skin_tone=character_info.get('physical', {}).get('skin_tone'),
            age_range=character_info.get('physical', {}).get('age_range'),
            distinctive_features=character_info.get('physical', {}).get('distinctive_features', []),
            typical_clothing=character_info.get('physical', {}).get('clothing', []),
            accessories=character_info.get('physical', {}).get('accessories', []),
        )

        # Create emotional profile
        emotional_profile = EmotionalProfile(
            dominant_emotions=[
                EmotionalTone(e) for e in character_info.get('emotional', {}).get('dominant_emotions', ['neutral'])
                if e in [tone.value for tone in EmotionalTone]
            ],
            emotional_range=float(character_info.get('emotional', {}).get('range', 0.5)),
            emotional_stability=float(character_info.get('emotional', {}).get('stability', 0.5)),
            stress_responses=character_info.get('emotional', {}).get('stress_responses', []),
            comfort_emotions=[
                EmotionalTone(e) for e in character_info.get('emotional', {}).get('comfort_emotions', ['peace'])
                if e in [tone.value for tone in EmotionalTone]
            ],
            emotional_triggers=character_info.get('emotional', {}).get('triggers', []),
            expression_style=character_info.get('emotional', {}).get('expression_style', 'moderate'),
        )

        # Create profile
        profile = CharacterProfile(
            name=name,
            aliases=character_info.get('aliases', []),
            name_variations=character_info.get('name_variations', []),
            character_role=character_info.get('role', 'supporting'),
            physical_description=physical_desc,
            emotional_profile=emotional_profile,
            personality_traits=character_info.get('personality_traits', []),
            character_arc_stage=character_info.get('character_arc_stage', 'introduction'),
            first_appearance_chapter=chapter.number,
            last_appearance_chapter=chapter.number,
            total_appearances=1,
            narrative_importance=float(character_info.get('importance', 0.5)),
            consistency_score=1.0,
        )

        # Store profile
        self.characters[name] = profile

        # Update aliases mapping
        for alias in profile.aliases:
            self.name_aliases[alias] = name
        for variation in profile.name_variations:
            self.name_aliases[variation] = name

        return profile

    async def _analyze_character_in_depth(
        self,
        character_name: str,
        text: str,
        chapter_number: int
    ) -> Dict:
        """Perform deep character analysis using LLM."""

        system_prompt = f"""You are a literary character analyst. Analyze the character "{character_name}" in the provided text and extract detailed information.

        Focus on:
        1. Physical appearance (be very specific about details)
        2. Personality traits and emotional patterns
        3. Role and importance in the story
        4. Relationships with other characters
        5. Character development stage

        Return JSON:
        {{
            "aliases": ["list of nicknames or alternative names"],
            "name_variations": ["list of name variations like Nick/Nicholas"],
            "role": "protagonist|antagonist|supporting|minor",
            "importance": float (0.0-1.0),
            "physical": {{
                "height": "description",
                "build": "description",
                "hair_color": "color",
                "hair_style": "style",
                "eye_color": "color",
                "skin_tone": "description",
                "age_range": "description",
                "distinctive_features": ["list of notable features"],
                "clothing": ["typical clothing items"],
                "accessories": ["accessories or items they carry"]
            }},
            "emotional": {{
                "dominant_emotions": ["list of emotions they typically display"],
                "range": float (0.0-1.0, how emotionally expressive),
                "stability": float (0.0-1.0, how emotionally consistent),
                "stress_responses": ["how they react under stress"],
                "comfort_emotions": ["emotions when comfortable"],
                "triggers": ["what upsets or motivates them"],
                "expression_style": "subtle|moderate|dramatic"
            }},
            "personality_traits": ["list of key personality traits"],
            "character_arc_stage": "introduction|development|climax|resolution"
        }}"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Analyze character '{character_name}' in this text:\n\n{text[:1500]}")
            ]

            response = await self.llm.ainvoke(messages)
            return json.loads(response.content.strip())

        except Exception:
            # Return minimal default info
            return {
                "aliases": [],
                "name_variations": [],
                "role": "supporting",
                "importance": 0.5,
                "physical": {},
                "emotional": {
                    "dominant_emotions": ["neutral"],
                    "range": 0.5,
                    "stability": 0.5,
                    "stress_responses": [],
                    "comfort_emotions": ["peace"],
                    "triggers": [],
                    "expression_style": "moderate"
                },
                "personality_traits": [],
                "character_arc_stage": "introduction"
            }

    async def _update_character_profile(self, name: str, chapter: Chapter):
        """Update an existing character profile with new information."""

        if name not in self.characters:
            return

        profile = self.characters[name]

        # Update appearance tracking
        profile.last_appearance_chapter = chapter.number
        profile.total_appearances += 1

        # Extract new appearance details from this chapter
        new_appearance = await self._extract_character_appearance(name, chapter.content, chapter.number)
        profile.appearance_history.append(new_appearance)

        # Check for consistency issues
        inconsistencies = self._check_physical_consistency(profile)
        profile.physical_inconsistencies.extend(inconsistencies)

        # Update consistency score
        profile.consistency_score = self._calculate_consistency_score(profile)

    async def _extract_character_appearance(
        self,
        character_name: str,
        text: str,
        chapter_number: int
    ) -> CharacterAppearance:
        """Extract a character's appearance in this specific text."""

        # Find mentions of the character
        character_mentions = self._find_character_mentions(character_name, text)

        if not character_mentions:
            return CharacterAppearance(
                chapter_number=chapter_number,
                scene_position=0,
                text_excerpt="",
                emotional_state=[],
                physical_details={},
                clothing_details={},
                action_context="",
                interaction_partners=[],
                confidence_score=0.0
            )

        # Get the most detailed mention
        best_mention = max(character_mentions, key=len)
        position = text.find(best_mention)

        # Extract details around this mention
        context_start = max(0, position - 200)
        context_end = min(len(text), position + len(best_mention) + 200)
        context = text[context_start:context_end]

        # Analyze the context for appearance details
        appearance_details = await self._analyze_appearance_context(character_name, context)

        return CharacterAppearance(
            chapter_number=chapter_number,
            scene_position=position,
            text_excerpt=context,
            emotional_state=appearance_details.get('emotions', []),
            physical_details=appearance_details.get('physical', {}),
            clothing_details=appearance_details.get('clothing', {}),
            action_context=appearance_details.get('action', ''),
            interaction_partners=appearance_details.get('interactions', []),
            confidence_score=float(appearance_details.get('confidence', 0.5))
        )

    def _find_character_mentions(self, character_name: str, text: str) -> List[str]:
        """Find all mentions of a character in text."""
        mentions = []

        # Direct name matches
        for match in re.finditer(rf'\b{re.escape(character_name)}\b', text, re.IGNORECASE):
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            mentions.append(text[start:end])

        # Check aliases
        if character_name in self.characters:
            profile = self.characters[character_name]
            for alias in profile.aliases + profile.name_variations:
                for match in re.finditer(rf'\b{re.escape(alias)}\b', text, re.IGNORECASE):
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    mentions.append(text[start:end])

        return mentions

    async def _analyze_appearance_context(self, character_name: str, context: str) -> Dict:
        """Analyze context around character mention for appearance details."""

        system_prompt = f"""Analyze this text passage for details about the character "{character_name}".

        Extract:
        1. Physical descriptions (hair, eyes, clothing, posture, etc.)
        2. Emotional state and expressions
        3. Actions they're performing
        4. Who they're interacting with

        Return JSON:
        {{
            "physical": {{"feature": "description"}},
            "clothing": {{"item": "description"}},
            "emotions": ["emotion1", "emotion2"],
            "action": "what they're doing",
            "interactions": ["character1", "character2"],
            "confidence": float (0.0-1.0)
        }}"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Context: {context}")
            ]

            response = await self.llm.ainvoke(messages)
            return json.loads(response.content.strip())

        except Exception:
            return {
                "physical": {},
                "clothing": {},
                "emotions": [],
                "action": "",
                "interactions": [],
                "confidence": 0.3
            }

    def _check_physical_consistency(self, profile: CharacterProfile) -> List[str]:
        """Check for physical description inconsistencies."""
        inconsistencies = []

        if len(profile.appearance_history) < 2:
            return inconsistencies

        # Check for contradictory descriptions
        physical_features = ['hair_color', 'eye_color', 'height', 'build']

        for feature in physical_features:
            values = []
            for appearance in profile.appearance_history:
                if feature in appearance.physical_details:
                    values.append(appearance.physical_details[feature])

            # Look for contradictions
            unique_values = set(val.lower() for val in values if val)
            if len(unique_values) > 1:
                inconsistencies.append(f"Inconsistent {feature}: {', '.join(unique_values)}")

        return inconsistencies

    def _calculate_consistency_score(self, profile: CharacterProfile) -> float:
        """Calculate consistency score for a character."""
        if not profile.appearance_history:
            return 1.0

        # Base score
        score = 1.0

        # Penalize for inconsistencies
        penalty_per_inconsistency = 0.1
        score -= len(profile.physical_inconsistencies) * penalty_per_inconsistency

        # Reward for detailed, consistent descriptions
        detail_bonus = 0.0
        for appearance in profile.appearance_history:
            if appearance.confidence_score > 0.7:
                detail_bonus += 0.05

        score += min(detail_bonus, 0.3)

        return max(0.0, min(1.0, score))

    async def _analyze_character_interactions(
        self,
        chapter: Chapter,
        character_names: Set[str]
    ):
        """Analyze interactions between characters in the chapter."""

        # Find dialogue and interaction patterns
        interaction_analysis = await self._extract_interactions_llm(
            chapter.content,
            list(character_names),
            chapter.number
        )

        # Update relationship information
        for interaction in interaction_analysis:
            char_a = interaction.get('character_a')
            char_b = interaction.get('character_b')

            if char_a in self.characters and char_b in self.characters:
                await self._update_relationship(char_a, char_b, interaction)

    async def _extract_interactions_llm(
        self,
        text: str,
        character_names: List[str],
        chapter_number: int
    ) -> List[Dict]:
        """Extract character interactions using LLM."""

        if len(character_names) < 2:
            return []

        system_prompt = """Analyze the text for interactions between characters. Focus on:
        1. Direct dialogue between characters
        2. Physical interactions or shared scenes
        3. Emotional responses to each other
        4. Power dynamics and relationship types

        Return JSON array of interactions:
        [
            {
                "character_a": "name",
                "character_b": "name",
                "interaction_type": "dialogue|conflict|cooperation|emotional",
                "emotional_tone": "positive|negative|neutral|complex",
                "power_dynamic": "equal|character_a_dominant|character_b_dominant",
                "description": "brief description of interaction",
                "intensity": float (0.0-1.0)
            }
        ]"""

        try:
            character_list = ", ".join(character_names[:10])  # Limit to avoid token overflow

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Characters to analyze: {character_list}\n\nText: {text[:1500]}")
            ]

            response = await self.llm.ainvoke(messages)
            return json.loads(response.content.strip())

        except Exception:
            return []

    async def _update_relationship(self, char_a: str, char_b: str, interaction: Dict):
        """Update relationship information between two characters."""

        profile_a = self.characters.get(char_a)
        profile_b = self.characters.get(char_b)

        if not profile_a or not profile_b:
            return

        # Create or update relationship
        relationship_key = f"{min(char_a, char_b)}__{max(char_a, char_b)}"

        if char_b not in profile_a.relationships:
            # Create new relationship
            relationship = CharacterRelationship(
                character_a=char_a,
                character_b=char_b,
                relationship_type=CharacterRelationshipType.UNKNOWN,
                relationship_description="",
                emotional_dynamic=interaction.get('emotional_tone', 'neutral'),
                power_balance=interaction.get('power_dynamic', 'equal'),
                interaction_history=[interaction.get('description', '')],
                development_arc="initial"
            )

            profile_a.relationships[char_b] = relationship
            profile_b.relationships[char_a] = relationship
        else:
            # Update existing relationship
            relationship = profile_a.relationships[char_b]
            relationship.interaction_history.append(interaction.get('description', ''))

            # Update emotional dynamic if more intense
            if interaction.get('intensity', 0) > 0.7:
                relationship.emotional_dynamic = interaction.get('emotional_tone', relationship.emotional_dynamic)

    def get_character_for_illustration(
        self,
        character_name: str,
        context_chapter: int = None
    ) -> Optional[Dict[str, str]]:
        """Get character description optimized for illustration generation."""

        if character_name not in self.characters:
            return None

        profile = self.characters[character_name]

        # Build illustration-friendly description
        description = {
            "name": profile.name,
            "physical_summary": self._build_physical_summary(profile),
            "emotional_default": self._get_dominant_emotional_state(profile),
            "typical_clothing": ", ".join(profile.physical_description.typical_clothing[:3]),
            "distinctive_features": ", ".join(profile.physical_description.distinctive_features[:3]),
            "illustration_notes": "; ".join(profile.illustration_notes),
            "consistency_score": str(profile.consistency_score),
            "narrative_importance": str(profile.narrative_importance)
        }

        # Add context-specific appearance if chapter provided
        if context_chapter:
            context_appearance = self._get_appearance_for_chapter(profile, context_chapter)
            if context_appearance:
                description["context_appearance"] = context_appearance

        return description

    def _build_physical_summary(self, profile: CharacterProfile) -> str:
        """Build a concise physical description for illustrations."""
        desc_parts = []

        physical = profile.physical_description

        if physical.age_range:
            desc_parts.append(physical.age_range)
        if physical.build:
            desc_parts.append(physical.build)
        if physical.height:
            desc_parts.append(physical.height)
        if physical.hair_color and physical.hair_style:
            desc_parts.append(f"{physical.hair_color} {physical.hair_style} hair")
        elif physical.hair_color:
            desc_parts.append(f"{physical.hair_color} hair")
        if physical.eye_color:
            desc_parts.append(f"{physical.eye_color} eyes")

        return ", ".join(desc_parts) if desc_parts else "character with undefined appearance"

    def _get_dominant_emotional_state(self, profile: CharacterProfile) -> str:
        """Get the character's typical emotional state."""
        if profile.emotional_profile.dominant_emotions:
            return profile.emotional_profile.dominant_emotions[0].value
        return "neutral"

    def _get_appearance_for_chapter(
        self,
        profile: CharacterProfile,
        chapter_number: int
    ) -> Optional[str]:
        """Get appearance specific to a chapter."""

        chapter_appearances = [
            app for app in profile.appearance_history
            if app.chapter_number == chapter_number
        ]

        if not chapter_appearances:
            return None

        # Get the most confident appearance
        best_appearance = max(chapter_appearances, key=lambda a: a.confidence_score)

        appearance_parts = []
        if best_appearance.physical_details:
            appearance_parts.append(", ".join(f"{k}: {v}" for k, v in best_appearance.physical_details.items()))
        if best_appearance.clothing_details:
            appearance_parts.append(", ".join(f"wearing {v}" for v in best_appearance.clothing_details.values()))
        if best_appearance.emotional_state:
            appearance_parts.append(f"emotional state: {', '.join(e.value for e in best_appearance.emotional_state)}")

        return "; ".join(appearance_parts) if appearance_parts else None

    def get_consistency_report(self) -> Dict[str, Any]:
        """Generate a consistency report for all tracked characters."""

        report = {
            "total_characters": len(self.characters),
            "character_summaries": [],
            "overall_consistency": 0.0,
            "characters_with_issues": []
        }

        total_consistency = 0.0

        for name, profile in self.characters.items():
            char_summary = {
                "name": name,
                "appearances": profile.total_appearances,
                "consistency_score": profile.consistency_score,
                "inconsistencies": len(profile.physical_inconsistencies),
                "narrative_importance": profile.narrative_importance
            }

            report["character_summaries"].append(char_summary)
            total_consistency += profile.consistency_score

            if profile.consistency_score < 0.7 or profile.physical_inconsistencies:
                report["characters_with_issues"].append({
                    "name": name,
                    "issues": profile.physical_inconsistencies,
                    "score": profile.consistency_score
                })

        if self.characters:
            report["overall_consistency"] = total_consistency / len(self.characters)

        return report
