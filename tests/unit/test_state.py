"""Comprehensive unit tests for the state module."""

import pytest
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from illustrator.state import ManuscriptState
from illustrator.models import (
    Chapter,
    ChapterAnalysis,
    EmotionalMoment,
    EmotionalTone,
    ImageProvider,
    IllustrationPrompt,
    ManuscriptMetadata
)


class TestManuscriptState:
    """Test the ManuscriptState TypedDict."""

    @pytest.fixture
    def sample_metadata(self):
        """Sample manuscript metadata for testing."""
        return ManuscriptMetadata(
            title="Test Novel",
            author="Test Author",
            genre="Fantasy",
            total_chapters=2,
            created_at=datetime.now().isoformat()
        )

    @pytest.fixture
    def sample_chapter(self):
        """Sample chapter for testing."""
        return Chapter(
            title="Chapter 1",
            content="This is test chapter content.",
            number=1,
            word_count=5
        )

    @pytest.fixture
    def sample_analysis(self, sample_chapter):
        """Sample chapter analysis for testing."""
        emotional_moment = EmotionalMoment(
            text_excerpt="emotional content",
            start_position=10,
            end_position=27,
            emotional_tones=[EmotionalTone.JOY],
            intensity_score=0.8,
            context="This is emotional content"
        )
        from illustrator.models import IllustrationPrompt
        illustration_prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="A mystical forest scene",
            style_modifiers=["mystical", "atmospheric"],
            technical_params={}
        )
        return ChapterAnalysis(
            chapter=sample_chapter,
            emotional_moments=[emotional_moment],
            dominant_themes=["Growth", "Discovery"],
            setting_description="A mystical forest",
            character_emotions={"protagonist": [EmotionalTone.JOY]},
            illustration_prompts=[illustration_prompt]
        )

    def test_manuscript_state_structure(self):
        """Test that ManuscriptState has the expected structure."""
        # Test that ManuscriptState is a TypedDict
        assert hasattr(ManuscriptState, '__annotations__')

        # Check expected fields
        expected_fields = {
            'messages',
            'manuscript_metadata',
            'current_chapter',
            'chapters_completed',
            'awaiting_chapter_input',
            'processing_complete',
            'illustrations_generated',
            'image_provider',
            'style_preferences',
            'analysis_depth',
            'current_analysis',
            'generated_images',
            'error_message',
            'retry_count'
        }

        actual_fields = set(ManuscriptState.__annotations__.keys())
        assert actual_fields == expected_fields

    def test_manuscript_state_field_types(self):
        """Test that ManuscriptState fields have correct type annotations."""
        annotations = ManuscriptState.__annotations__

        # Test specific field types
        assert 'messages' in annotations
        assert 'manuscript_metadata' in annotations
        assert 'current_chapter' in annotations
        assert 'chapters_completed' in annotations
        assert 'awaiting_chapter_input' in annotations
        assert 'processing_complete' in annotations
        assert 'illustrations_generated' in annotations
        assert 'image_provider' in annotations
        assert 'style_preferences' in annotations
        assert 'analysis_depth' in annotations
        assert 'current_analysis' in annotations
        assert 'generated_images' in annotations
        assert 'error_message' in annotations
        assert 'retry_count' in annotations

    def test_manuscript_state_creation_empty(self):
        """Test creating an empty ManuscriptState."""
        state: ManuscriptState = {
            'messages': [],
            'manuscript_metadata': None,
            'current_chapter': None,
            'chapters_completed': [],
            'awaiting_chapter_input': False,
            'processing_complete': False,
            'illustrations_generated': False,
            'image_provider': ImageProvider.DALLE,
            'style_preferences': {},
            'analysis_depth': 'basic',
            'current_analysis': None,
            'generated_images': [],
            'error_message': None,
            'retry_count': 0
        }

        assert isinstance(state, dict)
        assert len(state['messages']) == 0
        assert state['manuscript_metadata'] is None
        assert state['current_chapter'] is None
        assert len(state['chapters_completed']) == 0
        assert state['awaiting_chapter_input'] is False
        assert state['processing_complete'] is False
        assert state['illustrations_generated'] is False
        assert state['image_provider'] == ImageProvider.DALLE
        assert state['style_preferences'] == {}
        assert state['analysis_depth'] == 'basic'
        assert state['current_analysis'] is None
        assert len(state['generated_images']) == 0
        assert state['error_message'] is None
        assert state['retry_count'] == 0

    def test_manuscript_state_with_messages(self):
        """Test ManuscriptState with various message types."""
        messages = [
            SystemMessage(content="System message"),
            HumanMessage(content="Human message"),
            AIMessage(content="AI message")
        ]

        state: ManuscriptState = {
            'messages': messages,
            'manuscript_metadata': None,
            'current_chapter': None,
            'chapters_completed': [],
            'awaiting_chapter_input': False,
            'processing_complete': False,
            'illustrations_generated': False,
            'image_provider': ImageProvider.DALLE,
            'style_preferences': {},
            'analysis_depth': 'basic',
            'current_analysis': None,
            'generated_images': [],
            'error_message': None,
            'retry_count': 0
        }

        assert len(state['messages']) == 3
        assert isinstance(state['messages'][0], SystemMessage)
        assert isinstance(state['messages'][1], HumanMessage)
        assert isinstance(state['messages'][2], AIMessage)
        assert state['messages'][0].content == "System message"
        assert state['messages'][1].content == "Human message"
        assert state['messages'][2].content == "AI message"

    def test_manuscript_state_with_metadata(self, sample_metadata):
        """Test ManuscriptState with manuscript metadata."""
        state: ManuscriptState = {
            'messages': [],
            'manuscript_metadata': sample_metadata,
            'current_chapter': None,
            'chapters_completed': [],
            'awaiting_chapter_input': False,
            'processing_complete': False,
            'illustrations_generated': False,
            'image_provider': ImageProvider.DALLE,
            'style_preferences': {},
            'analysis_depth': 'basic',
            'current_analysis': None,
            'generated_images': [],
            'error_message': None,
            'retry_count': 0
        }

        assert state['manuscript_metadata'] == sample_metadata
        assert state['manuscript_metadata'].title == "Test Novel"
        assert state['manuscript_metadata'].author == "Test Author"
        assert state['manuscript_metadata'].genre == "Fantasy"

    def test_manuscript_state_with_chapter(self, sample_chapter):
        """Test ManuscriptState with current chapter."""
        state: ManuscriptState = {
            'messages': [],
            'manuscript_metadata': None,
            'current_chapter': sample_chapter,
            'chapters_completed': [],
            'awaiting_chapter_input': False,
            'processing_complete': False,
            'illustrations_generated': False,
            'image_provider': ImageProvider.DALLE,
            'style_preferences': {},
            'analysis_depth': 'basic',
            'current_analysis': None,
            'generated_images': [],
            'error_message': None,
            'retry_count': 0
        }

        assert state['current_chapter'] == sample_chapter
        assert state['current_chapter'].title == "Chapter 1"
        assert state['current_chapter'].number == 1
        assert state['current_chapter'].word_count == 5

    def test_manuscript_state_with_completed_chapters(self, sample_analysis):
        """Test ManuscriptState with completed chapter analyses."""
        state: ManuscriptState = {
            'messages': [],
            'manuscript_metadata': None,
            'current_chapter': None,
            'chapters_completed': [sample_analysis],
            'awaiting_chapter_input': False,
            'processing_complete': False,
            'illustrations_generated': False,
            'image_provider': ImageProvider.DALLE,
            'style_preferences': {},
            'analysis_depth': 'basic',
            'current_analysis': None,
            'generated_images': [],
            'error_message': None,
            'retry_count': 0
        }

        assert len(state['chapters_completed']) == 1
        assert state['chapters_completed'][0] == sample_analysis
        assert state['chapters_completed'][0].chapter.title == "Chapter 1"
        assert len(state['chapters_completed'][0].emotional_moments) == 1

    def test_manuscript_state_processing_flags(self):
        """Test ManuscriptState processing flags."""
        state: ManuscriptState = {
            'messages': [],
            'manuscript_metadata': None,
            'current_chapter': None,
            'chapters_completed': [],
            'awaiting_chapter_input': True,
            'processing_complete': True,
            'illustrations_generated': True,
            'image_provider': ImageProvider.DALLE,
            'style_preferences': {},
            'analysis_depth': 'basic',
            'current_analysis': None,
            'generated_images': [],
            'error_message': None,
            'retry_count': 0
        }

        assert state['awaiting_chapter_input'] is True
        assert state['processing_complete'] is True
        assert state['illustrations_generated'] is True

    def test_manuscript_state_image_provider_variants(self):
        """Test ManuscriptState with different image providers."""
        providers = [ImageProvider.DALLE, ImageProvider.IMAGEN4, ImageProvider.FLUX, ImageProvider.SEEDREAM]

        for provider in providers:
            state: ManuscriptState = {
                'messages': [],
                'manuscript_metadata': None,
                'current_chapter': None,
                'chapters_completed': [],
                'awaiting_chapter_input': False,
                'processing_complete': False,
                'illustrations_generated': False,
                'image_provider': provider,
                'style_preferences': {},
                'analysis_depth': 'basic',
                'current_analysis': None,
                'generated_images': [],
                'error_message': None,
                'retry_count': 0
            }

            assert state['image_provider'] == provider

    def test_manuscript_state_style_preferences(self):
        """Test ManuscriptState with style preferences."""
        style_prefs = {
            'art_style': 'watercolor',
            'color_palette': 'warm tones',
            'artistic_influences': 'Van Gogh',
            'custom_setting': 'mystical'
        }

        state: ManuscriptState = {
            'messages': [],
            'manuscript_metadata': None,
            'current_chapter': None,
            'chapters_completed': [],
            'awaiting_chapter_input': False,
            'processing_complete': False,
            'illustrations_generated': False,
            'image_provider': ImageProvider.DALLE,
            'style_preferences': style_prefs,
            'analysis_depth': 'basic',
            'current_analysis': None,
            'generated_images': [],
            'error_message': None,
            'retry_count': 0
        }

        assert state['style_preferences'] == style_prefs
        assert state['style_preferences']['art_style'] == 'watercolor'
        assert state['style_preferences']['color_palette'] == 'warm tones'
        assert state['style_preferences']['artistic_influences'] == 'Van Gogh'
        assert state['style_preferences']['custom_setting'] == 'mystical'

    def test_manuscript_state_analysis_depth_variants(self):
        """Test ManuscriptState with different analysis depths."""
        depths = ['basic', 'detailed', 'comprehensive']

        for depth in depths:
            state: ManuscriptState = {
                'messages': [],
                'manuscript_metadata': None,
                'current_chapter': None,
                'chapters_completed': [],
                'awaiting_chapter_input': False,
                'processing_complete': False,
                'illustrations_generated': False,
                'image_provider': ImageProvider.DALLE,
                'style_preferences': {},
                'analysis_depth': depth,
                'current_analysis': None,
                'generated_images': [],
                'error_message': None,
                'retry_count': 0
            }

            assert state['analysis_depth'] == depth

    def test_manuscript_state_with_current_analysis(self, sample_analysis):
        """Test ManuscriptState with current analysis."""
        state: ManuscriptState = {
            'messages': [],
            'manuscript_metadata': None,
            'current_chapter': None,
            'chapters_completed': [],
            'awaiting_chapter_input': False,
            'processing_complete': False,
            'illustrations_generated': False,
            'image_provider': ImageProvider.DALLE,
            'style_preferences': {},
            'analysis_depth': 'basic',
            'current_analysis': sample_analysis,
            'generated_images': [],
            'error_message': None,
            'retry_count': 0
        }

        assert state['current_analysis'] == sample_analysis
        assert state['current_analysis'].chapter.title == "Chapter 1"
        assert len(state['current_analysis'].emotional_moments) == 1
        assert len(state['current_analysis'].illustration_prompts) == 1

    def test_manuscript_state_with_generated_images(self):
        """Test ManuscriptState with generated images."""
        generated_images = [
            {
                'prompt_index': 0,
                'emotional_moment': 'A moment of joy',
                'image_data': 'base64_encoded_data',
                'metadata': {'provider': 'dalle', 'timestamp': '2023-01-01'}
            },
            {
                'prompt_index': 1,
                'emotional_moment': 'A moment of sadness',
                'image_data': 'base64_encoded_data_2',
                'metadata': {'provider': 'dalle', 'timestamp': '2023-01-02'}
            }
        ]

        state: ManuscriptState = {
            'messages': [],
            'manuscript_metadata': None,
            'current_chapter': None,
            'chapters_completed': [],
            'awaiting_chapter_input': False,
            'processing_complete': False,
            'illustrations_generated': False,
            'image_provider': ImageProvider.DALLE,
            'style_preferences': {},
            'analysis_depth': 'basic',
            'current_analysis': None,
            'generated_images': generated_images,
            'error_message': None,
            'retry_count': 0
        }

        assert len(state['generated_images']) == 2
        assert state['generated_images'][0]['prompt_index'] == 0
        assert state['generated_images'][0]['emotional_moment'] == 'A moment of joy'
        assert state['generated_images'][1]['prompt_index'] == 1
        assert state['generated_images'][1]['emotional_moment'] == 'A moment of sadness'

    def test_manuscript_state_error_handling(self):
        """Test ManuscriptState with error information."""
        state: ManuscriptState = {
            'messages': [],
            'manuscript_metadata': None,
            'current_chapter': None,
            'chapters_completed': [],
            'awaiting_chapter_input': False,
            'processing_complete': False,
            'illustrations_generated': False,
            'image_provider': ImageProvider.DALLE,
            'style_preferences': {},
            'analysis_depth': 'basic',
            'current_analysis': None,
            'generated_images': [],
            'error_message': 'Test error occurred',
            'retry_count': 2
        }

        assert state['error_message'] == 'Test error occurred'
        assert state['retry_count'] == 2

    def test_manuscript_state_complete_workflow(self, sample_metadata, sample_chapter, sample_analysis):
        """Test ManuscriptState representing a complete workflow."""
        messages = [
            AIMessage(content="Welcome to Manuscript Illustrator"),
            HumanMessage(content="Here's my chapter content..."),
            AIMessage(content="Analysis complete!")
        ]

        generated_images = [
            {
                'prompt_index': 0,
                'emotional_moment': 'Climactic scene',
                'image_data': 'image_data_here',
                'metadata': {'provider': 'dalle'}
            }
        ]

        style_prefs = {
            'art_style': 'digital painting',
            'color_palette': 'vibrant',
            'artistic_influences': 'Studio Ghibli'
        }

        state: ManuscriptState = {
            'messages': messages,
            'manuscript_metadata': sample_metadata,
            'current_chapter': sample_chapter,
            'chapters_completed': [sample_analysis],
            'awaiting_chapter_input': False,
            'processing_complete': True,
            'illustrations_generated': True,
            'image_provider': ImageProvider.DALLE,
            'style_preferences': style_prefs,
            'analysis_depth': 'comprehensive',
            'current_analysis': sample_analysis,
            'generated_images': generated_images,
            'error_message': None,
            'retry_count': 0
        }

        # Verify all aspects of the complete workflow state
        assert len(state['messages']) == 3
        assert state['manuscript_metadata'].title == "Test Novel"
        assert state['current_chapter'].title == "Chapter 1"
        assert len(state['chapters_completed']) == 1
        assert state['processing_complete'] is True
        assert state['illustrations_generated'] is True
        assert state['image_provider'] == ImageProvider.DALLE
        assert state['style_preferences']['art_style'] == 'digital painting'
        assert state['analysis_depth'] == 'comprehensive'
        assert state['current_analysis'] == sample_analysis
        assert len(state['generated_images']) == 1
        assert state['error_message'] is None
        assert state['retry_count'] == 0

    def test_manuscript_state_mutable_operations(self):
        """Test that ManuscriptState supports mutable operations."""
        state: ManuscriptState = {
            'messages': [],
            'manuscript_metadata': None,
            'current_chapter': None,
            'chapters_completed': [],
            'awaiting_chapter_input': False,
            'processing_complete': False,
            'illustrations_generated': False,
            'image_provider': ImageProvider.DALLE,
            'style_preferences': {},
            'analysis_depth': 'basic',
            'current_analysis': None,
            'generated_images': [],
            'error_message': None,
            'retry_count': 0
        }

        # Test adding messages
        state['messages'].append(AIMessage(content="Test message"))
        assert len(state['messages']) == 1

        # Test updating fields
        state['awaiting_chapter_input'] = True
        assert state['awaiting_chapter_input'] is True

        # Test updating retry count
        state['retry_count'] += 1
        assert state['retry_count'] == 1

        # Test adding style preferences
        state['style_preferences']['new_style'] = 'watercolor'
        assert state['style_preferences']['new_style'] == 'watercolor'

        # Test updating error message
        state['error_message'] = 'New error'
        assert state['error_message'] == 'New error'

    def test_manuscript_state_with_none_values(self):
        """Test ManuscriptState handling None values appropriately."""
        state: ManuscriptState = {
            'messages': [],
            'manuscript_metadata': None,
            'current_chapter': None,
            'chapters_completed': [],
            'awaiting_chapter_input': False,
            'processing_complete': False,
            'illustrations_generated': False,
            'image_provider': ImageProvider.DALLE,
            'style_preferences': {},
            'analysis_depth': 'basic',
            'current_analysis': None,
            'generated_images': [],
            'error_message': None,
            'retry_count': 0
        }

        assert state['manuscript_metadata'] is None
        assert state['current_chapter'] is None
        assert state['current_analysis'] is None
        assert state['error_message'] is None

    def test_manuscript_state_type_compatibility(self):
        """Test ManuscriptState type compatibility."""
        # Test that we can create a state that matches the TypedDict
        state: ManuscriptState = {
            'messages': [AIMessage(content="Test")],
            'manuscript_metadata': None,
            'current_chapter': None,
            'chapters_completed': [],
            'awaiting_chapter_input': True,
            'processing_complete': False,
            'illustrations_generated': False,
            'image_provider': ImageProvider.IMAGEN4,
            'style_preferences': {'test': 'value'},
            'analysis_depth': 'detailed',
            'current_analysis': None,
            'generated_images': [{'test': 'image'}],
            'error_message': 'Test error',
            'retry_count': 1
        }

        # Test that all fields are accessible
        assert isinstance(state['messages'], list)
        assert isinstance(state['awaiting_chapter_input'], bool)
        assert isinstance(state['processing_complete'], bool)
        assert isinstance(state['illustrations_generated'], bool)
        assert isinstance(state['image_provider'], ImageProvider)
        assert isinstance(state['style_preferences'], dict)
        assert isinstance(state['analysis_depth'], str)
        assert isinstance(state['generated_images'], list)
        assert isinstance(state['retry_count'], int)


class TestStateEdgeCases:
    """Test edge cases and potential issues with state management."""

    def test_state_with_large_data(self):
        """Test state handling with large amounts of data."""
        # Create large message list
        large_message_list = [
            AIMessage(content=f"Message {i}") for i in range(1000)
        ]

        # Create large style preferences
        large_style_prefs = {f"style_{i}": f"value_{i}" for i in range(100)}

        # Create large generated images list
        large_images = [
            {'image_id': i, 'data': f'data_{i}'} for i in range(50)
        ]

        state: ManuscriptState = {
            'messages': large_message_list,
            'manuscript_metadata': None,
            'current_chapter': None,
            'chapters_completed': [],
            'awaiting_chapter_input': False,
            'processing_complete': False,
            'illustrations_generated': False,
            'image_provider': ImageProvider.DALLE,
            'style_preferences': large_style_prefs,
            'analysis_depth': 'basic',
            'current_analysis': None,
            'generated_images': large_images,
            'error_message': None,
            'retry_count': 0
        }

        assert len(state['messages']) == 1000
        assert len(state['style_preferences']) == 100
        assert len(state['generated_images']) == 50

    def test_state_deep_copy_safety(self):
        """Test that state can be safely deep copied."""
        import copy

        original_state: ManuscriptState = {
            'messages': [AIMessage(content="Test")],
            'manuscript_metadata': None,
            'current_chapter': None,
            'chapters_completed': [],
            'awaiting_chapter_input': False,
            'processing_complete': False,
            'illustrations_generated': False,
            'image_provider': ImageProvider.DALLE,
            'style_preferences': {'style': 'value'},
            'analysis_depth': 'basic',
            'current_analysis': None,
            'generated_images': [],
            'error_message': None,
            'retry_count': 0
        }

        copied_state = copy.deepcopy(original_state)

        assert copied_state['messages'][0].content == original_state['messages'][0].content
        assert copied_state['image_provider'] == original_state['image_provider']
        assert copied_state['style_preferences'] == original_state['style_preferences']

        # Modify copied state to ensure independence
        copied_state['retry_count'] = 5
        assert original_state['retry_count'] == 0
        assert copied_state['retry_count'] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
