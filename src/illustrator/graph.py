"""Main LangGraph workflow for manuscript analysis and illustration generation."""

import logging
from datetime import datetime
from typing import Any, Dict, List, cast

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore

from illustrator.analysis import EmotionalAnalyzer
from illustrator.context import ManuscriptContext
from illustrator.models import (
    ChapterAnalysis,
    EmotionalMoment,
)
from illustrator.providers import ProviderFactory
from illustrator.state import ManuscriptState

logger = logging.getLogger(__name__)


async def initialize_session(state: ManuscriptState, runtime: Runtime[ManuscriptContext]) -> Dict[str, Any]:
    """Initialize a new manuscript analysis session."""
    logger.info("Initializing new manuscript analysis session")

    # Set up initial state
    init_message = AIMessage(content="""
Welcome to Manuscript Illustrator! 📚✨

I'm here to help you analyze your manuscript chapters and generate optimal illustration prompts for AI image generation.

Here's what I can do:
• Analyze chapters for emotional resonance and key moments
• Generate optimized prompts for DALL-E, Imagen4, or Flux
• Identify the most visually compelling scenes
• Provide detailed literary analysis

Ready to begin? Please share your first chapter with me.
""")

    return {
        "messages": [init_message],
        "awaiting_chapter_input": True,
        "processing_complete": False,
        "chapters_completed": [],
        "error_message": None,
        "retry_count": 0,
    }


async def analyze_chapter(state: ManuscriptState, runtime: Runtime[ManuscriptContext]) -> Dict[str, Any]:
    """Analyze a chapter for emotional moments and themes."""
    logger.info("Starting chapter analysis")

    if not state.get("current_chapter"):
        return {
            "error_message": "No chapter provided for analysis",
            "retry_count": state.get("retry_count", 0) + 1,
        }

    try:
        # Initialize the LLM and analyzer
        llm = init_chat_model(model=runtime.context.model)
        analyzer = EmotionalAnalyzer(llm)

        chapter = state["current_chapter"]
        context = runtime.context

        # Perform emotional analysis
        emotional_moments = await analyzer.analyze_chapter(
            chapter=chapter,
            max_moments=context.max_emotional_moments,
            min_intensity=context.min_intensity_threshold,
        )

        # Generate literary analysis using Claude
        analysis_prompt = context.analysis_prompt.format(
            time=datetime.now().isoformat(),
            user_preferences=_format_user_preferences(context),
            chapter_context=f"Chapter {chapter.number}: {chapter.title}",
        )

        analysis_messages = [
            SystemMessage(content=analysis_prompt),
            HumanMessage(content=f"""
Chapter Title: {chapter.title}
Word Count: {chapter.word_count:,}

Chapter Content:
{chapter.content}

Please provide a detailed analysis of this chapter, focusing on:
1. Dominant themes and motifs
2. Setting and atmospheric descriptions
3. Character emotional arcs
4. Visual elements that would translate well to illustrations
5. Overall mood and tone

Return your analysis in JSON format with these fields:
- dominant_themes: List of key themes
- setting_description: Description of the physical/atmospheric setting
- character_emotions: Object mapping character names to their emotional states
- visual_highlights: List of the most visually compelling moments
""")
        ]

        analysis_response = await llm.ainvoke(analysis_messages)

        # Parse the analysis (with error handling)
        try:
            import json
            analysis_data = json.loads(analysis_response.content.strip())

            dominant_themes = analysis_data.get('dominant_themes', [])
            setting_description = analysis_data.get('setting_description', 'Rich atmospheric setting')

            # Validate character emotions to ensure they use valid EmotionalTone values
            raw_character_emotions = analysis_data.get('character_emotions', {})
            character_emotions = {}

            from illustrator.models import EmotionalTone
            valid_tones = set(tone.value for tone in EmotionalTone)

            for character, emotions in raw_character_emotions.items():
                if isinstance(emotions, list):
                    # Filter out invalid emotion strings
                    valid_emotions = [EmotionalTone(emotion) for emotion in emotions if emotion in valid_tones]
                    if valid_emotions:
                        character_emotions[character] = valid_emotions

        except (json.JSONDecodeError, AttributeError):
            # Fallback if JSON parsing fails
            dominant_themes = ["Character development", "Narrative progression"]
            setting_description = f"Chapter {chapter.number} setting with rich atmospheric detail"
            character_emotions = {}

        # Generate illustration prompts for each emotional moment
        illustration_prompts = []

        if emotional_moments:
            # Initialize the image provider
            provider = ProviderFactory.create_provider(
                context.image_provider,
                openai_api_key=context.openai_api_key,
                google_credentials=context.google_credentials,
                google_project_id=runtime.context.user_id,  # Use user_id as project_id fallback
                huggingface_api_key=context.huggingface_api_key,
            )

            # Create style preferences dict
            style_preferences = {
                'art_style': context.default_art_style,
                'color_palette': context.color_palette,
                'artistic_influences': context.artistic_influences,
            }

            for moment in emotional_moments:
                prompt = await provider.generate_prompt(
                    emotional_moment=moment,
                    style_preferences=style_preferences,
                    context=setting_description,
                )
                illustration_prompts.append(prompt)

        # Create complete analysis
        chapter_analysis = ChapterAnalysis(
            chapter=chapter,
            emotional_moments=emotional_moments,
            dominant_themes=dominant_themes,
            setting_description=setting_description,
            character_emotions=character_emotions,
            illustration_prompts=illustration_prompts,
        )

        # Store analysis in the store for persistence
        await cast(BaseStore, runtime.store).aput(
            ("chapter_analyses", runtime.context.user_id),
            f"chapter_{chapter.number}",
            chapter_analysis.model_dump(),
        )

        # Generate response message
        response_content = f"""
✅ **Chapter {chapter.number} Analysis Complete**

📊 **Analysis Summary:**
- **{len(emotional_moments)}** high-intensity emotional moments identified
- **{len(illustration_prompts)}** illustration prompts generated
- **{chapter.word_count:,}** words analyzed

🎨 **Key Themes:** {', '.join(dominant_themes[:3])}

🖼️ **Visual Opportunities:**
{_format_emotional_moments(emotional_moments[:3])}

📝 **Setting:** {setting_description[:200]}{"..." if len(setting_description) > 200 else ""}

Your chapter has been analyzed and {len(illustration_prompts)} optimized prompts have been generated for {context.image_provider.value.upper()}.
"""

        return {
            "messages": [AIMessage(content=response_content)],
            "current_analysis": chapter_analysis,
            "awaiting_chapter_input": False,
            "error_message": None,
            "retry_count": 0,
        }

    except Exception as e:
        logger.error(f"Error during chapter analysis: {e}")
        return {
            "error_message": f"Analysis failed: {str(e)}",
            "retry_count": state.get("retry_count", 0) + 1,
        }


async def generate_illustrations(state: ManuscriptState, runtime: Runtime[ManuscriptContext]) -> Dict[str, Any]:
    """Generate actual illustrations using the selected provider."""
    if not state.get("current_analysis"):
        return {
            "error_message": "No analysis available for illustration generation",
            "retry_count": state.get("retry_count", 0) + 1,
        }

    try:
        analysis = state["current_analysis"]
        context = runtime.context

        # Initialize the image provider
        provider = ProviderFactory.create_provider(
            context.image_provider,
            openai_api_key=context.openai_api_key,
            google_credentials=context.google_credentials,
            google_project_id=runtime.context.user_id,
            huggingface_api_key=context.huggingface_api_key,
        )

        generated_images = []

        # Generate images for each illustration prompt
        for i, prompt in enumerate(analysis.illustration_prompts):
            try:
                result = await provider.generate_image(prompt)

                if result.get('success'):
                    generated_images.append({
                        'prompt_index': i,
                        'emotional_moment': analysis.emotional_moments[i].text_excerpt,
                        'image_data': result['image_data'],
                        'metadata': result['metadata'],
                    })
                else:
                    logger.error(f"Image generation failed for prompt {i}: {result.get('error')}")

            except Exception as e:
                logger.error(f"Error generating image {i}: {e}")

        # Store generated images
        if generated_images:
            await cast(BaseStore, runtime.store).aput(
                ("generated_images", runtime.context.user_id),
                f"chapter_{analysis.chapter.number}_images",
                generated_images,
            )

        response_content = f"""
🎨 **Illustration Generation Complete**

Generated **{len(generated_images)}** illustrations for Chapter {analysis.chapter.number}

{_format_generation_results(generated_images)}

All images have been saved and are ready for download.
"""

        return {
            "messages": [AIMessage(content=response_content)],
            "illustrations_generated": True,
            "generated_images": generated_images,
            "error_message": None,
            "retry_count": 0,
        }

    except Exception as e:
        logger.error(f"Error during illustration generation: {e}")
        return {
            "error_message": f"Illustration generation failed: {str(e)}",
            "retry_count": state.get("retry_count", 0) + 1,
        }


async def complete_chapter(state: ManuscriptState, runtime: Runtime[ManuscriptContext]) -> Dict[str, Any]:
    """Mark current chapter as complete and prepare for next chapter."""
    if state.get("current_analysis"):
        # Add current analysis to completed chapters
        completed_chapters = list(state.get("chapters_completed", []))
        completed_chapters.append(state["current_analysis"])

        # Count generated images
        generated_images = state.get("generated_images", [])
        image_count = len(generated_images)

        response_content = f"""
✅ **Chapter {state['current_analysis'].chapter.number} Processing Complete!**

You now have:
- Detailed emotional analysis
- {len(state['current_analysis'].illustration_prompts)} optimized illustration prompts
- {image_count} generated images

Ready for your next chapter! Please share the next chapter content, or let me know if you're finished.
"""

        return {
            "messages": [AIMessage(content=response_content)],
            "chapters_completed": completed_chapters,
            "current_chapter": None,
            "current_analysis": None,
            "generated_images": generated_images,  # Pass through generated images
            "awaiting_chapter_input": True,
            "error_message": None,
        }

    return {"awaiting_chapter_input": True}


async def handle_error(state: ManuscriptState, runtime: Runtime[ManuscriptContext]) -> Dict[str, Any]:
    """Handle errors and provide user feedback."""
    error_msg = state.get("error_message", "Unknown error occurred")
    retry_count = state.get("retry_count", 0)

    if retry_count < 3:
        response_content = f"""
⚠️ **Processing Error**

{error_msg}

I'll try to process this again. This is attempt {retry_count + 1} of 3.
"""
    else:
        response_content = f"""
❌ **Processing Failed**

{error_msg}

I've tried multiple times but couldn't process this chapter. Please check:
- Your API keys are valid
- The chapter content is properly formatted
- Your internet connection is stable

You can try again with a new chapter or contact support.
"""

    return {
        "messages": [AIMessage(content=response_content)],
        "awaiting_chapter_input": True if retry_count >= 3 else False,
    }


def route_next_step(state: ManuscriptState) -> str:
    """Determine the next step in the workflow."""
    # Check for errors
    if state.get("error_message") and state.get("retry_count", 0) > 0:
        return "handle_error"

    # Check if we're waiting for chapter input
    if state.get("awaiting_chapter_input", False):
        return END  # Wait for user input

    # If we have a current chapter but no analysis, analyze it
    if state.get("current_chapter") and not state.get("current_analysis"):
        return "analyze_chapter"

    # If we have analysis but no generated images, generate illustrations
    if state.get("current_analysis") and not state.get("illustrations_generated", False):
        return "generate_illustrations"

    # If we have analysis and illustrations, complete the chapter
    if state.get("current_analysis") and state.get("illustrations_generated", False):
        return "complete_chapter"

    return END


# Helper functions
def _format_user_preferences(context: ManuscriptContext) -> str:
    """Format user preferences for the analysis prompt."""
    prefs = []
    prefs.append(f"Image provider: {context.image_provider.value}")
    prefs.append(f"Art style: {context.default_art_style}")

    if context.color_palette:
        prefs.append(f"Color palette: {context.color_palette}")

    if context.artistic_influences:
        prefs.append(f"Artistic influences: {context.artistic_influences}")

    return "; ".join(prefs)


def _format_emotional_moments(moments: List[EmotionalMoment]) -> str:
    """Format emotional moments for display."""
    if not moments:
        return "No high-intensity moments identified."

    formatted = []
    for i, moment in enumerate(moments, 1):
        excerpt = moment.text_excerpt[:100] + "..." if len(moment.text_excerpt) > 100 else moment.text_excerpt
        tones = ", ".join([tone.value.title() for tone in moment.emotional_tones[:2]])
        formatted.append(f"{i}. **{tones}** - {excerpt}")

    return "\n".join(formatted)


def _format_generation_results(generated_images: List[Dict[str, Any]]) -> str:
    """Format image generation results for display."""
    if not generated_images:
        return "No images were generated due to errors."

    results = []
    for img in generated_images:
        excerpt = img['emotional_moment'][:80] + "..." if len(img['emotional_moment']) > 80 else img['emotional_moment']
        results.append(f"• **Scene:** {excerpt}")

    return "\n".join(results)


# Create the graph
builder = StateGraph(ManuscriptState, context_schema=ManuscriptContext)

# Add nodes
builder.add_node("initialize_session", initialize_session)
builder.add_node("analyze_chapter", analyze_chapter)
builder.add_node("generate_illustrations", generate_illustrations)
builder.add_node("complete_chapter", complete_chapter)
builder.add_node("handle_error", handle_error)

# Add edges
builder.add_edge("__start__", "initialize_session")
builder.add_edge("initialize_session", END)

# Add conditional routing
builder.add_conditional_edges(
    "analyze_chapter",
    route_next_step,
    ["generate_illustrations", "complete_chapter", "handle_error", END]
)

builder.add_conditional_edges(
    "generate_illustrations",
    route_next_step,
    ["complete_chapter", "handle_error", END]
)

builder.add_edge("complete_chapter", END)
builder.add_edge("handle_error", END)

# Compile the graph
graph = builder.compile()
graph.name = "ManuscriptIllustrator"

__all__ = ["graph"]