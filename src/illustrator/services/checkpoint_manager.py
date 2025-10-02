"""
Checkpoint manager for handling processing milestones and recovery points.
Provides high-level checkpoint creation and resume functionality.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from .session_persistence import SessionPersistenceService, SessionState
from ..models import EmotionalMoment


class CheckpointType(Enum):
    """Types of checkpoints in the processing workflow."""
    SESSION_START = "session_start"
    MANUSCRIPT_LOADED = "manuscript_loaded"
    CHAPTER_START = "chapter_start"
    CHAPTER_ANALYZED = "chapter_analyzed"
    PROMPTS_GENERATED = "prompts_generated"
    IMAGES_GENERATING = "images_generating"
    CHAPTER_COMPLETED = "chapter_completed"
    SESSION_COMPLETED = "session_completed"
    SESSION_PAUSED = "session_paused"
    ERROR_OCCURRED = "error_occurred"


class ProcessingStep(Enum):
    """Processing steps for workflow management."""
    INITIALIZING = "initializing"
    LOADING_MANUSCRIPT = "loading_manuscript"
    ANALYZING_CHAPTERS = "analyzing_chapters"
    GENERATING_PROMPTS = "generating_prompts"
    GENERATING_IMAGES = "generating_images"
    COMPLETING_SESSION = "completing_session"


class CheckpointManager:
    """Manager for creating and managing processing checkpoints."""

    def __init__(self, session_persistence: Optional[SessionPersistenceService] = None):
        """Initialize the checkpoint manager.

        Args:
            session_persistence: Session persistence service (optional, will create if not provided)
        """
        self.persistence = session_persistence or SessionPersistenceService()
        self.checkpoint_sequence = {}  # Track sequence numbers per session

    def create_session_start_checkpoint(self,
                                      session_id: str,
                                      manuscript_id: str,
                                      manuscript_title: str,
                                      total_chapters: int,
                                      style_config: Dict[str, Any],
                                      max_emotional_moments: int) -> str:
        """Create a checkpoint for session start.

        Args:
            session_id: Session ID
            manuscript_id: Manuscript ID
            manuscript_title: Manuscript title
            total_chapters: Total number of chapters
            style_config: Style configuration
            max_emotional_moments: Max emotional moments per chapter

        Returns:
            Checkpoint ID (UUID string)
        """
        checkpoint_data = {
            "manuscript_id": manuscript_id,
            "manuscript_title": manuscript_title,
            "total_chapters": total_chapters,
            "style_config": style_config,
            "max_emotional_moments": max_emotional_moments,
            "processing_plan": self._create_processing_plan(total_chapters)
        }

        checkpoint = self.persistence.create_checkpoint(
            session_id=session_id,
            checkpoint_type=CheckpointType.SESSION_START.value,
            chapter_number=0,
            step_name=ProcessingStep.INITIALIZING.value,
            sequence_number=self._get_next_sequence_number(session_id),
            checkpoint_data=checkpoint_data,
            progress_percent=5,
            next_action="load_manuscript"
        )

        self.persistence.log_session_event(
            session_id=session_id,
            level="info",
            message=f"Session started for manuscript: {manuscript_title} ({total_chapters} chapters)"
        )

        return str(checkpoint["id"])

    def create_manuscript_loaded_checkpoint(self,
                                          session_id: str,
                                          chapters_info: List[Dict[str, Any]]) -> str:
        """Create a checkpoint after manuscript is loaded.

        Args:
            session_id: Session ID
            chapters_info: Information about loaded chapters

        Returns:
            Checkpoint ID (UUID string)
        """
        checkpoint_data = {
            "chapters_loaded": len(chapters_info),
            "chapters_info": chapters_info,
            "processing_order": [chapter["number"] for chapter in chapters_info]
        }

        checkpoint = self.persistence.create_checkpoint(
            session_id=session_id,
            checkpoint_type=CheckpointType.MANUSCRIPT_LOADED.value,
            chapter_number=0,
            step_name=ProcessingStep.LOADING_MANUSCRIPT.value,
            sequence_number=self._get_next_sequence_number(session_id),
            checkpoint_data=checkpoint_data,
            progress_percent=10,
            next_action="analyze_first_chapter"
        )

        self.persistence.log_session_event(
            session_id=session_id,
            level="info",
            message=f"Manuscript loaded successfully with {len(chapters_info)} chapters"
        )

        return str(checkpoint["id"])

    def create_chapter_start_checkpoint(self,
                                      session_id: str,
                                      chapter_number: int,
                                      chapter_title: str,
                                      chapter_word_count: int,
                                      progress_percent: int) -> str:
        """Create a checkpoint at the start of chapter processing.

        Args:
            session_id: Session ID
            chapter_number: Chapter number being processed
            chapter_title: Chapter title
            chapter_word_count: Word count of the chapter
            progress_percent: Current progress percentage

        Returns:
            Checkpoint ID (UUID string)
        """
        checkpoint_data = {
            "chapter_number": chapter_number,
            "chapter_title": chapter_title,
            "chapter_word_count": chapter_word_count,
            "analysis_started": True
        }

        checkpoint = self.persistence.create_checkpoint(
            session_id=session_id,
            checkpoint_type=CheckpointType.CHAPTER_START.value,
            chapter_number=chapter_number,
            step_name=ProcessingStep.ANALYZING_CHAPTERS.value,
            sequence_number=self._get_next_sequence_number(session_id),
            checkpoint_data=checkpoint_data,
            progress_percent=progress_percent,
            next_action="analyze_chapter_content"
        )

        self.persistence.log_session_event(
            session_id=session_id,
            level="info",
            message=f"Started processing Chapter {chapter_number}: {chapter_title}",
            chapter_number=chapter_number,
            step_name=ProcessingStep.ANALYZING_CHAPTERS.value
        )

        return str(checkpoint["id"])

    def create_chapter_analyzed_checkpoint(self,
                                         session_id: str,
                                         chapter_number: int,
                                         emotional_moments: List[EmotionalMoment],
                                         analysis_results: Dict[str, Any],
                                         progress_percent: int) -> str:
        """Create a checkpoint after chapter analysis is complete.

        Args:
            session_id: Session ID
            chapter_number: Chapter number
            emotional_moments: Found emotional moments
            analysis_results: Analysis results
            progress_percent: Current progress percentage

        Returns:
            Checkpoint ID (UUID string)
        """
        # Serialize emotional moments for storage
        emotional_moments_data = []
        for moment in emotional_moments:
            moment_data = {
                "text_excerpt": moment.text_excerpt,
                "emotional_tones": [tone.value for tone in moment.emotional_tones],
                "intensity_score": moment.intensity_score,
                "characters_present": moment.characters_present,
                "setting_description": moment.setting_description,
                "narrative_context": moment.narrative_context
            }
            emotional_moments_data.append(moment_data)

        checkpoint_data = {
            "chapter_number": chapter_number,
            "emotional_moments_found": len(emotional_moments),
            "analysis_results": analysis_results,
            "analysis_completed": True
        }

        checkpoint = self.persistence.create_checkpoint(
            session_id=session_id,
            checkpoint_type=CheckpointType.CHAPTER_ANALYZED.value,
            chapter_number=chapter_number,
            step_name=ProcessingStep.ANALYZING_CHAPTERS.value,
            sequence_number=self._get_next_sequence_number(session_id),
            checkpoint_data=checkpoint_data,
            progress_percent=progress_percent,
            emotional_moments_data=emotional_moments_data,
            next_action="generate_prompts"
        )

        self.persistence.log_session_event(
            session_id=session_id,
            level="info",
            message=f"Chapter {chapter_number} analyzed - found {len(emotional_moments)} emotional moments",
            chapter_number=chapter_number,
            step_name=ProcessingStep.ANALYZING_CHAPTERS.value
        )

        return str(checkpoint["id"])

    def create_prompts_generated_checkpoint(self,
                                          session_id: str,
                                          chapter_number: int,
                                          generated_prompts: List[str],
                                          prompts_metadata: List[Dict[str, Any]],
                                          progress_percent: int) -> str:
        """Create a checkpoint after prompts are generated.

        Args:
            session_id: Session ID
            chapter_number: Chapter number
            generated_prompts: Generated prompts
            prompts_metadata: Metadata for each prompt
            progress_percent: Current progress percentage

        Returns:
            Checkpoint ID (UUID string)
        """
        checkpoint_data = {
            "chapter_number": chapter_number,
            "prompts_generated": len(generated_prompts),
            "prompts_metadata": prompts_metadata,
            "ready_for_generation": True
        }

        checkpoint = self.persistence.create_checkpoint(
            session_id=session_id,
            checkpoint_type=CheckpointType.PROMPTS_GENERATED.value,
            chapter_number=chapter_number,
            step_name=ProcessingStep.GENERATING_PROMPTS.value,
            sequence_number=self._get_next_sequence_number(session_id),
            checkpoint_data=checkpoint_data,
            generated_prompts=generated_prompts,
            progress_percent=progress_percent,
            next_action="generate_images"
        )

        self.persistence.log_session_event(
            session_id=session_id,
            level="info",
            message=f"Generated {len(generated_prompts)} prompts for Chapter {chapter_number}",
            chapter_number=chapter_number,
            step_name=ProcessingStep.GENERATING_PROMPTS.value
        )

        return str(checkpoint["id"])

    def create_images_generating_checkpoint(self,
                                          session_id: str,
                                          chapter_number: int,
                                          images_to_generate: int,
                                          current_image_index: int,
                                          progress_percent: int) -> str:
        """Create a checkpoint during image generation.

        Args:
            session_id: Session ID
            chapter_number: Chapter number
            images_to_generate: Total images to generate
            current_image_index: Current image being generated (0-based)
            progress_percent: Current progress percentage

        Returns:
            Checkpoint ID (UUID string)
        """
        checkpoint_data = {
            "chapter_number": chapter_number,
            "images_to_generate": images_to_generate,
            "current_image_index": current_image_index,
            "generation_in_progress": True
        }

        checkpoint = self.persistence.create_checkpoint(
            session_id=session_id,
            checkpoint_type=CheckpointType.IMAGES_GENERATING.value,
            chapter_number=chapter_number,
            step_name=ProcessingStep.GENERATING_IMAGES.value,
            sequence_number=self._get_next_sequence_number(session_id),
            checkpoint_data=checkpoint_data,
            images_generated_count=current_image_index,
            progress_percent=progress_percent,
            next_action=f"generate_image_{current_image_index}"
        )

        return str(checkpoint["id"])

    def create_chapter_completed_checkpoint(self,
                                          session_id: str,
                                          chapter_number: int,
                                          images_generated: int,
                                          total_images_so_far: int,
                                          progress_percent: int) -> str:
        """Create a checkpoint after chapter processing is complete.

        Args:
            session_id: Session ID
            chapter_number: Chapter number
            images_generated: Images generated for this chapter
            total_images_so_far: Total images generated so far
            progress_percent: Current progress percentage

        Returns:
            Checkpoint ID (UUID string)
        """
        checkpoint_data = {
            "chapter_number": chapter_number,
            "images_generated": images_generated,
            "chapter_completed": True
        }

        checkpoint = self.persistence.create_checkpoint(
            session_id=session_id,
            checkpoint_type=CheckpointType.CHAPTER_COMPLETED.value,
            chapter_number=chapter_number,
            step_name=ProcessingStep.GENERATING_IMAGES.value,
            sequence_number=self._get_next_sequence_number(session_id),
            checkpoint_data=checkpoint_data,
            images_generated_count=images_generated,
            total_images_at_checkpoint=total_images_so_far,
            progress_percent=progress_percent,
            next_action="process_next_chapter" if chapter_number > 0 else "complete_session"
        )

        self.persistence.log_session_event(
            session_id=session_id,
            level="success",
            message=f"Chapter {chapter_number} completed - generated {images_generated} images",
            chapter_number=chapter_number,
            step_name=ProcessingStep.GENERATING_IMAGES.value
        )

        return str(checkpoint["id"])

    def create_session_completed_checkpoint(self,
                                          session_id: str,
                                          total_images_generated: int,
                                          total_chapters_processed: int) -> str:
        """Create a checkpoint for completed session.

        Args:
            session_id: Session ID
            total_images_generated: Total images generated
            total_chapters_processed: Total chapters processed

        Returns:
            Checkpoint ID (UUID string)
        """
        checkpoint_data = {
            "session_completed": True,
            "total_images_generated": total_images_generated,
            "total_chapters_processed": total_chapters_processed,
            "completion_time": datetime.utcnow().isoformat()
        }

        checkpoint = self.persistence.create_checkpoint(
            session_id=session_id,
            checkpoint_type=CheckpointType.SESSION_COMPLETED.value,
            chapter_number=total_chapters_processed,
            step_name=ProcessingStep.COMPLETING_SESSION.value,
            sequence_number=self._get_next_sequence_number(session_id),
            checkpoint_data=checkpoint_data,
            total_images_at_checkpoint=total_images_generated,
            progress_percent=100,
            next_action="session_complete"
        )

        self.persistence.log_session_event(
            session_id=session_id,
            level="success",
            message=f"Session completed successfully! Generated {total_images_generated} images across {total_chapters_processed} chapters"
        )

        return str(checkpoint["id"])

    def create_pause_checkpoint(self,
                               session_id: str,
                               chapter_number: int,
                               current_step: ProcessingStep,
                               progress_percent: int,
                               pause_reason: str = "user_requested") -> str:
        """Create a checkpoint when session is paused.

        Args:
            session_id: Session ID
            chapter_number: Current chapter number
            current_step: Current processing step
            progress_percent: Current progress percentage
            pause_reason: Reason for pausing

        Returns:
            Checkpoint ID (UUID string)
        """
        checkpoint_data = {
            "chapter_number": chapter_number,
            "current_step": current_step.value,
            "pause_reason": pause_reason,
            "paused_at": datetime.utcnow().isoformat(),
            "session_paused": True
        }

        checkpoint = self.persistence.create_checkpoint(
            session_id=session_id,
            checkpoint_type=CheckpointType.SESSION_PAUSED.value,
            chapter_number=chapter_number,
            step_name=current_step.value,
            sequence_number=self._get_next_sequence_number(session_id),
            checkpoint_data=checkpoint_data,
            progress_percent=progress_percent,
            next_action="resume_processing"
        )

        self.persistence.log_session_event(
            session_id=session_id,
            level="info",
            message=f"Session paused at Chapter {chapter_number} during {current_step.value}",
            chapter_number=chapter_number,
            step_name=current_step.value
        )

        return str(checkpoint["id"])

    def create_error_checkpoint(self,
                               session_id: str,
                               chapter_number: int,
                               current_step: ProcessingStep,
                               error_message: str,
                               error_details: Dict[str, Any],
                               progress_percent: int) -> str:
        """Create a checkpoint when an error occurs.

        Args:
            session_id: Session ID
            chapter_number: Current chapter number
            current_step: Current processing step
            error_message: Error message
            error_details: Additional error details
            progress_percent: Current progress percentage

        Returns:
            Checkpoint ID (UUID string)
        """
        checkpoint_data = {
            "chapter_number": chapter_number,
            "current_step": current_step.value,
            "error_message": error_message,
            "error_details": error_details,
            "error_occurred": True,
            "error_time": datetime.utcnow().isoformat()
        }

        checkpoint = self.persistence.create_checkpoint(
            session_id=session_id,
            checkpoint_type=CheckpointType.ERROR_OCCURRED.value,
            chapter_number=chapter_number,
            step_name=current_step.value,
            sequence_number=self._get_next_sequence_number(session_id),
            checkpoint_data=checkpoint_data,
            progress_percent=progress_percent,
            is_resumable=True,  # Errors can usually be recovered from
            next_action="retry_step"
        )

        self.persistence.log_session_event(
            session_id=session_id,
            level="error",
            message=f"Error in Chapter {chapter_number} during {current_step.value}: {error_message}",
            chapter_number=chapter_number,
            step_name=current_step.value
        )

        return str(checkpoint["id"])

    def get_resume_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information needed to resume a session.

        Args:
            session_id: Session ID

        Returns:
            Resume information dictionary or None
        """
        resume_data = self.persistence.get_session_for_resume(session_id)
        if not resume_data:
            return None

        db_session, session_state, latest_checkpoint = resume_data

        resume_info = {
            "session_id": session_id,
            "manuscript_id": session_state.manuscript_id,
            "external_session_id": session_state.external_session_id,
            "status": session_state.status,
            "progress_percent": session_state.progress_percent,
            "current_chapter": session_state.current_chapter,
            "total_chapters": session_state.total_chapters,
            "last_completed_chapter": session_state.last_completed_chapter,
            "total_images_generated": session_state.total_images_generated,
            "style_config": session_state.style_config,
            "max_emotional_moments": session_state.max_emotional_moments,
            "can_resume": True,
            "paused_at": session_state.paused_at,
            "generated_images": session_state.generated_images
        }

        if latest_checkpoint:
            checkpoint_data = latest_checkpoint.get("checkpoint_data") or {}
            resume_info.update({
                "latest_checkpoint_id": latest_checkpoint["id"],
                "latest_checkpoint_type": latest_checkpoint.get("checkpoint_type"),
                "next_action": latest_checkpoint.get("next_action"),
                "checkpoint_data": checkpoint_data,
            })

            checkpoint_type = latest_checkpoint.get("checkpoint_type")
            emotional_moments_data = latest_checkpoint.get("emotional_moments_data") or []
            generated_prompts = latest_checkpoint.get("generated_prompts") or []

            if checkpoint_type == CheckpointType.CHAPTER_ANALYZED.value:
                resume_info["resume_at"] = "prompts_generation"
                resume_info["emotional_moments"] = emotional_moments_data
            elif checkpoint_type == CheckpointType.PROMPTS_GENERATED.value:
                resume_info["resume_at"] = "image_generation"
                resume_info["generated_prompts"] = generated_prompts
            elif checkpoint_type == CheckpointType.IMAGES_GENERATING.value:
                resume_info["resume_at"] = "continue_image_generation"
                resume_info["current_image_index"] = checkpoint_data.get("current_image_index", 0)
            elif checkpoint_type == CheckpointType.CHAPTER_COMPLETED.value:
                resume_info["resume_at"] = "next_chapter"
            elif checkpoint_type == CheckpointType.ERROR_OCCURRED.value:
                resume_info["resume_at"] = "retry_from_error"
                resume_info["error_details"] = checkpoint_data

        return resume_info

    def _create_processing_plan(self, total_chapters: int) -> Dict[str, Any]:
        """Create a processing plan for the session.

        Args:
            total_chapters: Total number of chapters

        Returns:
            Processing plan dictionary
        """
        return {
            "total_chapters": total_chapters,
            "estimated_steps": total_chapters * 4,  # analyze, prompts, images, complete per chapter
            "steps": [
                f"Chapter {i}: Analyze" for i in range(1, total_chapters + 1)
            ] + [
                f"Chapter {i}: Generate Prompts" for i in range(1, total_chapters + 1)
            ] + [
                f"Chapter {i}: Generate Images" for i in range(1, total_chapters + 1)
            ] + [
                f"Chapter {i}: Complete" for i in range(1, total_chapters + 1)
            ]
        }

    def _get_next_sequence_number(self, session_id: str) -> int:
        """Get the next sequence number for a session.

        Args:
            session_id: Session ID

        Returns:
            Next sequence number
        """
        if session_id not in self.checkpoint_sequence:
            self.checkpoint_sequence[session_id] = 0

        self.checkpoint_sequence[session_id] += 1
        return self.checkpoint_sequence[session_id]

    def close(self):
        """Close the checkpoint manager and its dependencies."""
        if self.persistence:
            self.persistence.close()
