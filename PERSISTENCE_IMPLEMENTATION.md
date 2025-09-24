# Session Persistence and Checkpoint Implementation

This document describes the complete implementation of database persistence, processing checkpoints, resume logic, and file-based session storage for the Illustrator application.

## 🎯 Implementation Overview

The implementation provides four key capabilities:

1. **Database Persistence**: Store session data, progress, and checkpoints in a database
2. **Processing Checkpoints**: Save processing state at key milestones
3. **Resume Logic**: Detect and resume from last checkpoint
4. **File-Based Session Storage**: JSON files for server restart recovery

## 🗄️ Database Models

### Core Tables

#### `ProcessingSession`
- **Purpose**: Main session tracking table
- **Key Fields**:
  - `id` (UUID): Primary key
  - `manuscript_id` (UUID): Associated manuscript
  - `external_session_id` (String): Web session ID
  - `status` (String): Current status (pending, running, completed, failed, paused)
  - `progress_percent` (Integer): 0-100 progress
  - `current_chapter` (Integer): Currently processing chapter
  - `total_chapters` (Integer): Total chapters to process
  - `style_config` (JSON): Style configuration used
  - `max_emotional_moments` (Integer): Max moments per chapter
  - `can_resume` (Boolean): Whether session can be resumed

#### `ProcessingCheckpoint`
- **Purpose**: Store processing checkpoints for resume capability
- **Key Fields**:
  - `id` (UUID): Primary key
  - `session_id` (UUID): Foreign key to ProcessingSession
  - `checkpoint_type` (String): Type of checkpoint (see types below)
  - `chapter_number` (Integer): Chapter being processed
  - `sequence_number` (Integer): Order of checkpoints
  - `checkpoint_data` (JSON): Comprehensive checkpoint state
  - `processing_state` (JSON): Detailed processing state
  - `generated_prompts` (JSON): Prompts generated so far
  - `emotional_moments_data` (JSON): Analyzed emotional moments
  - `next_action` (String): What to do next when resuming

#### `ProcessingLog`
- **Purpose**: Detailed logging for debugging and state tracking
- **Key Fields**:
  - `session_id` (UUID): Associated session
  - `level` (String): Log level (info, warning, error, success)
  - `message` (Text): Log message
  - `chapter_number` (Integer): Optional chapter context
  - `step_name` (String): Optional step context

#### `SessionImage`
- **Purpose**: Track images generated during a session
- **Key Fields**:
  - `session_id` (UUID): Associated session
  - `illustration_id` (UUID): Generated illustration
  - `checkpoint_id` (UUID): Associated checkpoint
  - `generation_order` (Integer): Overall generation order
  - `web_url` (String): Web URL for display
  - `prompt_used` (Text): Prompt used for generation

## 🔄 Checkpoint Types

### Available Checkpoint Types:
1. `SESSION_START` - Session initialization
2. `MANUSCRIPT_LOADED` - Manuscript and chapters loaded
3. `CHAPTER_START` - Started processing a chapter
4. `CHAPTER_ANALYZED` - Completed chapter analysis
5. `PROMPTS_GENERATED` - Generated illustration prompts
6. `IMAGES_GENERATING` - Started image generation
7. `CHAPTER_COMPLETED` - Finished processing a chapter
8. `SESSION_COMPLETED` - Session completed successfully
9. `SESSION_PAUSED` - Session paused by user
10. `ERROR_OCCURRED` - Error encountered during processing

### Processing Steps:
1. `INITIALIZING` - Starting up
2. `LOADING_MANUSCRIPT` - Loading manuscript data
3. `ANALYZING_CHAPTERS` - Analyzing chapter content
4. `GENERATING_PROMPTS` - Creating image prompts
5. `GENERATING_IMAGES` - Generating images
6. `COMPLETING_SESSION` - Finalizing session

## 🔧 Service Classes

### `SessionPersistenceService`

**Purpose**: Handles database and file-based session storage

**Key Methods**:
- `create_session()`: Create new processing session
- `update_session_status()`: Update session status
- `create_checkpoint()`: Create processing checkpoint
- `get_resumable_sessions()`: Get sessions that can be resumed
- `get_session_for_resume()`: Get complete session info for resuming
- `add_session_image()`: Track generated images
- `log_session_event()`: Log session events

**File Storage**:
- Sessions stored in `illustrator_output/sessions/active_sessions/`
- Checkpoints stored in `illustrator_output/sessions/checkpoints/`
- Recovery data in `illustrator_output/sessions/recovery/`

### `CheckpointManager`

**Purpose**: High-level checkpoint creation and resume functionality

**Key Methods**:
- `create_session_start_checkpoint()`: Session initialization
- `create_manuscript_loaded_checkpoint()`: After manuscript load
- `create_chapter_start_checkpoint()`: Chapter processing start
- `create_chapter_analyzed_checkpoint()`: After chapter analysis
- `create_prompts_generated_checkpoint()`: After prompt generation
- `create_images_generating_checkpoint()`: During image generation
- `create_chapter_completed_checkpoint()`: Chapter completion
- `create_session_completed_checkpoint()`: Session completion
- `create_pause_checkpoint()`: User-requested pause
- `create_error_checkpoint()`: Error handling
- `get_resume_info()`: Get information for resuming

## 🚀 Enhanced Workflow Integration

### Modified `run_processing_workflow`

The main processing workflow now includes:

1. **Resume Detection**: Check if resuming from checkpoint
2. **Session Initialization**: Create or restore session state
3. **Checkpoint Creation**: Create checkpoints at each milestone:
   - Session start
   - Manuscript loaded
   - Chapter start (for each chapter)
   - Chapter analyzed (after analysis)
   - Prompts generated (after prompt creation)
   - Images generating (during generation)
   - Chapter completed (after each chapter)
   - Session completed (at end)

4. **Error Handling**: Create error checkpoints on failures
5. **Pause Support**: Create pause checkpoints when requested

### Resume Logic

When resuming:
1. Load resume information using `checkpoint_manager.get_resume_info()`
2. Restore session state from database and files
3. Determine resume point based on last checkpoint type
4. Skip completed chapters/steps
5. Continue from appropriate checkpoint

## 📡 API Endpoints

### New Endpoints:

#### `POST /api/process/resume/{session_id}`
- Resume processing from last checkpoint
- Returns session information and resume status

#### `GET /api/process/resumable`
- Get all sessions that can be resumed
- Returns list of resumable sessions with metadata

### Enhanced Endpoints:

#### `POST /api/process`
- Now supports resume detection
- Automatically reconnects to existing sessions

#### `POST /api/process/{session_id}/pause`
- Enhanced with checkpoint creation
- Creates pause checkpoint for resume capability

## 🗂️ File Structure

```
illustrator_output/
├── sessions/
│   ├── active_sessions/          # JSON session files
│   │   └── {session_id}.json
│   ├── checkpoints/              # Checkpoint files
│   │   └── {session_id}_{seq}.json
│   └── recovery/                 # Recovery data
├── generated_images/             # Generated images
└── exports/                      # Exported manuscripts
```

## 💾 Session File Format

```json
{
  "session_id": "uuid",
  "manuscript_id": "uuid",
  "external_session_id": "web_session_id",
  "status": "running",
  "progress_percent": 45,
  "current_chapter": 2,
  "total_chapters": 5,
  "style_config": {...},
  "max_emotional_moments": 10,
  "last_completed_step": "chapter_analyzed",
  "last_completed_chapter": 1,
  "processed_chapters": [1],
  "current_prompts": [...],
  "generated_images": [...],
  "emotional_moments": [...],
  "total_images_generated": 8,
  "started_at": "2023-01-01T00:00:00",
  "last_heartbeat": "2023-01-01T00:30:00"
}
```

## 🧪 Testing

The implementation includes comprehensive testing:

- **Unit Tests**: `test_persistence.py` - Tests core functionality
- **Import Tests**: Verifies all modules import correctly
- **File Operations**: Tests JSON persistence
- **Directory Structure**: Verifies required directories

## 🔄 Recovery Scenarios

### Server Restart Recovery:
1. Server restarts
2. Client reconnects to session
3. System loads session from JSON files
4. Processing resumes from last checkpoint

### Error Recovery:
1. Error occurs during processing
2. Error checkpoint created with full state
3. User can resume from error checkpoint
4. System retries from last successful step

### User Pause/Resume:
1. User requests pause
2. Pause checkpoint created
3. Processing stops gracefully
4. User can resume later from pause point

## 🏗️ Implementation Status

✅ **Database Models**: Complete with all necessary tables and relationships
✅ **Session Persistence Service**: Complete with DB and file storage
✅ **Checkpoint Manager**: Complete with all checkpoint types
✅ **Workflow Integration**: Complete with full checkpoint creation
✅ **Resume Logic**: Complete with intelligent resume detection
✅ **File-Based Storage**: Complete JSON persistence system
✅ **API Endpoints**: Complete with resume and status endpoints
✅ **Error Handling**: Complete with error checkpoints and recovery
✅ **Testing**: Complete with comprehensive test suite

## 📝 Usage Examples

### Starting a New Session:
```python
# Session is automatically created with checkpoints
POST /api/process
{
  "manuscript_id": "uuid",
  "style_config": {...},
  "max_emotional_moments": 10
}
```

### Resuming from Checkpoint:
```python
# Resume from last checkpoint
POST /api/process/resume/{session_id}

# Get resumable sessions
GET /api/process/resumable
```

### Monitoring Progress:
```python
# Check session status
GET /api/process/status/{manuscript_id}

# WebSocket connection for real-time updates
WS /ws/processing/{session_id}
```

The implementation provides a robust, fault-tolerant system that ensures no processing work is lost and users can seamlessly resume from any interruption point.