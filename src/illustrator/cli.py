"""Command line interface for the Manuscript Illustrator."""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from illustrator.context import ManuscriptContext
from illustrator.models import (
    Chapter,
    ChapterAnalysis,
    ManuscriptMetadata,
    SavedManuscript,
)

console = Console()


class ManuscriptCLI:
    """Command line interface for manuscript processing."""

    def __init__(self):
        """Initialize the CLI interface."""
        self.context: ManuscriptContext | None = None
        self.chapters: List[Chapter] = []
        self.manuscript_metadata: ManuscriptMetadata | None = None
        self.completed_analyses: List[ChapterAnalysis] = []

    def setup_environment(self):
        """Load environment variables and validate configuration."""
        load_dotenv('.env')

        # Validate required API keys based on selected provider
        image_provider = os.getenv('DEFAULT_IMAGE_PROVIDER', 'dalle')

        if image_provider == 'dalle' and not os.getenv('OPENAI_API_KEY'):
            console.print("[red]Error: OPENAI_API_KEY required for DALL-E[/red]")
            sys.exit(1)
        elif image_provider == 'imagen4':
            if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS') or not os.getenv('GOOGLE_PROJECT_ID'):
                console.print("[red]Error: Google Cloud credentials required for Imagen4[/red]")
                sys.exit(1)
        elif image_provider == 'flux' and not os.getenv('HUGGINGFACE_API_KEY'):
            console.print("[red]Error: HUGGINGFACE_API_KEY required for Flux[/red]")
            sys.exit(1)

        if not os.getenv('ANTHROPIC_API_KEY'):
            console.print("[red]Error: ANTHROPIC_API_KEY required for Claude analysis[/red]")
            sys.exit(1)

    def display_welcome(self):
        """Display welcome message and application information."""
        welcome_text = Text()
        welcome_text.append("📚 ", style="bold blue")
        welcome_text.append("Manuscript Illustrator", style="bold blue")
        welcome_text.append(" ✨", style="bold yellow")

        welcome_panel = Panel(
            Text.assemble(
                "\nAnalyze your manuscript chapters and generate AI illustrations\n\n",
                ("• Enter chapter content with CTRL+D when finished\n", "green"),
                ("• Choose from DALL-E, Imagen4, or Flux for image generation\n", "green"),
                ("• Get emotional analysis and optimal illustration prompts\n", "green"),
            ),
            title=welcome_text,
            border_style="blue",
            padding=(1, 2),
        )

        console.print(welcome_panel)

    def get_manuscript_metadata(self) -> ManuscriptMetadata:
        """Collect metadata about the manuscript."""
        console.print("\n[bold cyan]📖 Manuscript Information[/bold cyan]")

        title = Prompt.ask("[green]Manuscript title")
        author = Prompt.ask("[green]Author name", default="")
        genre = Prompt.ask("[green]Genre", default="")

        return ManuscriptMetadata(
            title=title,
            author=author,
            genre=genre,
            total_chapters=0,  # Will update as chapters are added
            created_at=datetime.now().isoformat(),
        )

    def load_style_config(self, config_path: str) -> Dict[str, Any]:
        """Load style configuration from JSON file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Convert config to expected format
            style_prefs = {
                "image_provider": "dalle",  # default
                "art_style": config.get("style_name", "digital painting"),
                "color_palette": None,
                "artistic_influences": None,
                "style_config": config  # Store full config for later use
            }

            console.print(f"[green]✅ Loaded style config: {config.get('style_name', 'Unknown')}[/green]")
            return style_prefs

        except Exception as e:
            console.print(f"[red]❌ Error loading style config: {e}[/red]")
            raise

    def get_user_preferences(self, style_config_path: str | None = None) -> Dict[str, Any]:
        """Collect user preferences for image generation."""
        if style_config_path:
            return self.load_style_config(style_config_path)

        console.print("\n[bold cyan]🎨 Style Preferences[/bold cyan]")

        # Check for predefined style configurations
        predefined_styles = []
        if Path("advanced_eh_shepard_config.json").exists():
            predefined_styles.append(("advanced_eh_shepard", "E.H. Shepard Enhanced Pencil Illustration", "advanced_eh_shepard_config.json"))
        if Path("eh_shepard_pencil_config.json").exists():
            predefined_styles.append(("eh_shepard", "E.H. Shepard Pencil Sketch", "eh_shepard_pencil_config.json"))

        if predefined_styles:
            console.print("\n[green]Available predefined styles:[/green]")
            console.print("  0. Custom style (configure manually)")
            for i, (key, name, _) in enumerate(predefined_styles, 1):
                console.print(f"  {i}. {name}")

            while True:
                try:
                    style_choice = int(Prompt.ask("\nSelect style option", default="0"))
                    if style_choice == 0:
                        break  # Continue with manual configuration
                    elif 1 <= style_choice <= len(predefined_styles):
                        _, _, config_file = predefined_styles[style_choice - 1]
                        return self.load_style_config(config_file)
                    else:
                        console.print(f"[red]Invalid choice. Please select 0-{len(predefined_styles)}.[/red]")
                except ValueError:
                    console.print("[red]Please enter a number.[/red]")

        # Image provider selection
        providers = [
            ("dalle", "DALL-E 3 (OpenAI)"),
            ("imagen4", "Imagen4 (Google)"),
            ("flux", "Flux 1.1 Pro (HuggingFace)")
        ]

        console.print("\n[green]Available image providers:[/green]")
        for i, (key, name) in enumerate(providers, 1):
            console.print(f"  {i}. {name}")

        while True:
            try:
                choice = int(Prompt.ask("\nSelect image provider", default="1"))
                if 1 <= choice <= len(providers):
                    selected_provider = providers[choice - 1][0]
                    break
                else:
                    console.print("[red]Invalid choice. Please select 1-3.[/red]")
            except ValueError:
                console.print("[red]Please enter a number.[/red]")

        # Art style preferences
        art_style = Prompt.ask(
            "[green]Preferred art style[/green]",
            default="digital painting"
        )

        color_palette = Prompt.ask(
            "[green]Color palette preference[/green]",
            default="",
            show_default=False
        )

        artistic_influences = Prompt.ask(
            "[green]Artistic influences (e.g., 'Monet', 'Studio Ghibli')[/green]",
            default="",
            show_default=False
        )

        return {
            "image_provider": selected_provider,
            "art_style": art_style,
            "color_palette": color_palette or None,
            "artistic_influences": artistic_influences or None,
        }

    def input_chapter(self, chapter_number: int) -> Chapter | None:
        """Get chapter input from user with CTRL+D handling."""
        console.print(f"\n[bold yellow]📝 Chapter {chapter_number}[/bold yellow]")

        title = Prompt.ask(f"[green]Chapter {chapter_number} title")

        console.print("\n[green]Enter chapter content (press CTRL+D when finished):[/green]")
        console.print("[dim]Tip: You can paste large amounts of text. Press CTRL+D on a new line when done.[/dim]\n")

        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            # CTRL+D was pressed
            pass

        content = '\n'.join(lines).strip()

        if not content:
            console.print("[yellow]No content entered. Skipping chapter.[/yellow]")
            return None

        word_count = len(content.split())

        console.print(f"\n[green]✅ Chapter captured: {word_count:,} words[/green]")

        return Chapter(
            title=title,
            content=content,
            number=chapter_number,
            word_count=word_count,
        )

    def confirm_continue(self) -> bool:
        """Ask user if they want to add another chapter."""
        return Confirm.ask("\n[yellow]Add another chapter?[/yellow]", default=True)

    async def process_chapters(self, style_preferences: Dict[str, Any]):
        """Process all chapters through the LangGraph workflow."""
        if not self.chapters:
            console.print("[red]No chapters to process![/red]")
            return

        console.print(f"\n[bold cyan]🔄 Processing {len(self.chapters)} chapters...[/bold cyan]")

        # Import necessary components
        from illustrator.graph import graph
        from illustrator.context import ManuscriptContext
        from illustrator.state import ManuscriptState
        from illustrator.models import ImageProvider
        from langgraph.store.memory import InMemoryStore
        import os
        import uuid

        # Create runtime context
        context = ManuscriptContext(
            user_id=str(uuid.uuid4()),
            image_provider=ImageProvider(style_preferences["image_provider"]),
            default_art_style=style_preferences.get("art_style", "digital painting"),
            color_palette=style_preferences.get("color_palette"),
            artistic_influences=style_preferences.get("artistic_influences"),
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
            google_credentials=os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
            huggingface_api_key=os.getenv('HUGGINGFACE_API_KEY'),
        )

        # Create store and compile graph
        store = InMemoryStore()
        compiled_graph = graph.compile(store=store)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            for i, chapter in enumerate(self.chapters, 1):
                task = progress.add_task(
                    f"Analyzing Chapter {i}: {chapter.title}",
                    total=None
                )

                try:
                    # Create initial state for this chapter
                    initial_state = {
                        "messages": [],
                        "manuscript_metadata": self.manuscript_metadata,
                        "current_chapter": chapter,
                        "chapters_completed": self.completed_analyses,
                        "awaiting_chapter_input": False,
                        "processing_complete": False,
                        "illustrations_generated": False,
                        "image_provider": context.image_provider,
                        "style_preferences": style_preferences,
                        "analysis_depth": "detailed",
                        "current_analysis": None,
                        "generated_images": [],
                        "error_message": None,
                        "retry_count": 0,
                    }

                    # Process the chapter through the graph
                    result = await compiled_graph.ainvoke(
                        input=initial_state,
                        config={
                            "thread_id": f"chapter_{i}",
                            "configurable": {
                                "context": context
                            }
                        }
                    )

                    # Extract the analysis from the result
                    if result.get("current_analysis"):
                        analysis = result["current_analysis"]
                        self.completed_analyses.append(analysis)

                        # Save generated images if they exist
                        if result.get("generated_images"):
                            await self._save_generated_images(result["generated_images"], chapter, i)

                        generated_count = len(result.get("generated_images", []))
                        progress.update(task, description=f"✅ Chapter {i}: Analysis complete, {generated_count} images generated")
                    elif result.get("error_message"):
                        console.print(f"\n[yellow]⚠️ Chapter {i} analysis had issues: {result['error_message']}[/yellow]")
                        progress.update(task, description=f"⚠️ Chapter {i}: Completed with warnings")
                    else:
                        progress.update(task, description=f"✅ Chapter {i}: Processing complete")

                except Exception as e:
                    console.print(f"\n[red]❌ Error processing Chapter {i}: {e}[/red]")
                    progress.update(task, description=f"❌ Chapter {i}: Error occurred")

        console.print(f"\n[bold green]✨ Processing complete! Analyzed {len(self.chapters)} chapters.[/bold green]")

    async def _save_generated_images(self, generated_images: List[Dict[str, Any]], chapter: Chapter, chapter_num: int):
        """Save generated images to files."""
        if not generated_images or not self.manuscript_metadata:
            return

        # Create images directory
        output_dir = Path("illustrator_output") / self.manuscript_metadata.title.replace(" ", "_")
        images_dir = output_dir / "generated_images" / f"chapter_{chapter_num:02d}"
        images_dir.mkdir(parents=True, exist_ok=True)

        for i, img_data in enumerate(generated_images, 1):
            try:
                # Extract image data
                if 'image_data' in img_data:
                    image_content = img_data['image_data']

                    # Handle different image data formats
                    if isinstance(image_content, str):
                        # Base64 encoded image
                        import base64
                        if image_content.startswith('data:image/'):
                            # Remove data URL prefix
                            image_content = image_content.split(',')[1]
                        image_bytes = base64.b64decode(image_content)
                    elif isinstance(image_content, bytes):
                        # Raw image bytes
                        image_bytes = image_content
                    else:
                        console.print(f"[yellow]Warning: Unsupported image format for Chapter {chapter_num}, Image {i}[/yellow]")
                        continue

                    # Determine file extension (default to PNG)
                    file_ext = "png"
                    if 'metadata' in img_data and 'format' in img_data['metadata']:
                        file_ext = img_data['metadata']['format'].lower()

                    # Create filename
                    safe_title = "".join(c for c in chapter.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    filename = f"ch{chapter_num:02d}_img{i:02d}_{safe_title[:20]}.{file_ext}"

                    # Save image file
                    image_path = images_dir / filename
                    with open(image_path, 'wb') as f:
                        f.write(image_bytes)

                    console.print(f"[green]📷 Saved image: {image_path}[/green]")

                    # Save prompt/metadata file
                    metadata_file = images_dir / f"{filename}.txt"
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        f.write(f"Chapter: {chapter.title}\n")
                        f.write(f"Emotional Moment: {img_data.get('emotional_moment', 'N/A')}\n")
                        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
                        if 'metadata' in img_data:
                            f.write("Generation Metadata:\n")
                            for key, value in img_data['metadata'].items():
                                f.write(f"  {key}: {value}\n")

            except Exception as e:
                console.print(f"[red]❌ Error saving image {i} for Chapter {chapter_num}: {e}[/red]")

    def display_results_summary(self):
        """Display a summary of the processing results."""
        if not self.chapters:
            return

        table = Table(title="📊 Analysis Summary")
        table.add_column("Chapter", style="cyan", width=8)
        table.add_column("Title", style="white")
        table.add_column("Words", justify="right", style="green")
        table.add_column("Moments", justify="right", style="yellow")
        table.add_column("Prompts", justify="right", style="blue")
        table.add_column("Status", style="green")

        # Create a lookup for analyses by chapter number
        analysis_lookup = {}
        for analysis in self.completed_analyses:
            if hasattr(analysis, 'chapter'):
                analysis_lookup[analysis.chapter.number] = analysis
            elif isinstance(analysis, dict) and 'chapter' in analysis:
                analysis_lookup[analysis['chapter']['number']] = analysis

        for chapter in self.chapters:
            analysis = analysis_lookup.get(chapter.number)

            if analysis:
                if hasattr(analysis, 'emotional_moments'):
                    moments_count = len(analysis.emotional_moments)
                    prompts_count = len(analysis.illustration_prompts)
                elif isinstance(analysis, dict):
                    moments_count = len(analysis.get('emotional_moments', []))
                    prompts_count = len(analysis.get('illustration_prompts', []))
                else:
                    moments_count = 0
                    prompts_count = 0

                status = "✅ Analyzed"
            else:
                moments_count = 0
                prompts_count = 0
                status = "❌ Failed"

            table.add_row(
                str(chapter.number),
                chapter.title,
                f"{chapter.word_count:,}",
                str(moments_count),
                str(prompts_count),
                status
            )

        console.print(f"\n{table}")

        # Summary stats
        total_words = sum(ch.word_count for ch in self.chapters)
        total_moments = sum(len(analysis.emotional_moments) if hasattr(analysis, 'emotional_moments')
                          else len(analysis.get('emotional_moments', [])) if isinstance(analysis, dict)
                          else 0 for analysis in self.completed_analyses)
        total_prompts = sum(len(analysis.illustration_prompts) if hasattr(analysis, 'illustration_prompts')
                          else len(analysis.get('illustration_prompts', [])) if isinstance(analysis, dict)
                          else 0 for analysis in self.completed_analyses)

        # Count generated images (check if output directory exists)
        total_images = 0
        if self.manuscript_metadata:
            output_dir = Path("illustrator_output") / self.manuscript_metadata.title.replace(" ", "_")
            images_dir = output_dir / "generated_images"
            if images_dir.exists():
                # Count image files (not txt files)
                image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
                for chapter_dir in images_dir.iterdir():
                    if chapter_dir.is_dir():
                        total_images += len([f for f in chapter_dir.iterdir()
                                           if f.suffix.lower() in image_extensions])

        console.print(f"\n[bold cyan]Total manuscript: {len(self.chapters)} chapters, {total_words:,} words[/bold cyan]")
        console.print(f"[bold cyan]Generated: {total_moments} emotional moments, {total_prompts} illustration prompts, {total_images} images[/bold cyan]")

    def save_results(self):
        """Save processing results to files."""
        if not self.manuscript_metadata or not self.chapters:
            return

        # Create output directory
        output_dir = Path("illustrator_output") / self.manuscript_metadata.title.replace(" ", "_")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare analyses for serialization
        serialized_analyses = []
        for analysis in self.completed_analyses:
            if hasattr(analysis, 'dict'):
                # Pydantic model
                serialized_analyses.append(analysis.dict())
            elif hasattr(analysis, 'model_dump'):
                # Pydantic v2 model
                serialized_analyses.append(analysis.model_dump())
            elif isinstance(analysis, dict):
                # Already a dictionary
                serialized_analyses.append(analysis)
            else:
                # Convert to dict manually
                try:
                    serialized_analyses.append(analysis.__dict__)
                except:
                    # Skip this analysis if we can't serialize it
                    console.print(f"[yellow]Warning: Could not serialize one analysis result[/yellow]")

        # Save manuscript data
        manuscript_data = {
            "metadata": self.manuscript_metadata.dict() if hasattr(self.manuscript_metadata, 'dict') else self.manuscript_metadata.model_dump(),
            "chapters": [ch.dict() if hasattr(ch, 'dict') else ch.model_dump() for ch in self.chapters],
            "completed_analyses": serialized_analyses,
            "processing_date": datetime.now().isoformat(),
        }

        output_file = output_dir / "manuscript_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(manuscript_data, f, indent=2, ensure_ascii=False)

        console.print(f"\n[green]📁 Results saved to: {output_file}[/green]")

        # Also save individual prompt files for easy access
        if serialized_analyses:
            prompts_dir = output_dir / "illustration_prompts"
            prompts_dir.mkdir(exist_ok=True)

            for i, analysis in enumerate(serialized_analyses, 1):
                if 'illustration_prompts' in analysis and analysis['illustration_prompts']:
                    chapter_title = analysis.get('chapter', {}).get('title', f'Chapter_{i}')
                    safe_title = "".join(c for c in chapter_title if c.isalnum() or c in (' ', '-', '_')).rstrip()

                    prompt_file = prompts_dir / f"chapter_{i:02d}_{safe_title}.txt"
                    with open(prompt_file, 'w', encoding='utf-8') as f:
                        f.write(f"Chapter {i}: {chapter_title}\n")
                        f.write("="*50 + "\n\n")

                        for j, prompt in enumerate(analysis['illustration_prompts'], 1):
                            f.write(f"Illustration Prompt {j}:\n")
                            f.write("-"*25 + "\n")
                            f.write(f"{prompt}\n\n")

            console.print(f"[green]📝 Individual prompt files saved to: {prompts_dir}[/green]")

    def save_manuscript_draft(self, name: str | None = None) -> str:
        """Save the current manuscript draft to file."""
        if not self.manuscript_metadata or not self.chapters:
            raise ValueError("No manuscript data to save")

        # Create saved manuscripts directory
        saved_dir = Path("saved_manuscripts")
        saved_dir.mkdir(exist_ok=True)

        # Generate filename if not provided
        if not name:
            safe_title = "".join(c for c in self.manuscript_metadata.title if c.isalnum() or c in (' ', '-', '_')).strip()
            name = f"{safe_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        file_path = saved_dir / f"{name}.json"

        # Create saved manuscript object
        saved_manuscript = SavedManuscript(
            metadata=self.manuscript_metadata,
            chapters=self.chapters,
            saved_at=datetime.now().isoformat(),
            file_path=str(file_path)
        )

        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(saved_manuscript.model_dump(), f, indent=2, ensure_ascii=False)

        console.print(f"[green]💾 Manuscript draft saved: {file_path}[/green]")
        return str(file_path)

    def list_saved_manuscripts(self) -> List[SavedManuscript]:
        """List all saved manuscript drafts."""
        saved_dir = Path("saved_manuscripts")
        if not saved_dir.exists():
            return []

        manuscripts = []
        for file_path in saved_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                manuscript = SavedManuscript(**data)
                manuscripts.append(manuscript)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load {file_path}: {e}[/yellow]")

        # Sort by saved date (newest first)
        manuscripts.sort(key=lambda m: m.saved_at, reverse=True)
        return manuscripts

    def load_manuscript_draft(self, file_path: str) -> bool:
        """Load a saved manuscript draft."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            saved_manuscript = SavedManuscript(**data)

            # Load into current CLI instance
            self.manuscript_metadata = saved_manuscript.metadata
            self.chapters = saved_manuscript.chapters
            self.completed_analyses = []  # Reset analyses when loading draft

            console.print(f"[green]📖 Loaded manuscript: {self.manuscript_metadata.title}[/green]")
            console.print(f"[cyan]Chapters loaded: {len(self.chapters)}[/cyan]")
            return True

        except Exception as e:
            console.print(f"[red]❌ Error loading manuscript: {e}[/red]")
            return False

    def display_saved_manuscripts_menu(self) -> str | None:
        """Display menu of saved manuscripts and let user choose one."""
        manuscripts = self.list_saved_manuscripts()

        if not manuscripts:
            console.print("[yellow]No saved manuscripts found.[/yellow]")
            return None

        console.print("\n[bold cyan]📚 Saved Manuscripts[/bold cyan]")

        table = Table()
        table.add_column("#", width=3)
        table.add_column("Title", style="white")
        table.add_column("Chapters", justify="right", style="green")
        table.add_column("Saved", style="dim")

        for i, manuscript in enumerate(manuscripts, 1):
            # Format saved date
            try:
                saved_date = datetime.fromisoformat(manuscript.saved_at)
                date_str = saved_date.strftime("%Y-%m-%d %H:%M")
            except:
                date_str = manuscript.saved_at

            table.add_row(
                str(i),
                manuscript.metadata.title,
                str(len(manuscript.chapters)),
                date_str
            )

        console.print(table)

        while True:
            try:
                choice = Prompt.ask(f"\nSelect manuscript to load (1-{len(manuscripts)}, or 'q' to quit)")

                if choice.lower() == 'q':
                    return None

                choice_num = int(choice)
                if 1 <= choice_num <= len(manuscripts):
                    return manuscripts[choice_num - 1].file_path
                else:
                    console.print(f"[red]Invalid choice. Please select 1-{len(manuscripts)}.[/red]")

            except ValueError:
                console.print("[red]Please enter a number or 'q' to quit.[/red]")


@click.group()
def cli():
    """Manuscript Illustrator - Analyze chapters and generate AI illustrations."""
    pass


@cli.command()
@click.option(
    '--interactive/--batch',
    default=True,
    help='Run in interactive mode or batch mode'
)
@click.option(
    '--config-file',
    type=click.Path(exists=True),
    help='Path to configuration file'
)
@click.option(
    '--style-config',
    type=click.Path(exists=True),
    help='Path to JSON style configuration file (e.g., eh_shepard_pencil_config.json)'
)
@click.option(
    '--load',
    type=click.Path(exists=True),
    help='Load a previously saved manuscript draft'
)
@click.option(
    '--list-saved',
    is_flag=True,
    help='List all saved manuscript drafts'
)
def analyze(interactive: bool, config_file: str | None, style_config: str | None, load: str | None, list_saved: bool):
    """Run CLI analysis (original functionality)."""
    cli = ManuscriptCLI()

    try:
        # Setup
        cli.setup_environment()

        # Handle list-saved flag
        if list_saved:
            manuscripts = cli.list_saved_manuscripts()
            if manuscripts:
                console.print("\n[bold cyan]📚 Saved Manuscripts[/bold cyan]")
                for i, manuscript in enumerate(manuscripts, 1):
                    try:
                        saved_date = datetime.fromisoformat(manuscript.saved_at)
                        date_str = saved_date.strftime("%Y-%m-%d %H:%M")
                    except:
                        date_str = manuscript.saved_at

                    console.print(f"  {i}. {manuscript.metadata.title} ({len(manuscript.chapters)} chapters, saved {date_str})")
                    console.print(f"     File: {manuscript.file_path}")
            else:
                console.print("[yellow]No saved manuscripts found.[/yellow]")
            return

        # Handle load flag
        if load:
            if cli.load_manuscript_draft(load):
                console.print("[green]Manuscript loaded successfully. You can now process it or add more chapters.[/green]")
            else:
                sys.exit(1)

        if interactive:
            # Interactive mode
            cli.display_welcome()

            # Check if we already loaded a manuscript
            if not cli.manuscript_metadata:
                # Offer to load existing manuscript or create new
                console.print("\n[bold cyan]📖 Manuscript Options[/bold cyan]")
                console.print("  1. Create new manuscript")
                console.print("  2. Load saved manuscript")

                while True:
                    try:
                        choice = int(Prompt.ask("\nSelect option", default="1"))
                        if choice == 1:
                            # Create new manuscript
                            cli.manuscript_metadata = cli.get_manuscript_metadata()
                            break
                        elif choice == 2:
                            # Load existing manuscript
                            manuscript_path = cli.display_saved_manuscripts_menu()
                            if manuscript_path:
                                if cli.load_manuscript_draft(manuscript_path):
                                    break
                                else:
                                    continue
                            else:
                                # User cancelled, create new
                                cli.manuscript_metadata = cli.get_manuscript_metadata()
                                break
                        else:
                            console.print("[red]Invalid choice. Please select 1 or 2.[/red]")
                    except ValueError:
                        console.print("[red]Please enter a number.[/red]")

            # Get user preferences
            style_preferences = cli.get_user_preferences(style_config)

            # Collect chapters (continue from existing if loaded)
            if cli.chapters:
                console.print(f"\n[green]📚 Current manuscript has {len(cli.chapters)} chapters[/green]")
                add_more = Confirm.ask("Add more chapters?", default=False)
            else:
                add_more = True

            if add_more:
                chapter_number = len(cli.chapters) + 1
                while True:
                    chapter = cli.input_chapter(chapter_number)
                    if chapter:
                        cli.chapters.append(chapter)
                        chapter_number += 1

                    # Ask to save draft or continue
                    console.print("\n[cyan]Options:[/cyan]")
                    console.print("  1. Add another chapter")
                    console.print("  2. Save draft and continue adding")
                    console.print("  3. Save draft and process")
                    console.print("  4. Process without saving")

                    while True:
                        try:
                            option = int(Prompt.ask("Select option", default="1"))
                            if option == 1:
                                break  # Continue loop to add another chapter
                            elif option == 2:
                                try:
                                    cli.save_manuscript_draft()
                                    break  # Continue loop to add another chapter
                                except Exception as e:
                                    console.print(f"[red]❌ Error saving draft: {e}[/red]")
                                    continue
                            elif option == 3:
                                try:
                                    cli.save_manuscript_draft()
                                except Exception as e:
                                    console.print(f"[red]❌ Error saving draft: {e}[/red]")
                                # Exit chapter collection loop
                                chapter_number = None
                                break
                            elif option == 4:
                                # Exit chapter collection loop
                                chapter_number = None
                                break
                            else:
                                console.print("[red]Invalid choice. Please select 1-4.[/red]")
                        except ValueError:
                            console.print("[red]Please enter a number.[/red]")

                    if chapter_number is None:
                        break  # Exit main chapter collection loop
            elif cli.chapters and Confirm.ask("Save current draft?", default=True):
                try:
                    cli.save_manuscript_draft()
                except Exception as e:
                    console.print(f"[red]❌ Error saving draft: {e}[/red]")

            if cli.chapters:
                # Update total chapters in metadata
                cli.manuscript_metadata.total_chapters = len(cli.chapters)

                # Process chapters
                asyncio.run(cli.process_chapters(style_preferences))

                # Display results
                cli.display_results_summary()

                # Save results
                if Confirm.ask("\n[yellow]Save results to file?[/yellow]", default=True):
                    cli.save_results()
            else:
                console.print("\n[yellow]No chapters were added. Exiting.[/yellow]")

        else:
            # Batch mode (could read from files, etc.)
            console.print("[yellow]Batch mode not yet implemented.[/yellow]")
            sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n\n[yellow]⚠️  Operation cancelled by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]❌ Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option(
    '--host',
    default='127.0.0.1',
    help='Host to bind to (default: 127.0.0.1)'
)
@click.option(
    '--port',
    default=8000,
    type=int,
    help='Port to bind to (default: 8000)'
)
@click.option(
    '--reload',
    is_flag=True,
    help='Enable auto-reload for development'
)
@click.option(
    '--open-browser',
    is_flag=True,
    default=True,
    help='Automatically open browser (default: True)'
)
def start(host: str, port: int, reload: bool, open_browser: bool):
    """Start the web interface."""
    try:
        import webbrowser
        import threading
        import time
        from illustrator.web.app import run_server

        console.print(Panel.fit(
            f"[bold blue]🚀 Starting Manuscript Illustrator Web Interface[/bold blue]\n\n"
            f"[green]• Server: http://{host}:{port}[/green]\n"
            f"[green]• Environment: {'Development' if reload else 'Production'}[/green]\n"
            f"[green]• Auto-reload: {'Enabled' if reload else 'Disabled'}[/green]",
            title="Web Server",
            border_style="blue"
        ))

        # Open browser in a separate thread after a short delay
        if open_browser:
            def open_browser_delayed():
                time.sleep(2)  # Give server time to start
                try:
                    webbrowser.open(f"http://{host}:{port}")
                    console.print(f"[green]🌐 Opened browser at http://{host}:{port}[/green]")
                except Exception:
                    console.print(f"[yellow]Could not open browser automatically. Please visit: http://{host}:{port}[/yellow]")

            threading.Thread(target=open_browser_delayed, daemon=True).start()

        # Start the server
        run_server(host=host, port=port, reload=reload)

    except ImportError:
        console.print("[red]❌ Web dependencies not installed. Please install with:[/red]")
        console.print("[yellow]pip install 'illustrator[web]'[/yellow]")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]🛑 Web server stopped.[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Failed to start web server: {e}[/red]")
        sys.exit(1)


# Set main command to the group
def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()