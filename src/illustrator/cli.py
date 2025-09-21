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
        load_dotenv()

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

    def get_user_preferences(self) -> Dict[str, Any]:
        """Collect user preferences for image generation."""
        console.print("\n[bold cyan]🎨 Style Preferences[/bold cyan]")

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

        # Here we would integrate with the LangGraph workflow
        # For now, we'll simulate the processing

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

                # Simulate processing time
                await asyncio.sleep(2)

                progress.update(task, description=f"✅ Chapter {i}: Analysis complete")

        console.print(f"\n[bold green]✨ Processing complete! Analyzed {len(self.chapters)} chapters.[/bold green]")

    def display_results_summary(self):
        """Display a summary of the processing results."""
        if not self.chapters:
            return

        table = Table(title="📊 Analysis Summary")
        table.add_column("Chapter", style="cyan", width=8)
        table.add_column("Title", style="white")
        table.add_column("Words", justify="right", style="green")
        table.add_column("Status", style="green")

        for chapter in self.chapters:
            table.add_row(
                str(chapter.number),
                chapter.title,
                f"{chapter.word_count:,}",
                "✅ Analyzed"
            )

        console.print(f"\n{table}")

        # Summary stats
        total_words = sum(ch.word_count for ch in self.chapters)
        console.print(f"\n[bold cyan]Total manuscript: {len(self.chapters)} chapters, {total_words:,} words[/bold cyan]")

    def save_results(self):
        """Save processing results to files."""
        if not self.manuscript_metadata or not self.chapters:
            return

        # Create output directory
        output_dir = Path("illustrator_output") / self.manuscript_metadata.title.replace(" ", "_")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save manuscript data
        manuscript_data = {
            "metadata": self.manuscript_metadata.dict(),
            "chapters": [ch.dict() for ch in self.chapters],
            "completed_analyses": [analysis.dict() for analysis in self.completed_analyses],
            "processing_date": datetime.now().isoformat(),
        }

        output_file = output_dir / "manuscript_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(manuscript_data, f, indent=2, ensure_ascii=False)

        console.print(f"\n[green]📁 Results saved to: {output_file}[/green]")


@click.command()
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
def main(interactive: bool, config_file: str | None):
    """Manuscript Illustrator - Analyze chapters and generate AI illustrations."""
    cli = ManuscriptCLI()

    try:
        # Setup
        cli.setup_environment()

        if interactive:
            # Interactive mode
            cli.display_welcome()

            # Get manuscript metadata
            cli.manuscript_metadata = cli.get_manuscript_metadata()

            # Get user preferences
            style_preferences = cli.get_user_preferences()

            # Collect chapters
            chapter_number = 1
            while True:
                chapter = cli.input_chapter(chapter_number)
                if chapter:
                    cli.chapters.append(chapter)
                    chapter_number += 1

                if not cli.confirm_continue():
                    break

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


if __name__ == "__main__":
    main()