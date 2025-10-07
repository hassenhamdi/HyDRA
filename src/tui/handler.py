# src/tui/handler.py
import os
import uuid
import yaml
import subprocess
import sys
import traceback
from datetime import datetime
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from src.core.reasoning_loop import ReasoningLoop
from src.utils.config_loader import get_config, ConfigLoader
from src.agents.memory_agent import HydraMemoryAgent
import warnings
warnings.filterwarnings("ignore")

class TUIHandler:
    def __init__(self, user_id: str, profile: str):
        self.user_id = user_id
        self.session_id = str(uuid.uuid4())
        self.console = Console() 
        self.memory_agent = HydraMemoryAgent()
        self.last_answer_content = ""
        self.last_answer_title = ""
        self.autosave_enabled = False
        self.autoingest_enabled = False
        self.reports_dir = "HyDRA/reports"
        self.logs_dir = "HyDRA/logs"
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        self.print_welcome_message()

    # ... (All helper methods like print_welcome_message, print_help, etc., remain the same)
    def print_welcome_message(self):
        profile_name = get_config().get('profile_name', 'N/A')
        welcome_panel = Panel(
            f"[bold cyan]Welcome to the HyDRA Agent TUI![/bold cyan]\n\n"
            f"Profile: [yellow]{profile_name}[/yellow] | User: [yellow]{self.user_id}[/yellow]\n"
            "Type your query to begin, or use `/help` to see available commands.",
            title="HyDRA - Hybrid Dynamic RAG Agent",
            border_style="green", expand=False
        )
        self.console.print(welcome_panel)

    def print_help(self):
        help_text = """
[bold]Available Commands:[/bold]
  [cyan]/profile [name][/cyan]   - View or switch the deployment profile.
  [cyan]/pref [preference][/cyan]  - Set a user preference for personalization.
  [cyan]/new[/cyan]             - Start a new chat session (clears history).
  [cyan]/save [filename.md][/cyan] - Save the last report. Uses generated title if no filename is given.
  [cyan]/ingest <filename.md>[/cyan]- Ingest a saved report into the RAG knowledge base.
  [cyan]/autosave [on|off][/cyan]  - Toggle autosaving of all generated reports.
  [cyan]/autoingest [on|off][/cyan] - Toggle auto-ingestion of saved reports. (Requires autosave)
  [cyan]/quit[/cyan] or [cyan]/exit[/cyan] - Exit the HyDRA TUI.
        """
        self.console.print(Panel(help_text, title="Help", border_style="yellow"))
        
    def _ingest_report(self, filename: str):
        if not filename.endswith(".md"):
            filename += ".md"
            
        file_path = os.path.join(self.reports_dir, filename)
        if not os.path.exists(file_path):
            self.console.print(Panel(f"[bold red]Error: Cannot ingest. File not found: {file_path}[/bold red]", border_style="red"))
            return

        profile = get_config()['profile_name']
        command = [
            "python", "-m", "data_processing.ingest",
            "--path", os.path.abspath(file_path),
            "--profile", profile
        ]
        with self.console.status(f"[yellow]Ingesting {filename}...[/yellow]", spinner="dots"):
            result = subprocess.run(command, capture_output=True, text=True, cwd="HyDRA")
        
        if result.returncode == 0:
             self.console.print(Panel(f"Report [cyan]{filename}[/cyan] successfully ingested into the knowledge base.", border_style="green"))
        else:
             self.console.print(Panel(f"[bold red]Ingestion Failed for {filename}[/bold red]\n\n[cyan]Exit Code: {result.returncode}[/cyan]\n\n[bold]Output:[/bold]\n{result.stderr}", title="Ingestion Result", border_style="red"))

    def _save_report(self, filename: str) -> bool:
        if not self.last_answer_content:
            return False
        
        if not filename.endswith(".md"):
            filename += ".md"
            
        file_path = os.path.join(self.reports_dir, filename)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.last_answer_content)
            
            if not self.autosave_enabled:
                 self.console.print(Panel(f"Report successfully saved to:\n[cyan]{file_path}[/cyan]", border_style="green"))
            
            if self.autoingest_enabled:
                self._ingest_report(filename)
                
            return True
        except Exception as e:
            self.console.print(Panel(f"[bold red]Error saving report: {e}[/bold red]", border_style="red"))
            return False

    def handle_command(self, command: str):
        parts = command.strip().split(" ", 1)
        cmd, arg = (parts[0], parts[1]) if len(parts) > 1 else (parts[0], "")

        if cmd in ["/quit", "/exit"]: return False
        elif cmd == "/help": self.print_help()
        elif cmd == "/profile":
            if arg:
                try:
                    ConfigLoader.load(arg)
                    self.console.print(Panel(f"Switched to profile: [bold cyan]{arg}[/bold cyan]", border_style="green"))
                except ValueError as e:
                    with open("HyDRA/configs/deployment_profiles.yaml", "r") as f:
                        all_profiles = yaml.safe_load(f)
                        available = list(all_profiles.get("profiles", {}).keys())
                    error_message = f"[bold red]Error: {e}[/bold red]\n\nAvailable profiles are: [cyan]{', '.join(available)}[/cyan]"
                    self.console.print(Panel(error_message, border_style="red"))
            else:
                self.console.print(f"Current profile: [bold cyan]{get_config()['profile_name']}[/bold cyan]")
        elif cmd == "/pref":
            if arg:
                self.memory_agent.save_preference(self.user_id, self.session_id, arg)
                self.console.print(Panel(f"Preference saved for user '[cyan]{self.user_id}[/cyan]'.", border_style="green"))
            else:
                self.console.print(Panel("[bold red]Usage: /pref [your preference text][/bold red]", border_style="red"))
        elif cmd == "/new":
            self.session_id = str(uuid.uuid4())
            self.last_answer_content = ""
            self.last_answer_title = ""
            self.console.print(Panel(f"New session started: [cyan]{self.session_id}[/cyan]", border_style="green"))
        elif cmd == "/save":
            filename = arg if arg else self.last_answer_title
            if not filename:
                 self.console.print(Panel("[bold red]Error: No filename provided and no title was generated for the last report.[/bold red]", border_style="red"))
            else:
                self._save_report(filename)
        elif cmd == "/ingest":
            if arg:
                self._ingest_report(arg)
            else:
                self.console.print(Panel("[bold red]Usage: /ingest <filename.md>[/bold red]", border_style="red"))
        elif cmd == "/autosave":
            if arg.lower() == "on":
                self.autosave_enabled = True
                self.console.print(Panel("Autosave has been [bold green]enabled[/bold green].", border_style="green"))
            elif arg.lower() == "off":
                self.autosave_enabled = False
                self.console.print(Panel("Autosave has been [bold red]disabled[/bold red].", border_style="yellow"))
            else:
                status = "[bold green]enabled[/bold green]" if self.autosave_enabled else "[bold red]disabled[/bold red]"
                self.console.print(Panel(f"Autosave is currently {status}. Use '/autosave on' or '/autosave off' to change.", border_style="blue"))
        elif cmd == "/autoingest":
            if arg.lower() == "on":
                self.autoingest_enabled = True
                self.autosave_enabled = True
                self.console.print(Panel("Auto-ingest has been [bold green]enabled[/bold green]. Autosave was also enabled.", border_style="green"))
            elif arg.lower() == "off":
                self.autoingest_enabled = False
                self.console.print(Panel("Auto-ingest has been [bold red]disabled[/bold red].", border_style="yellow"))
            else:
                status = "[bold green]enabled[/bold green]" if self.autoingest_enabled else "[bold red]disabled[/bold red]"
                self.console.print(Panel(f"Auto-ingest is currently {status}. Use '/autoingest on' or '/autoingest off' to change.", border_style="blue"))
        else:
            self.console.print(Panel(f"[bold red]Unknown command: '{cmd}'.[/bold red]", border_style="red"))
        return True

    def start_chat(self):
        while True:
            try:
                query = self.console.input(f"[bold magenta]You: [/bold magenta]")
                if query.startswith("/"):
                    if not self.handle_command(query): break
                    continue

                final_answer_chunks = []
                report_title = "untitled_report"

                with self.console.status("[bold yellow]HyDRA is thinking...[/bold yellow]", spinner="dots") as status:
                    def update_tui_callback(message: str, category: str):
                        status.update(f"[bold yellow]HyDRA is thinking... ([italic]{category}[/italic])[/bold yellow]")

                    gemini_api_key = os.getenv("GEMINI_API_KEY")
                    hydra_loop = ReasoningLoop(gemini_api_key, self.user_id, self.session_id)
                    
                    # --- UNIFIED AND CORRECTED EXECUTION BLOCK ---
                    # The run method is a generator that yields content and returns the title
                    response_generator = hydra_loop.run(query, callback=update_tui_callback)
                    
                    # We display the final panel *after* the status spinner is complete
                    final_panel = Panel(Markdown(""), title="[bold green]HyDRA's Answer[/bold green]", border_style="green", title_align="left")
                    with Live(final_panel, console=self.console, vertical_overflow="visible", refresh_per_second=10) as live:
                        try:
                            # Iterate through the generator to get the content chunks
                            for chunk in response_generator:
                                final_answer_chunks.append(chunk)
                                # Create a new panel with the updated content for each frame
                                new_panel = Panel(
                                    Markdown("".join(final_answer_chunks)),
                                    title="[bold green]HyDRA's Answer[/bold green]",
                                    border_style="green",
                                    title_align="left"
                                )
                                live.update(new_panel)
                        except StopIteration as e:
                            # This is the correct way to get the return value from the generator
                            report_title = e.value if e.value else report_title
                
                self.last_answer_content = "".join(final_answer_chunks)
                self.last_answer_title = report_title
                
                if self.autosave_enabled:
                    self._save_report(self.last_answer_title)

            except (KeyboardInterrupt, EOFError):
                break
            except Exception:
                self.console.print_exception(show_locals=True)

        self.console.print("\n[bold yellow]Exiting HyDRA. Goodbye![/bold yellow]")