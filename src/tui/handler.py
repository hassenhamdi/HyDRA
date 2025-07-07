# src/tui/handler.py
import os
import uuid
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from src.core.reasoning_loop import ReasoningLoop
from src.utils.config_loader import ConfigLoader
from src.agents.memory_agent import HydraMemoryAgent

class TUIHandler:
    def __init__(self, user_id: str, profile: str):
        self.user_id = user_id
        self.session_id = str(uuid.uuid4())
        self.console = Console()
        self.memory_agent = HydraMemoryAgent()
        ConfigLoader.load(profile)
        self.print_welcome_message()

    def print_welcome_message(self):
        profile_name = ConfigLoader.load().get('profile_name', 'N/A')
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
  [cyan]/quit[/cyan] or [cyan]/exit[/cyan] - Exit the HyDRA TUI.
        """
        self.console.print(Panel(help_text, title="Help", border_style="yellow"))

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
                    self.console.print(Panel(f"[bold red]Error: {e}[/bold red]", border_style="red"))
            else:
                self.console.print(f"Current profile: [bold cyan]{ConfigLoader.load()['profile_name']}[/bold cyan]")
        elif cmd == "/pref":
            if arg:
                self.memory_agent.save_preference(self.user_id, arg)
                self.console.print(Panel(f"Preference saved for user '[cyan]{self.user_id}[/cyan]'.", border_style="green"))
            else:
                self.console.print(Panel("[bold red]Usage: /pref [your preference text][/bold red]", border_style="red"))
        elif cmd == "/new":
            self.session_id = str(uuid.uuid4())
            self.console.print(Panel(f"New session started: [cyan]{self.session_id}[/cyan]", border_style="green"))
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

                with self.console.status("[bold yellow]HyDRA is thinking...[/bold yellow]", spinner="dots") as status:
                    def update_tui_callback(message: str, category: str):
                        status.update(f"[bold yellow]HyDRA is thinking... ([italic]{category}[/italic])[/bold yellow]")
                        self.console.print(f"[dim cyan] -> {message}[/dim cyan]")

                    gemini_api_key = os.getenv("GEMINI_API_KEY")
                    hydra_loop = ReasoningLoop(gemini_api_key, self.user_id, self.session_id)
                    final_answer = hydra_loop.run(query, callback=update_tui_callback)

                self.console.print(Panel(
                    Markdown(final_answer), title="[bold green]HyDRA's Answer[/bold green]",
                    border_style="green", title_align="left"
                ))

            except (KeyboardInterrupt, EOFError): break
        
        self.console.print("\n[bold yellow]Exiting HyDRA. Goodbye![/bold yellow]")