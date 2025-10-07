import os
import argparse
import sys
import traceback
from dotenv import load_dotenv
from src.tui.handler import TUIHandler
from src.utils.config_loader import ConfigLoader
from src.services.model_registry import ModelRegistry
from rich.console import Console
def main():
    """
    Main entrypoint for the HyDRA RAG Agent application.
    Initializes the system and starts the interactive TUI chat handler.
    """
    load_dotenv()
    # This console is primarily for startup messages and the final crash report.
    # The TUI handler will manage its own console instance.
    console = Console()

    parser = argparse.ArgumentParser(
        description="HyDRA: An Interactive, Hybrid, and Dynamic RAG Agent.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--profile", 
        type=str, 
        default=os.getenv("HYDRA_PROFILE", "development"), 
        help="The deployment profile to use (e.g., 'development', 'production_balanced')."
    )
    parser.add_argument(
        "--user_id", 
        type=str, 
        default="default_user", 
        help="A unique identifier for the user, used for personalization and memory."
    )
    args = parser.parse_args()

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key or "YOUR_GOOGLE_API_KEY_HERE" in gemini_api_key:
        console.print("[bold red]ERROR: GEMINI_API_KEY is not set.[/bold red]")
        console.print("Please create a '.env' file, add your key, and try again.")
        return

    try:
        # Load the configuration once at the start.
        ConfigLoader.load(args.profile)
        # Initialize the centralized models once.
        ModelRegistry.initialize_models()
    except ValueError as e:
        console.print(f"[bold red]ERROR: Invalid profile specified.[/bold red]")
        console.print(f"{e}")
        return
        
    try:
        # The TUI handler now contains all the main application logic.
        tui = TUIHandler(user_id=args.user_id, profile=args.profile)
        tui.start_chat()
    except Exception as e:
        # --- Robust, Final Crash Handler ---
        # This block is the last line of defense. It prints a detailed
        # traceback to the standard console, avoiding the rich console
        # which may be in an unstable state.
        print("\n" + "="*80, file=sys.stderr)
        print("A critical error occurred that terminated the TUI.", file=sys.stderr)
        print("Please review the traceback below for details.", file=sys.stderr)
        print("="*80 + "\n", file=sys.stderr)
        traceback.print_exc()
        # The TUI's own "on-error" logging should have already created a
        # detailed crash report file. This final print is for immediate
        # visibility in the terminal.

if __name__ == "__main__":
    main()
