import os
import argparse
from dotenv import load_dotenv
from src.tui.handler import TUIHandler
from src.utils.config_loader import ConfigLoader
from rich.console import Console

def main():
    """
    Main entrypoint for the HyDRA RAG Agent application.
    Initializes the system and starts the interactive TUI chat handler.
    """
    # Load environment variables from a .env file
    load_dotenv()
    console = Console()

    # --- Argument Parsing ---
    # Sets up how users can run the application from the command line.
    parser = argparse.ArgumentParser(
        description="HyDRA: An Interactive, Hybrid, and Dynamic RAG Agent.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--profile", 
        type=str, 
        default=os.getenv("HYDRA_PROFILE", "development"), 
        help="The deployment profile to use (e.g., 'development', 'production_balanced').\nThis determines performance and accuracy trade-offs."
    )
    parser.add_argument(
        "--user_id", 
        type=str, 
        default="default_user", 
        help="A unique identifier for the user, used for personalization and memory."
    )
    args = parser.parse_args()

    # --- API Key Validation ---
    # Ensures the application can connect to the required LLM service.
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key or "YOUR_GOOGLE_API_KEY_HERE" in gemini_api_key:
        console.print("[bold red]ERROR: GEMINI_API_KEY is not set.[/bold red]")
        console.print("Please create a '.env' file, add your key (e.g., GEMINI_API_KEY=\"your-key\"), and try again.")
        return

    # --- Configuration Loading ---
    # This crucial step loads the selected deployment profile, which dictates
    # how all other modules (retrieval, indexing, etc.) will behave.
    try:
        ConfigLoader.load(args.profile)
    except ValueError as e:
        console.print(f"[bold red]ERROR: Invalid profile specified.[/bold red]")
        console.print(f"{e}")
        return
        
    # --- Application Launch ---
    # Initializes and starts the main Terminal User Interface handler.
    try:
        tui = TUIHandler(user_id=args.user_id, profile=args.profile)
        tui.start_chat()
    except Exception as e:
        console.print(f"\n[bold red]A critical error occurred during application runtime:[/bold red]")
        console.print(f"{e}")
        console.print("Please check your configuration and ensure all services (like Milvus) are running correctly.")

if __name__ == "__main__":
    main()