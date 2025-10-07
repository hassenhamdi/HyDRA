# HyDRA Changelog - v0.2.0

This release marks a fundamental architectural evolution for HyDRA, moving from a simple agent pipeline to a sophisticated, iterative reasoning and learning system. The introduction of the **HELP/SIMPSON** learning loop and the upgrade of executor agents to a **ReAct-style** operational flow are the cornerstones of this update.

## 💥 Breaking Changes

*   **Core Reasoning Loop Rearchitecture:** The `ReasoningLoop` has been completely redesigned. It no longer executes a static, pre-generated plan. Instead, it engages in an iterative cycle where the `MetaPlanner` generates one step at a time, and the results of each step are fed back into the prompt. This enables more dynamic and complex problem-solving.
*   **Planner Prompt & Output:** The `MetaPlanner` agent no longer outputs a simple JSON list of tasks. It now uses a more complex prompt and outputs reasoning steps interspersed with special tokens (`<|begin_call_subtask|>`) to invoke tools, reflecting the new iterative nature.
*   **Synthesis Agent Output:** The `SynthesisAgent` now returns a structured JSON object containing `report_title` and `report_content`, not a raw string. This required updates to the `ReasoningLoop` and `TUIHandler` to parse the new format.

## 🚀 New Features

*   **HELP/SIMPSON Autonomous Learning System:**
    *   **`MemoryAgent`**: A new agent responsible for all interactions with a persistent Milvus memory store. It handles saving and retrieving strategic guidance, user preferences, and conversation summaries.
    *   **`PostInteractionAnalyzer`**: A new agent that runs after each session to analyze the full transcript, infer implicit user preferences (e.g., "prefers concise summaries"), and store them for future personalization.
    *   **Policy Reflection**: The `AdaptiveCoordinator` now performs a "policy reflection" step after each delegated task. It uses an LLM to analyze the outcome and generate a concise, learned heuristic (e.g., "Broad search queries are ineffective") which is saved by the `MemoryAgent`. This allows the system to improve its delegation strategy over time.

*   **Iterative Executor Agents (ReAct-style):**
    *   **`DeepSearchAgent` Overhaul**:
        *   Now operates as an iterative agent that can perform a series of actions (`SEARCH`, `FETCH`, `FINISH`).
        -   Integrates with **`langchain-mcp-adapters`** to fetch the full content of web pages, a massive improvement over relying solely on search snippets.
        -   Maintains an internal history of its actions and observations to inform its next step.
        -   Returns a full `action_trace` for the HELP/SIMPSON learning loop.
    *   **`AdvancedVectorSearchAgent` Overhaul**:
        *   Upgraded to an iterative agent that can perform multiple refined searches (`QUERY`, `HYDE_QUERY`) against the internal knowledge base.
        *   This enables it to handle complex questions that require synthesizing information from multiple documents.

*   **Enhanced Terminal User Interface (TUI):**
    *   **Streaming Responses**: The final answer is now streamed to the console using `rich.live`, providing a real-time, ChatGPT-like experience.
    *   **Knowledge Curation Commands**: Added new slash commands for managing the knowledge base:
        *   `/save [filename]`: Saves the last generated report to the `reports/` directory.
        *   `/ingest <filename>`: Ingests a saved report back into the Milvus knowledge base.
        *   `/autosave [on|off]`: Toggles automatic saving of all generated reports.
        *   `/autoingest [on|off]`: Toggles automatic ingestion of newly saved reports.

## 🔧 Refactors & Improvements

*   **Centralized Model Management (`ModelRegistry`):**
    *   A new `ModelRegistry` service was introduced to load and manage the BGE embedding and reranker models as singletons. This prevents them from being loaded into memory multiple times, significantly reducing startup time and memory footprint.
*   **Dynamic Milvus Configuration:**
    *   The `milvus_setup.py` script now dynamically sets the dense vector data type to `FLOAT16_VECTOR` or `FLOAT_VECTOR` based on the `use_fp16` setting in the active deployment profile.
    *   The setup script now correctly drops existing collections to ensure a clean state on re-initialization.
*   **Robust Configuration Loading:**
    *   The `ConfigLoader` utility has been improved with a more robust pattern. The configuration is now explicitly loaded once at application startup, and subsequent calls to `get_config()` retrieve the cached version.
*   **Main Application Entrypoint:**
    *   `main.py` now includes a more robust, system-level crash handler to catch and report critical errors that might terminate the TUI.

## 📝 Documentation

*   **`README.md` Overhaul**: The main `README` has been significantly expanded with:
    *   A "Why HyDRA?" section explaining the project's philosophy and its synthesis of concepts from papers like HiRA, HM-RAG, and HyDE.
    *   A detailed technical explanation of the **HELP/SIMPSON** learning loop.
    *   An improved architectural diagram reflecting the new learning loop.
*   **`roadmap.md`**: A new file has been added to formally track planned features and the future direction of the project.
*   **Sample Data**: Added `data/history_of_ai.md` as a sample document for first-time users to ingest.

## 🛠️ Build & Dependencies

*   Added new dependencies to `requirements.txt`:
    *   `asyncddgs`, `nest-asyncio`: For the improved, asynchronous web search capabilities.
    *   `langchain-mcp-adapters`: To support fetching full web page content.
    *   `grpcio-tools`, `langchain-community`: General supporting libraries.
*   Removed `duckduckgo-search` in favor of the more powerful `asyncddgs`.

## 🧪 Testing & Tooling

*   **`hybrid search test.ipynb`**: Added a new Jupyter Notebook to provide a clear, executable example of the end-to-end hybrid retrieval pipeline, from setup and ingestion to querying.