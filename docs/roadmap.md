# HyDRA Project Roadmap

This document outlines the planned features and improvements for the HyDRA project.

- [x] **Fully debugging and fix issue within project**. In its current state it is much like land full of mines (bugs 🐞🐞🐞).

- [ ] **Comprehensive Testing &amp; Benchmarking:** Rigorously evaluate HyDRA's performance on standard RAG benchmarks (e.g., GAIA, HotpotQA) to quantify its accuracy and efficiency.

- [ ] **Autonomous Knowledge Curation &amp; Temporal Intelligence:**
    - Develop a mechanism for agents to intelligently cache findings from web searches with timestamps in a dedicated 'Web Cache' within Milvus.
    - Implement a reasoning step for the Meta-Planner to check this "Web Cache" before initiating new external searches.
    - Teach the Meta-Planner to reason about data freshness based on query keywords (e.g., "latest," "today") and the timestamp of cached data to decide whether to use the cache or perform a new, time-bounded search (e.g., "search for news since yesterday").
    - This will allow HyDRA to learn new information from the web, avoid redundant work, and handle time-sensitive queries with greater accuracy and efficiency.

- [ ] **Full Multimodal Support:** Integrate vision and audio tools into dedicated `Executor` agents.

- [ ] **Hybrid Multimodal Search:** Integrate colpali model embedding raw pdf and images for seamless hybrid search across text, document and images.

- [ ] Suppress INFO and Warning logging displayed in the TUI.
