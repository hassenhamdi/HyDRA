---
# Jekyll Front Matter for GitHub Pages
layout: default
title: HyDRA - Hybrid Dynamic RAG Agents
---

<style>
  /* Professional & Clean Styling for a Research Project Page */
  body { 
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    line-height: 1.7; 
    color: #333;
    background-color: #ffffff;
  }
  .container { 
    max-width: 960px; 
    margin: auto; 
    padding: 25px; 
  }
  h1, h2, h3 { 
    border-bottom: 1px solid #eaecef; 
    padding-bottom: 0.4em; 
    margin-top: 2em;
    font-weight: 600;
  }
  h1 { font-size: 2.5em; text-align: center; border-bottom: none; }
  h2 { font-size: 1.75em; }
  h3 { font-size: 1.25em; border-bottom: none; }
  code { 
    background-color: #f6f8fa; 
    border: 1px solid #eaecef;
    border-radius: 6px; 
    padding: 0.2em 0.4em; 
    font-size: 85%; 
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  }
  pre { 
    background-color: #f6f8fa; 
    border: 1px solid #eaecef;
    border-radius: 6px; 
    padding: 16px; 
    overflow: auto; 
  }
  pre > code { border: none; padding: 0; }
  a { color: #0366d6; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .badges { margin: 1.5em 0; }
  .project-header { text-align: center; margin-bottom: 2em; }
  .project-header img.banner { max-width: 800px; margin-bottom: 1em; }
  .project-header p { font-size: 1.2em; color: #586069; }
  .toc {
    background-color: #f9f9f9;
    border: 1px solid #e1e4e8;
    border-radius: 8px;
    padding: 1em 1.5em;
    margin: 2em 0;
  }
  .toc ul { list-style-type: none; padding-left: 0; }
  .toc ul li { margin-bottom: 0.5em; }
  .video-container { text-align: center; margin: 2.5em 0; }
  .video-container a { display: inline-block; border: 1px solid #e1e4e8; border-radius: 8px; padding: 5px; transition: transform 0.2s ease-in-out; }
  .video-container a:hover { transform: scale(1.02); }
  .video-container img { max-width: 100%; display: block; border-radius: 6px; }
  details { background-color: #fafbfc; border: 1px solid #e1e4e8; border-radius: 6px; padding: 15px; margin: 1em 0; }
  summary { font-weight: 600; cursor: pointer; }
  .footer { text-align: center; margin-top: 3em; color: #6a737d; font-size: 0.9em; }
</style>

<div class="container">

<!-- HEADER: Title, Subtitle, Badges -->
<header class="project-header">
  <img src="https://github.com/user-attachments/assets/213dd4c2-d10b-41ab-b157-8b97a987305a" alt="HyDRA Framework Banner" class="banner">
  <h1>HyDRA: Hybrid Dynamic RAG Agents</h1>
  <p>A novel framework for transforming Retrieval-Augmented Generation into a dynamic, learning reasoning system.</p>
  <div class="badges">
    <a href="https://github.com/hassenhamdi/HyDRA/stargazers"><img src="https://img.shields.io/github/stars/hassenhamdi/HyDRA?style=for-the-badge" alt="GitHub Stars"></a>
    <a href="https://github.com/hassenhamdi/HyDRA"><img src="https://img.shields.io/github/v/release/hassenhamdi/HyDRA?style=for-the-badge" alt="Release"></a>
    <a href="https://github.com/hassenhamdi/HyDRA/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License: MIT"></a>
  </div>
</header>

<!-- TABLE OF CONTENTS -->
<nav class="toc">
  <h3>Table of Contents</h3>
  <ul>
    <li><a href="#abstract">1. Abstract</a></li>
    <li><a href="#introduction-and-motivation">2. Introduction & Motivation</a></li>
    <li><a href="#core-methodology-the-hydra-triad">3. Core Methodology: The HyDRA Triad</a></li>
    <li><a href="#key-features-and-technical-specifications">4. Key Features & Technical Specifications</a></li>
    <li><a href="#getting-started-reproducibility">5. Getting Started & Reproducibility</a></li>
    <li><a href="#citation">6. Citation</a></li>
  </ul>
</nav>

<!-- DEMO VIDEO -->
<section class="video-container">
  <h2>Framework Demonstration</h2>
  <a href="https://github.com/user-attachments/assets/327a96a7-e45e-474c-9984-9d63032d5378
" target="_blank" rel="noopener noreferrer">
    <img src="https://github.com/user-attachments/assets/327a96a7-e45e-474c-9984-9d63032d5378
" alt="HyDRA Project Demo Video Thumbnail">
  </a>
</section>

<!-- ABSTRACT -->
<h2 id="abstract">1. Abstract</h2>
<p>
  Conventional Retrieval-Augmented Generation (RAG) systems operate as static, linear pipelines, exhibiting limitations in handling complex, multi-hop queries and adapting to new information. This work introduces HyDRA, a novel, three-tiered agentic framework designed to address these challenges by enabling dynamic, iterative reasoning. The framework is architected upon three synergistic pillars: a hierarchical agent structure for modular separation of concerns (strategy, coordination, execution); a dynamic reasoning loop inspired by the ReAct paradigm for adaptive, stateful problem-solving; and an autonomous learning system (HELP/SIMPSON) for continuous self-improvement based on past performance analysis. By synthesizing and extending seminal concepts from recent literature (HiRA, HM-RAG, HyDE), HyDRA creates a RAG system that is not merely powerful, but intelligent, adaptive, and architecturally robust.
</p>

<!-- INTRODUCTION & MOTIVATION -->
<h2 id="introduction-and-motivation">2. Introduction & Motivation</h2>
<p>
  The field of large language models is rapidly advancing, with RAG emerging as a critical technique for grounding responses in factual knowledge. However, the majority of existing RAG implementations remain "first-generation"—they retrieve and generate in a single, fixed pass. This paradigm falls short when queries require synthesizing information from multiple sources or adjusting a search strategy based on initial findings.
</p>
<p>
  HyDRA was conceived to bridge this gap, drawing inspiration from leading academic research to build a practical, integrated system. Our central hypothesis is that by equipping a RAG system with a structured agentic hierarchy and a mechanism for learning from experience, we can significantly enhance its reasoning capabilities and operational efficiency. HyDRA is our open-source contribution towards this next generation of intelligent RAG.
</p>

<!-- CORE METHODOLOGY -->
<h2 id="core-methodology-the-hydra-triad">3. Core Methodology: The HyDRA Triad</h2>
<p>
  HyDRA's architecture is founded on three core principles that enable it to reason, act, and learn with high proficiency.
</p>

<h3>3.1 Hierarchical Agentic Structure</h3>
<p>A clear separation of concerns ensures modularity and predictable behavior.</p>
<ul>
  <li><strong>Meta-Planner:</strong> The strategic reasoning layer. It analyzes the query and conversation history to determine the most salient next step in the problem-solving process.</li>
  <li><strong>Adaptive Coordinator:</strong> The operational management layer. It delegates tasks from the planner to the most suitable specialist agent, leveraging learned performance heuristics to optimize selection.</li>
  - <strong>Executors:</strong> The task execution layer. A pool of specialized agents with distinct tools, such as the <code>AdvancedVectorSearchAgent</code> for querying internal knowledge or the <code>DeepSearchAgent</code> for real-time web research.
</ul>

<h3>3.2 Dynamic Reasoning Loop (ReAct)</h3>
<p>HyDRA employs a dynamic <strong>Reasoning-Acting loop</strong>, eschewing static, pre-computed plans for a more flexible, iterative approach.</p>
<ol>
  <li>The <code>Meta-Planner</code> observes the current state and decides on the single best <strong>action</strong> to take.</li>
  <li>The <code>Coordinator</code> delegates this action to an <code>Executor</code>, which performs the task and returns an <strong>observation</strong>.</li>
  <li>This observation is appended to the system's history, informing a new cycle of reasoning. This loop enables HyDRA to tackle complex questions, recover from failed steps, and dynamically adjust its strategy.</li>
</ol>

<h3>3.3 Autonomous Learning (HELP/SIMPSON)</h3>
<p>The <strong>Heuristic Experience-based Learning Policy (HELP)</strong> system facilitates long-term memory and self-improvement. After each user interaction, a four-stage learning cycle is initiated:</p>
<ul>
  <li><strong>Observe & Critique:</strong> The system reviews the full conversation transcript, evaluating the efficiency and success of each step.</li>
  <li><strong>Memorize:</strong> It formulates and stores a concise, actionable "policy" in its Milvus memory (e.g., <em>"For queries about recent events, DeepSearchAgent is more effective than AdvancedVectorSearchAgent"</em>). It also learns the user's implicit preferences (e.g., formatting).</li>
  <li><strong>Adapt:</strong> In subsequent sessions, the <code>AdaptiveCoordinator</code> retrieves these learned policies as "strategic guidance," enabling it to make smarter, experience-based decisions.</li>
</ul>

<!-- FEATURES & SPECS -->
<h2 id="key-features-and-technical-specifications">4. Key Features & Technical Specifications</h2>
<ul>
  <li><strong>State-of-the-Art Retrieval Pipeline:</strong> Implements a three-stage retrieval process:
    <ol>
      <li><strong>Hybrid Search:</strong> Fuses dense (semantic) and sparse (lexical) vectors using BGE-M3 embeddings.</li>
      <li><strong>Reciprocal Rank Fusion (RRF):</strong> Merges search results from both modalities within Milvus.</li>
      <li><strong>Cross-Encoder Reranking:</strong> Utilizes a powerful BGE-Reranker model for final precision ranking.</li>
    </ol>
  </li>
  <li><strong>Iterative ReAct-style Agents:</strong> Enables dynamic, multi-step reasoning for complex problem-solving.</li>
  <li><strong>Continuous Self-Improvement:</strong> The HELP/SIMPSON learning loop optimizes agent delegation and planning over time.</li>
  <li><strong>Interactive TUI with Knowledge Management:</strong> A rich Terminal User Interface with streaming responses and commands (<code>/save</code>, <code>/ingest</code>) to curate the agent's knowledge base.</li>
  <li><strong>Configurable Deployment Profiles:</strong> Easily switch between profiles (e.g., <code>development</code>, <code>production_balanced</code>) to manage performance and resource trade-offs, including vector quantization.</li>
</ul>

<!-- GETTING STARTED -->
<h2 id="getting-started-reproducibility">5. Getting Started & Reproducibility</h2>
<p>To replicate our setup and run the framework, please follow these steps.</p>

<h3>5.1 Prerequisites</h3>
<ul>
  <li>Python 3.10+</li>
  <li>Docker and Docker Compose</li>
  <li>A Google Gemini API Key</li>
</ul>

<h3>5.2 System Setup</h3>
<details>
  <summary><strong>Click to expand setup instructions</strong></summary>
  <ol>
    <li>
      <p><strong>Install Milvus Standalone (Recommended Vector Database)</strong></p>
      <p>For Linux & macOS:</p>
      <pre><code>curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start</code></pre>
      <p>For Windows, please follow the <a href="https://milvus.io/docs/install_standalone-windows.md">official documentation</a> using Docker Desktop and WSL2.</p>
    </li>
    <li>
      <p><strong>Clone the HyDRA Repository & Install Dependencies</strong></p>
      <pre><code>git clone https://github.com/hassenhamdi/HyDRA.git
cd HyDRA
pip install -r requirements.txt</code></pre>
    </li>
    <li>
      <p><strong>Configure Environment Variables</strong></p>
      <pre><code># Create the .env file from the example
cp .env.example .env

# Now, edit the .env file and add your GEMINI_API_KEY</code></pre>
    </li>
    <li>
      <p><strong>Initialize Milvus Collections & Ingest Data</strong></p>
      <pre><code># 1. Create the Milvus collections as defined by the profile
python -m src.services.milvus_setup --profile development

# 2. Ingest your initial documents (sample provided)
python -m data_processing.ingest --path ./data --profile development</code></pre>
    </li>
  </ol>
</details>

<h3>5.3 Running the Application</h3>
<p>Launch the interactive Terminal User Interface:</p>
<pre><code>python main.py --profile development --user_id my_research_id</code></pre>

<!-- CITATION -->
<h2 id="citation">6. Citation</h2>
<p>If you use HyDRA in your research, please cite the project to credit this work. We appreciate your support.</p>
<pre><code>@software{hydra_agent_2025,
  author       = {Hassen Hamdi},
  title        = {HyDRA: Hybrid Dynamic RAG Agents},
  month        = {July},
  year         = {2025},
  publisher    = {GitHub},
  version      = {0.2.0},
  url          = {https://github.com/hassenhamdi/HyDRA}
}</code></pre>

---

<!-- FOOTER -->
<footer class="footer">
  <p>
    <a href="https://github.com/hassenhamdi/HyDRA">GitHub Repository</a> |
    <a href="https://github.com/hassenhamdi/HyDRA/blob/main/roadmap.md">Project Roadmap</a>
  </p>
  <p>Released under the MIT License.</p>
</footer>

</div>
