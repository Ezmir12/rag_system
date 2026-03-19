# ⛓️ Agent Logic & Chains
The "Brain" of the system, constructed using LangChain Expression Language (LCEL).

### Key Features:
- **History-Aware Rewriter:** Before searching, the AI looks at the `chat_history` to understand pronouns (e.g., "What does *it* say?").
- **Streaming Logic:** Built to support word-by-word streaming for a smooth user experience.
- **Contextual QA:** A specialized prompt that forces the AI to answer *only* based on the retrieved documents.

### Why this matters:
This module transforms a simple search engine into a conversational assistant. It allows for natural, multi-turn dialogues where the AI remembers what was said 5 minutes ago.