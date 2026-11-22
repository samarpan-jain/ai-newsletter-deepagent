# Deep Agents AI Newsletter

The LangChain DeepAgent acts as the newsletter editor. It delegates each topic
to a research subagent, which uses the news-search tool. After all researchers
return, the editor calls a Markdown-generation tool that builds the newsletter
from their captured search results. Search results remain the source of truth
for headlines, descriptions, dates, and links.

The default topics are AI advancements and AI layoffs. Search results are limited to the
previous 30 days, and concurrent requests are rate-limited to respect the
provider's request limits.

## Setup

1. Install the project dependencies with `uv sync`.
2. Set `NEWS_API_KEY` in `.env` for GNews search.
3. Install and start the selected model with `ollama pull qwen3:8b`.
4. Run `uv run python deepagent.py`, or specify topics with
   `uv run python deepagent.py --topic "AI in healthcare" --topic "AI robotics"`.

The generated Markdown and HTML are written to `newsLetters/`. If the editor
does not run all of its research tasks and Markdown tool, generation fails
explicitly instead of producing an empty newsletter.
