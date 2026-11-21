import os
import re
from datetime import datetime, timedelta, timezone
from html import escape, unescape
import logging
from threading import Lock
from time import monotonic, sleep
from typing import Sequence

import markdown as md
import nh3
import requests
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field


_SEARCH_REQUEST_LOCK = Lock()
_LAST_SEARCH_STARTED = 0.0
_MIN_SEARCH_INTERVAL_SECONDS = 1.1
NEWS_LOOKBACK_DAYS = 30
SOURCES_SECTION = "Sources"
UNSUPPORTED_COPY_CLAIMS = re.compile(
    r"\b(?:surge|increas\w*|ris\w*|trend\w*|impact\w*|caus\w*|notable|"
    r"significant|major|growing)\b",
    flags=re.IGNORECASE,
)
logger = logging.getLogger(__name__)


class TopicRequest(BaseModel):
    topic: str = Field(min_length=1, description="Newsletter section name.")
    query: str = Field(min_length=1, description="Exact query to send to the news search tool.")


class NewsStory(BaseModel):
    headline: str = Field(description="The article headline, as supplied by search results.")
    summary: str = Field(description="The article description, as supplied by search results.")
    url: str = Field(min_length=1, description="The article URL supplied by search results.")
    published_date: str | None = Field(
        default=None,
        description="The publication date supplied by search results, if present.",
    )
    source_name: str | None = Field(
        default=None,
        description="The publisher name supplied by search results, if present.",
    )


class TopicResearch(BaseModel):
    stories: list[NewsStory]
    note: str | None = None


def parse_search_results(result: str) -> TopicResearch:
    if result.startswith("Error:"):
        return TopicResearch(stories=[], note=result)

    stories: list[NewsStory] = []
    seen_urls: set[str] = set()
    for article in result.split("\n\n"):
        fields: dict[str, str] = {}
        for line in article.splitlines():
            key, separator, value = line.partition(": ")
            if separator:
                fields[key.casefold()] = value.strip()

        headline = fields.get("title")
        summary = fields.get("description")
        url = fields.get("url", "")
        if (
            not headline
            or not summary
            or not url.startswith(("http://", "https://"))
            or url in seen_urls
        ):
            continue

        seen_urls.add(url)
        stories.append(
            NewsStory(
                headline=headline,
                summary=summary,
                url=url,
                published_date=fields.get("published"),
                source_name=fields.get("source"),
            )
        )

    if stories:
        return TopicResearch(stories=stories)
    if result.strip():
        return TopicResearch(stories=[], note=result.strip())
    return TopicResearch(stories=[], note="No recent articles were returned for this topic.")


def _rate_limited_get(url: str, **kwargs) -> requests.Response:
    global _LAST_SEARCH_STARTED
    with _SEARCH_REQUEST_LOCK:
        elapsed = monotonic() - _LAST_SEARCH_STARTED
        if elapsed < _MIN_SEARCH_INTERVAL_SECONDS:
            sleep(_MIN_SEARCH_INTERVAL_SECONDS - elapsed)
        _LAST_SEARCH_STARTED = monotonic()
        return requests.get(url, **kwargs)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; background: #f4f6fb; color: #172033; }}
    body {{ margin: 0 auto; max-width: 900px; padding: 3rem 1.25rem; line-height: 1.7; }}
    header {{ border-bottom: 3px solid #5564e8; margin-bottom: 2rem; padding-bottom: 1rem; }}
    header p {{ color: #5564e8; font-size: .75rem; font-weight: 700; letter-spacing: .12em; margin: 0 0 .4rem; text-transform: uppercase; }}
    h1, h2, h3 {{ line-height: 1.25; }}
    h1 {{ font-size: clamp(2rem, 5vw, 3rem); margin: 0; }}
    h2 {{ border-bottom: 1px solid #dfe3ed; margin-top: 2.5rem; padding-bottom: .6rem; }}
    main > ul {{ list-style: none; margin: 1rem 0; padding: 0; }}
    main > ul > li {{ background: #fff; border: 1px solid #e3e7f0; border-radius: .75rem; box-shadow: 0 3px 12px #1720330b; margin: .8rem 0; padding: 1rem 1.15rem; }}
    a {{ color: #4352cf; text-decoration-thickness: 1px; text-underline-offset: 3px; }}
    main > ul > li a {{ font-size: .9rem; }}
    table {{ border-collapse: collapse; display: block; max-width: 100%; overflow-x: auto; }}
    th, td {{ border: 1px solid #dadce0; padding: 0.5rem 0.75rem; text-align: left; }}
    pre {{ background: #f1f3f4; border-radius: 0.4rem; overflow-x: auto; padding: 1rem; }}
    code {{ font-family: ui-monospace, monospace; }}
    img {{ height: auto; max-width: 100%; }}
    @media (max-width: 600px) {{ body {{ padding: 1.5rem 1rem; }} main > ul > li {{ padding: .8rem; }} }}
  </style>
</head>
<body>
  <header>
    <p>AI News Briefing</p>
    <h1>{title}</h1>
  </header>
  <main>
    {body}
  </main>
</body>
</html>"""


@tool
def search_ainews(query: str, max_results: int = 5) -> str:
    """
    Search for AI news articles based on a query.

    Args:
        query (str): The search query.
        max_results (int): The maximum number of results to return.

    Returns:
        str: A formatted string containing the search results.
    """
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return "Error: NEWS_API_KEY environment variable is not set."

    url = "https://gnews.io/api/v4/search"
    now = datetime.now(timezone.utc)
    date_from = (
        (now - timedelta(days=NEWS_LOOKBACK_DAYS))
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    date_to = now.isoformat(timespec="seconds").replace("+00:00", "Z")
    response: requests.Response | None = None
    try:
        for attempt in range(3):
            response = _rate_limited_get(
                url,
                params={
                    "q": query,
                    "lang": "en",
                    "apikey": api_key,
                    "max": max(1, min(max_results, 10)),
                    "from": date_from,
                    "to": date_to,
                },
                timeout=15,
            )
            if response.status_code != 429 or attempt == 2:
                break
            sleep(_MIN_SEARCH_INTERVAL_SECONDS * (attempt + 1))
    except requests.exceptions.RequestException as e:
        detail = str(e).replace(api_key, "[redacted]")
        return f"Error: News search request failed: {detail}"

    if response is None:
        raise RuntimeError("The news provider request did not return a response.")

    try:
        data = response.json()
    except ValueError:
        return "Error: News provider returned an invalid JSON response."

    if not isinstance(data, dict):
        return "Error: News provider returned an unexpected response."

    if not response.ok:
        detail = data.get("errors", data.get("message", response.reason))
        return f"Error: News provider returned HTTP {response.status_code}: {detail}"

    articles = data.get("articles")
    if not isinstance(articles, list):
        return "Error: News provider response did not contain an articles list."

    if not articles:
        return "No articles found for the given query."

    results = []
    for article in articles:
        if not isinstance(article, dict):
            continue

        title = article.get("title", "No title")
        description = article.get("description", "No description")
        article_url = article.get("url", "")
        published_at = article.get("publishedAt")
        source = article.get("source")
        source_name = source.get("name") if isinstance(source, dict) else None
        details = [
            f"Title: {title}",
            f"Description: {description}",
            f"URL: {article_url}",
        ]
        if published_at:
            details.append(f"Published: {published_at}")
        if source_name:
            details.append(f"Source: {source_name}")
        results.append("\n".join(details))

    if not results:
        return "No usable articles were returned for the given query."

    return "\n\n".join(results)


def _render_stories(stories: list[NewsStory], note: str | None) -> list[str]:
    lines: list[str] = []
    if note:
        lines.extend([note.strip(), ""])
    for story in stories:
        citation = f"[{story.source_name or 'Source'}]({story.url})"
        if story.published_date:
            citation += f", {story.published_date}"
        lines.append(f"- **{story.headline}** — {story.summary} ({citation})")
    if not stories and not note:
        lines.append("No recent articles were returned for this topic.")
    return lines


def _newsletter_markdown(
    title: str,
    introduction: str,
    topic_requests: Sequence[TopicRequest],
    research_store: dict[str, TopicResearch],
) -> str:
    parts = [f"# {title}", "", introduction.strip(), ""]
    all_stories: list[NewsStory] = []

    for request in topic_requests:
        research = research_store.get(request.topic)
        if research is None:
            raise RuntimeError(f"No research result was collected for {request.topic!r}.")
        parts.extend(
            (
                f"## {request.topic}",
                "",
                *_render_stories(research.stories, research.note),
                "",
            )
        )
        all_stories.extend(research.stories)

    parts.extend((f"## {SOURCES_SECTION}", ""))
    seen_urls: set[str] = set()
    sources: list[str] = []
    for story in all_stories:
        if story.url in seen_urls:
            continue
        seen_urls.add(story.url)
        source = f"- [{story.source_name or story.headline}]({story.url})"
        if story.published_date:
            source += f" — {story.published_date}"
        sources.append(source)
    parts.extend(sources or ["No sources were found in the recent search."])
    return "\n".join(parts).strip()


def _grounded_copy(
    topic_requests: Sequence[TopicRequest],
    research_store: dict[str, TopicResearch],
) -> tuple[str, str]:
    counts = [
        f"{len(research_store[request.topic].stories)} "
        f"{'article' if len(research_store[request.topic].stories) == 1 else 'articles'} "
        f"for {request.topic}"
        for request in topic_requests
    ]
    return (
        "AI News Briefing",
        f"This issue includes {', '.join(counts)} from searches in the past "
        f"{NEWS_LOOKBACK_DAYS} days. See each section for its source links and notes.",
    )


def create_newsletter_markdown_tool(
    topic_requests: Sequence[TopicRequest],
    research_store: dict[str, TopicResearch],
) -> BaseTool:
    @tool("generate_newsletter_markdown", return_direct=True)
    def generate_newsletter_markdown(title: str, introduction: str) -> str:
        """Build Markdown from the source-verified results collected by the topic subagents."""
        missing = [
            request.topic
            for request in topic_requests
            if request.topic not in research_store
        ]
        if missing:
            raise RuntimeError("Research is missing for: " + ", ".join(missing))

        if not title.strip() or not introduction.strip():
            title, introduction = _grounded_copy(topic_requests, research_store)
        elif UNSUPPORTED_COPY_CLAIMS.search(f"{title} {introduction}"):
            logger.warning(
                "The editor supplied unsupported trend or impact language; "
                "using a factual title and introduction instead."
            )
            title, introduction = _grounded_copy(topic_requests, research_store)

        return _newsletter_markdown(
            title,
            introduction,
            topic_requests,
            research_store,
        )

    return generate_newsletter_markdown


def create_topic_search_tool(
    topic: str,
    query: str,
    research_store: dict[str, TopicResearch],
) -> BaseTool:
    @tool("search_news", return_direct=True)
    def search_news() -> str:
        """Search recent news for this research subagent's assigned topic."""
        result = search_ainews.invoke({"query": query, "max_results": 5})
        if not isinstance(result, str):
            raise RuntimeError(f"News search for {topic} returned non-text results.")

        research = parse_search_results(result)
        research_store[topic] = research
        if research.note:
            return f"Search for {topic} finished with a note: {research.note}"

        count = len(research.stories)
        return (
            f"Search for {topic} found {count} "
            f"{'article' if count == 1 else 'articles'}."
        )

    return search_news


@tool
def markdown_to_html(markdown_text: str, title: str = "This Week in AI News") -> str:
    """Convert a Markdown newsletter into a complete, styled HTML page.
    Returns the full HTML document as a string."""
    body = md.markdown(markdown_text, extensions=["tables", "fenced_code"])
    body = nh3.clean(body)
    heading_match = re.match(r"\s*<h1\b[^>]*>(.*?)</h1>\s*", body, flags=re.IGNORECASE | re.DOTALL)
    if heading_match:
        title = unescape(re.sub(r"<[^>]*>", "", heading_match.group(1)))
        body = body[heading_match.end():]
    safe_title = escape(title)
    return HTML_TEMPLATE.format(title=safe_title, body=body)