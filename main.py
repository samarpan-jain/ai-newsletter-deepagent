import asyncio
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from tools import search_ainews, markdown_to_html

load_dotenv()

def _validate_newsletter(markdown_text: str) -> None:
    if markdown_text.lstrip().startswith("{"):
        raise RuntimeError("The newsletter builder returned structured data, not Markdown.")

    required_sections = ("Advancements in AI", "AI Layoffs", "Sources")
    missing = [
        section
        for section in required_sections
        if not re.search(
            rf"(?im)^#{{1,6}}\s+{re.escape(section)}\b",
            markdown_text,
        )
    ]
    if missing:
        raise RuntimeError(
            "The editor returned an incomplete newsletter; missing sections: "
            + ", ".join(missing)
        )


def _validate_newsletter_html(html_text: str) -> None:
    if not html_text.lstrip().startswith("<!DOCTYPE html>"):
        raise RuntimeError("The markdown_to_html tool returned invalid HTML.")

    required_sections = ("Advancements in AI", "AI Layoffs", "Sources")
    missing = [
        section
        for section in required_sections
        if not re.search(
            rf"(?is)<h[1-6]\b[^>]*>\s*{re.escape(section)}\s*</h[1-6]>",
            html_text,
        )
    ]
    if missing:
        raise RuntimeError(
            "The HTML newsletter is incomplete; missing sections: "
            + ", ".join(missing)
        )


async def _research_topic(topic: str, query: str) -> str:
    for attempt in range(3):
        result = await asyncio.to_thread(
            search_ainews.invoke,
            {"query": query, "max_results": 5},
        )
        if not isinstance(result, str):
            raise RuntimeError(f"{topic} search returned non-text results.")
        if not result.startswith("Error:"):
            return result
        if "HTTP 429" not in result or attempt == 2:
            raise RuntimeError(f"{topic} search failed: {result}")
        await asyncio.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"{topic} search failed after retries.")


def _newsletter_markdown(searches: list[tuple[str, str]]) -> str:
    sections = [""]
    sources = []
    seen_urls = set()

    for topic, result in searches:
        sections.extend((f"## {topic}", ""))
        story_count = 0
        for article in result.split("\n\n"):
            fields = {}
            for line in article.splitlines():
                key, separator, value = line.partition(": ")
                if separator:
                    fields[key.casefold()] = value.strip()
            title = fields.get("title")
            description = fields.get("description")
            url = fields.get("url", "")
            if not title or not description or not url.startswith(("http://", "https://")):
                continue
            if url in seen_urls:
                continue

            seen_urls.add(url)
            story_count += 1
            date = fields.get("published")
            summary = f"- **{title}** — {description}"
            if date:
                summary += f" *(Published: {date})*"
            sections.extend((summary, ""))
            sources.append(f"- [{title}]({url})")

        if not story_count:
            sections.extend(("No usable sources were returned for this topic.", ""))

    sections.extend(("## Sources", ""))
    sections.extend(sources or ["No source links were returned."])
    return "\n".join(sections)


async def _edit_newsletter() -> str:
    searches = []
    for topic, query in (
        ("Advancements in AI", "recent advancements in artificial intelligence"),
        ("AI Layoffs", "AI layoffs"),
    ):
        if searches:
            await asyncio.sleep(1.1)
        searches.append((topic, await _research_topic(topic, query)))

    markdown_newsletter = _newsletter_markdown(searches)
    _validate_newsletter(markdown_newsletter)
    html = markdown_to_html.invoke(
        {
            "markdown_text": markdown_newsletter,
            "title": "This Week in AI News",
        }
    )
    if not isinstance(html, str):
        raise RuntimeError("The markdown_to_html tool returned non-text output.")
    _validate_newsletter_html(html)
    return html


async def _generate_newsletter() -> None:
    newsletter_html = await _edit_newsletter()
    output_path = Path(__file__).resolve().with_name("newsletter.html")
    output_path.write_text(newsletter_html, encoding="utf-8")
    print(f"Newsletter saved to {output_path}")


def main() -> None:
    try:
        asyncio.run(_generate_newsletter())
    except RuntimeError as error:
        print(f"Newsletter generation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
