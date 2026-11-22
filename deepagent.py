import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Sequence
import httpx
from dotenv import load_dotenv
from deepagents import (
    CompiledSubAgent,
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama
from tools import (
    NEWS_LOOKBACK_DAYS,
    SOURCES_SECTION,
    TopicRequest,
    TopicResearch,
    create_newsletter_markdown_tool,
    create_topic_search_tool,
    markdown_to_html,
)

load_dotenv()

MODEL_NAME = "qwen3:8b"
DEFAULT_TOPIC_QUERIES = (
    ("Advancements in AI", "new AI model research artificial intelligence technology"),
    ("AI Layoffs", "AI layoffs"),
)

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
    reasoning=False,
    num_predict=384,
)

register_harness_profile(
    f"ollama:{MODEL_NAME}",
    HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        excluded_tools=frozenset(
            {"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute"}
        ),
    ),
)


def _tool_name_for_topic(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", topic.casefold()).strip("_")
    return f"research_{slug or 'topic'}"


def _make_research_subagent(
    request: TopicRequest,
    research_store: dict[str, TopicResearch],
) -> CompiledSubAgent:
    subagent_name = _tool_name_for_topic(request.topic)
    search_tool = create_topic_search_tool(request.topic, request.query, research_store)
    researcher = create_agent(
        model=llm,
        tools=[search_tool],
        system_prompt=f"""
            You are the research subagent assigned to {request.topic!r}.
            Call search_news exactly once. Do not answer before searching and do not invent
            or summarize article facts.
        """,
        name=subagent_name,
    )

    return {
        "name": subagent_name,
        "description": (
            f"Search recent news for {request.topic!r} using the assigned query."
        ),
        "runnable": researcher,
    }


def _create_editor(
    topic_requests: Sequence[TopicRequest],
    research_store: dict[str, TopicResearch],
):
    subagents = [
        _make_research_subagent(request, research_store)
        for request in topic_requests
    ]
    markdown_tool = create_newsletter_markdown_tool(topic_requests, research_store)
    return create_deep_agent(
        model=llm,
        subagents=subagents,
        tools=[markdown_tool],
        system_prompt=f"""
            You are the newsletter editor for articles from the past {NEWS_LOOKBACK_DAYS} days.
            For each research-plan entry, call task once with its exact subagent_type, topic,
            and query. Delegate independent topics in parallel and wait for every task.
            Then call generate_newsletter_markdown once with a neutral title and factual
            one-sentence introduction. Do not search or invent article facts. Return the
            Markdown tool's result unchanged.
        """,
        name="newsletter_editor",
    )


def _resolve_topics() -> list[TopicRequest]:
    requests =[TopicRequest(topic=topic, query=query) for topic, query in DEFAULT_TOPIC_QUERIES]
    if not requests:
        raise RuntimeError("At least one newsletter topic is required.")
    if len({request.topic for request in requests}) != len(requests):
        raise RuntimeError("Newsletter topic names must be unique.")
    if len({_tool_name_for_topic(request.topic) for request in requests}) != len(requests):
        raise RuntimeError("Newsletter topic names must map to unique researcher names.")
    if any(request.topic.casefold() == SOURCES_SECTION.casefold() for request in requests):
        raise RuntimeError(f"{SOURCES_SECTION!r} is reserved for the source list.")
    return requests


async def run_newsletter_editor() -> str:
    topic_requests = _resolve_topics()

    research_store: dict[str, TopicResearch] = {}
    editor = _create_editor(topic_requests, research_store)
    prompt = (
        "Create a newsletter using this research plan. Delegate each entry to "
        "its specified researcher, then call the Markdown tool.\n\n"
        + json.dumps(
            [
                {
                    "topic": request.topic,
                    "query": request.query,
                    "subagent_type": _tool_name_for_topic(request.topic),
                }
                for request in topic_requests
            ],
            ensure_ascii=False,
        )
    )
    try:
        result = await asyncio.wait_for(
            editor.ainvoke({"messages": [{"role": "user", "content": prompt}]}),
            timeout=600,
        )
    except httpx.ConnectError:
        raise RuntimeError(
            f"Could not connect to Ollama at http://localhost:11434 using {MODEL_NAME}. "
            "Start Ollama and verify the model is installed."
        ) from None
    except TimeoutError:
        raise RuntimeError(
            "The Deep Agent editor timed out while delegating research or building Markdown. "
            "Research subagents now call the search tool directly; verify Ollama is responsive."
        ) from None

    markdown_messages = [
        message
        for message in result.get("messages", [])
        if isinstance(message, ToolMessage)
        and message.name == "generate_newsletter_markdown"
        and message.status != "error"
    ]
    if not markdown_messages:
        raise RuntimeError(
            "The DeepAgent editor did not call its research and Markdown tools. "
            f"Confirm {MODEL_NAME} supports tool calling."
        )
    if len(research_store) != len(topic_requests):
        missing = sorted(set(request.topic for request in topic_requests) - research_store.keys())
        raise RuntimeError(
            "The editor did not complete research for: " + ", ".join(missing)
        )
    markdown = markdown_messages[-1].content
    if not isinstance(markdown, str) or not markdown.strip():
        raise RuntimeError("The Markdown generation tool returned no newsletter content.")
    return markdown


def _validate_html(html_text: str, topics: Sequence[TopicRequest]) -> None:
    if not html_text.lstrip().startswith("<!DOCTYPE html>"):
        raise RuntimeError("The markdown_to_html tool returned invalid HTML.")

    missing = [
        section
        for section in [request.topic for request in topics] + [SOURCES_SECTION]
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


async def generate_newsletter() -> tuple[str, str]:
    topic_requests = _resolve_topics()
    markdown_text = await run_newsletter_editor()
    html_text = markdown_to_html.invoke({"markdown_text": markdown_text})
    if not isinstance(html_text, str):
        raise RuntimeError("The markdown_to_html tool returned non-text output.")
    _validate_html(html_text, topic_requests)
    return markdown_text, html_text


async def _run() -> None:
    markdown_text, html_text = await generate_newsletter()
    output_dir = Path(__file__).resolve().parent / "newsLetters"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "newsletter.md").write_text(markdown_text, encoding="utf-8")
    (output_dir / "newsletter.html").write_text(html_text, encoding="utf-8")
    print(f"Newsletter Markdown saved to {output_dir / 'newsletter.md'}")
    print(f"Newsletter HTML saved to {output_dir / 'newsletter.html'}")


def main() -> None:
    try:
        asyncio.run(_run())
    except RuntimeError as error:
        print(f"Newsletter generation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
