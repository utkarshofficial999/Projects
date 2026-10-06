"""
agent_engine.tools.builtin.search_tool
======================================

A web-search tool that performs a DuckDuckGo HTML search and returns
the top results as structured text.

The tool uses only the standard library (``urllib``) so it has no
external dependencies. In production you would swap the HTTP layer
for ``httpx`` or a dedicated search API, but the interface stays the
same.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from agent_engine.tools.base_tool import BaseTool, ToolExecutionError
from agent_engine.tools.registry import register_tool

logger = logging.getLogger(__name__)

# DuckDuckGo HTML endpoint (no API key required)
_DDG_URL = "https://html.duckduckgo.com/html/"
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


class SearchArgs(BaseModel):
    """Arguments for the search tool."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="The search query string.",
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return.",
    )


@register_tool
class SearchTool(BaseTool):
    """Perform a web search and return top results."""

    name = "web_search"
    description = (
        "Search the web using DuckDuckGo. Returns a list of result "
        "titles, URLs, and snippets."
    )
    arguments_schema = SearchArgs

    def execute(self, args: SearchArgs) -> str:
        """Execute the search.

        Returns a formatted string of results, or an error message.
        """
        try:
            results = self._search(args.query, args.max_results)
        except Exception as exc:  # noqa: BLE001
            raise ToolExecutionError(
                self.name, f"Search failed: {exc}"
            ) from exc

        if not results:
            return f"No results found for: {args.query!r}"

        lines: List[str] = [f"Search results for: {args.query!r}\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}")
            lines.append(f"   URL: {r['url']}")
            if r.get("snippet"):
                lines.append(f"   {r['snippet']}")
            lines.append("")
        return "\n".join(lines).strip()

    # ------------------------------------------------------------------
    # Internal HTTP helper
    # ------------------------------------------------------------------

    def _search(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Fetch and parse DuckDuckGo HTML results."""
        params = urllib.parse.urlencode({"q": query})
        url = f"{_DDG_URL}?{params}"

        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        return self._parse_results(html, max_results)

    @staticmethod
    def _parse_results(html: str, max_results: int) -> List[Dict[str, str]]:
        """Extract result entries from DuckDuckGo HTML.

        Uses a simple regex approach to avoid a heavy HTML parser
        dependency. Each result block looks like::

            <a class="result__a" href="...">Title</a>
            <a class="result__snippet" ...>Snippet text</a>
        """
        results: List[Dict[str, str]] = []

        # Match result links and their snippets
        pattern = re.compile(
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
            r'(?:.*?<a[^>]*class="result__snippet"[^>]*>(.*?)</a>)?',
            re.DOTALL,
        )

        for match in pattern.finditer(html):
            if len(results) >= max_results:
                break
            raw_url = match.group(1)
            title = _strip_tags(match.group(2))
            snippet = _strip_tags(match.group(3) or "")

            # DuckDuckGo wraps URLs in a redirect; extract the real URL
            real_url = _extract_ddg_url(raw_url)

            results.append(
                {"title": title, "url": real_url, "snippet": snippet}
            )

        return results


def _strip_tags(text: str) -> str:
    """Remove HTML tags and unescape entities."""
    import html as html_mod

    text = re.sub(r"<[^>]+>", "", text)
    return html_mod.unescape(text).strip()


def _extract_ddg_url(raw: str) -> str:
    """Extract the real URL from a DuckDuckGo redirect link."""
    # DDG redirect format: //duckduckgo.com/l/?uddg=<urlencoded>&rut=...
    if "uddg=" in raw:
        parsed = urllib.parse.urlparse(raw)
        qs = urllib.parse.parse_qs(parsed.query)
        if "uddg" in qs:
            return qs["uddg"][0]
    return raw
