# dictionary_server.py  ───────────────────────────────────────────────────────
from fastmcp import FastMCP
dictionary = FastMCP("dictionary")
_DEFS = {
    "mcp": "Model Context Protocol – a standard tool-calling interface.",
    "python": "A high-level, dynamically typed programming language.",
    "cat": "A small domesticated carnivorous mammal.",
}

@dictionary.tool(description="Return a short English definition of a word.")
def define(word: str) -> str:
    """One-sentence dictionary lookup."""
    return _DEFS.get(word.lower(), f"Sorry, no definition for “{word}”.")

if __name__ == "__main__":
    dictionary.run(port=8005, transport="streamable-http")
