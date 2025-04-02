import os
BASE_URL = "https://api.deepinfra.com/v1/openai"
API_KEY = os.getenv("DEEPINFRA_API_KEY")
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

def get_url_markdown(url: str) -> str:
    """Get contents of URL as nicely formatted markdown."""
    import requests
    from markdownify import markdownify as md
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return md(response.text)
    except Exception as e:
        return f"Error: {str(e)}"

def ask(question: str, url: str) -> str:
    """Ask a question about a web page returned from search results.
    
    Args:
        question: The question to be answered (by an LLM who will be given the web page contents)
        url: The URL of the web page to query

    Returns:
        A LLM-generated answer to the question based on the web page contents.

    Examples:
        {"question": "What is the capital of France?", "url": "https://en.wikipedia.org/wiki/France"} -> "The capital of France is Paris."
        {"question": "How many people live in the United States?", "url": "https://en.wikipedia.org/wiki/United_States"} -> "The population of the United States is approximately 340 million people."
    """
    
    contents = get_url_markdown(url)[:50000]

    if contents.startswith("Error:"):
        return "Error: Failed to fetch URL contents."

    from openai import OpenAI
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    prompt = f"""Answer the following question based on the provided web page contents:

    Question: {question}

    Page: {url}

    Page contents:
    {contents}
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
        )
        return response.choices[0].message.content or "Error: No response from model."
    except Exception as e:
        return f"Error: {str(e)}"