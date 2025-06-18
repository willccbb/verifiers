import os

import chromadb
from chromadb.utils import embedding_functions
from datasets import load_dataset
from openai import OpenAI

import verifiers as vf
from verifiers.rubrics.judge_rubric import JudgeRubric


"""
Multi-GPU training (single node, 4 training + 4 inference)

CUDA_VISIBLE_DEVICES=0,1,2,3 vf-vllm --model willcb/Qwen3-8B-Wiki-Search-SFT

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file configs/zero3.yaml verifiers/examples/wiki_search.py
"""

WIKI_DIR = "notebooks/data/wiki" 
CHROMA_DB_DIR = "notebooks/.chroma_db" 

# Create embedding function using OpenAI

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ.get("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)
db_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = db_client.get_collection("wiki_titles", embedding_function=openai_ef) # type: ignore
 
def search_pages(query: str) -> list[dict]:
    """Search for top 10 relevant articles using title embedding similarity.
    
    Args:
        query (str): The query to search for.

    Returns:
        list[dict]: A list of dicts with page_id and title.

    Examples:
        "basketball" -> [{"page_id": "basketball", "title": "Basketball"}, {"page_id": "basketball_rules", "title": "Basketball Rules"}, ...]
    """
    results = collection.query(
        query_texts=[query],
        n_results=10
    )
    
    # Format results
    output = []
    for i in range(len(results['ids'][0])):
        output.append({
            "page_id": results['ids'][0][i],
            "title": results['metadatas'][0][i]['title'] # type: ignore
        })
    
    return output

# test search_pages
print(search_pages("basketball"))

def view_sections(page_id: str) -> list[dict]:
    """View the sections of a page.
    
    Args:
        page_id (str): The ID of the page to view.

    Returns:
        list[dict]: A list of dicts with section_id and section_name.

    Examples:
        "basketball" -> [{"section_id": "basketball:history", "section_name": "History"}, ...]
    """
    # Find the file for this page_id
    results = collection.get(ids=[page_id])
    if not results['ids']:
        raise ValueError(f"Page not found: {page_id}")
    
    filename = results['metadatas'][0]['title'] + '.md'  # type: ignore
    filepath = os.path.join(WIKI_DIR, filename) # type: ignore
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    sections = []
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('#'):
            # Extract section name (remove # and whitespace)
            section_name = line.lstrip('#').strip()
            # Create section ID
            section_id = f"{page_id}:{section_name.lower().replace(' ', '_')}"
            sections.append({
                "section_id": section_id,
                "section_name": section_name,
                "start_line": i
            })
    
    # If no sections found, return the whole page as one section
    if not sections:
        sections.append({
            "section_id": f"{page_id}:full",
            "section_name": "Full Page",
            "start_line": 0
        })
    
    return [{"section_id": s["section_id"], "section_name": s["section_name"]} 
            for s in sections]

# test view_sections
view_sections("baseball")


def read_section(section_id: str) -> str:
    """Read a section of a page.
    
    Args:
        section_id (str): The ID of the section to read.

    Returns:
        str: The content of the section.
        
    Examples:
        "baseball:finnish_baseball" -> "Finnish baseball is a sport that is played in Finland..."
    """
    # Parse section_id
    if ':' not in section_id:
        raise ValueError("Invalid section_id format. Expected: page_id:section_name")
    
    page_id, section_name_id = section_id.split(':', 1)
    
    # Get the file
    results = collection.get(ids=[page_id])
    if not results['ids']:
        raise ValueError(f"Page not found: {page_id}")
    
    filename = results['metadatas'][0]['title'] + '.md' # type: ignore
    filepath = os.path.join(WIKI_DIR, filename)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # Special case for "full" section
    if section_name_id == "full":
        return content
    
    # Find the section
    section_start = None
    section_end = None
    
    for i, line in enumerate(lines):
        if line.startswith('#'):
            current_section = line.lstrip('#').strip().lower().replace(' ', '_')
            if current_section == section_name_id and section_start is None:
                section_start = i
            elif section_start is not None and section_end is None:
                section_end = i
                break
    
    # If section found
    if section_start is not None:
        if section_end is None:
            section_end = len(lines)
        return '\n'.join(lines[section_start:section_end])
    else:
        raise ValueError(f"Section not found: {section_id}")
    
print(read_section("baseball:finnish_baseball"))


system_prompt = """
You are a search agent who has access to the following tools for searching over a set of Wikipedia articles:
- search_pages(query: str) -> list[str]: Search the wiki for pages that match the query.
- view_sections(page_id: str) -> list[str]: View the sections of a page.
- read_section(section_id: str) -> str: Read a section of a page.

{tool_descriptions}

You will be given a question, and you must use the tools to find the answer to the question.

You may make up to 10 tool calls before giving your final answer.

In each turn, respond in the following format:

<think>
[your thoughts here]
</think>
<tool>
{{
    "name": "search_pages", # name of the tool to call
    "args": {{
        "query": "query" # arguments to pass to the tool
    }}
}}
</tool>

When you have found the answer, respond in the following format:
<think>
[your thoughts here]
</think>
<answer>
[final answer here]
</answer>
"""

tools = [
    search_pages,
    view_sections,
    read_section,
]

dataset = load_dataset("willcb/wiki-trivia-questions", split="train")

vf_env = vf.ToolEnv(
    dataset=dataset,
    system_prompt=system_prompt,
    tools=tools,
    max_turns=10,
    max_concurrent=512
)

judge_client = OpenAI(base_url="http://0.0.0.0:8008/v1", api_key="EMPTY")
judge_model = "Qwen/Qwen2.5-7B-Instruct"
judge_rubric = JudgeRubric(
    judge_client=judge_client,
    judge_model=judge_model,
    parser=vf_env.parser
)
vf_env.rubric = vf.RubricGroup(rubrics=[judge_rubric, vf_env.rubric])


model_name = "willcb/Qwen3-8B-Wiki-Search-SFT"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "wiki-trivia-grpo_" + model_name.split("/")[-1].lower()

training_args=vf.grpo_defaults(run_name=run_name)
training_args.per_device_train_batch_size=12
training_args.num_generations=24
training_args.gradient_accumulation_steps=12
training_args.num_iterations=1
training_args.num_train_epochs=5
training_args.max_prompt_length=1024
training_args.max_completion_length=4096
training_args.max_steps=500
training_args.save_steps=100

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train() 