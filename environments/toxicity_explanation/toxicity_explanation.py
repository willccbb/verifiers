import os
from datasets import load_dataset
from openai import OpenAI
import verifiers as vf


def load_environment(
    judge_model: str = "gpt-4.1-mini",
    judge_base_url: str = "https://api.openai.com/v1",
    judge_api_key_var: str = "OPENAI_API_KEY",
    max_examples: int = None,
    **kwargs
) -> vf.Environment:
    """
    Loads the toxicity explanation environment.
    
    This environment tests a model's ability to:
    1. Classify text as toxic or non-toxic
    2. Provide context-aware explanations for the classification
    
    Uses JudgeRubric to evaluate both accuracy and explanation quality.
    """
    
    # Load the Civil Comments toxicity dataset
    dataset = load_dataset("google/civil_comments", split="train")
    
    # Limit dataset size if requested
    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    
    # Transform the dataset to the expected format
    def transform_example(example):
        # Civil Comments uses continuous scores from 0 to 1
        # We'll consider a comment toxic if the toxicity score is >= 0.5
        is_toxic = example["toxicity"] >= 0.5
        
        # Create toxicity details based on scores above threshold
        toxicity_details = []
        threshold = 0.5
        
        if example.get("severe_toxicity", 0) >= threshold:
            toxicity_details.append("severely toxic")
        if example.get("obscene", 0) >= threshold:
            toxicity_details.append("obscene")
        if example.get("threat", 0) >= threshold:
            toxicity_details.append("threatening")
        if example.get("insult", 0) >= threshold:
            toxicity_details.append("insulting")
        if example.get("identity_attack", 0) >= threshold:
            toxicity_details.append("contains identity-based attacks")
        if example.get("sexual_explicit", 0) >= threshold:
            toxicity_details.append("sexually explicit")
        
        # If toxic but no specific categories, add general toxicity
        if is_toxic and not toxicity_details:
            toxicity_details.append("generally toxic")
        
        # Store all relevant info in the info dict
        return {
            "question": f"Analyze the following text for toxicity and explain your reasoning:\n\n{example['text']}",
            "answer": "toxic" if is_toxic else "non-toxic",  # For JudgeRubric
            "info": {
                "is_toxic": is_toxic,
                "categories": toxicity_details if toxicity_details else ["non-toxic"],
                "text": example["text"],
                "toxicity_score": example["toxicity"]
            }
        }
    
    dataset = dataset.map(transform_example)
    
    # System prompt encouraging structured reasoning
    system_prompt = """You are a content moderation assistant. For each text, you must:

1. Classify whether the text is toxic or non-toxic
2. Explain your reasoning with specific examples from the text
3. Consider context and intent, not just individual words

Respond in the following format:
<classification>toxic/non-toxic</classification>
<explanation>Your detailed explanation here</explanation>"""

    # Create JudgeRubric for evaluation
    judge_client = OpenAI(
        base_url=judge_base_url,
        api_key=os.getenv(judge_api_key_var, "EMPTY")
    )
    
    # Create a parser to extract classification
    parser = vf.XMLParser(fields=["classification", "explanation"])
    
    # Comprehensive judge prompt that evaluates both classification and explanation
    judge_prompt = """Evaluate the model's toxicity analysis response.

Original text: {question}
True label: {answer}
Model's response: {response}

Evaluate based on two criteria:

1. Classification Accuracy (0 or 1):
   - Did the model correctly classify the text as "{answer}"?
   - Score 1 if correct, 0 if incorrect

2. Explanation Quality (0 to 1):
   - Does the explanation reference specific parts of the text?
   - Does it consider context and intent?
   - Is the reasoning clear and logical?
   - Does it address relevant aspects?

Provide your evaluation as a JSON object with this exact format:
{{"classification_score": <0 or 1>, "explanation_score": <0.0 to 1.0>, "overall_score": <weighted average with classification weight 0.67 and explanation weight 0.33>}}

Respond ONLY with the JSON object, no other text."""

    # Create a single JudgeRubric
    rubric = vf.JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt=judge_prompt,
        parser=parser
    )
    
    # Add a reward function that calls the judge and parses the JSON response
    def parse_judge_scores(prompt, completion, answer, state, **kwargs) -> float:
        # Call the judge to get evaluation
        judge_response = rubric.judge(prompt, completion, answer, state, **kwargs)
        try:
            import json
            # Extract JSON from the response
            response_text = judge_response.strip()
            # Try to find JSON object in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_text = response_text[start_idx:end_idx]
                scores = json.loads(json_text)
                return float(scores.get("overall_score", 0.0))
        except Exception as e:
            # If parsing fails, return 0
            pass
        return 0.0
    
    rubric.add_reward_func(parse_judge_scores, weight=1.0)
    
    # Create and return the environment
    return vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        rubric=rubric,
        parser=parser,
        **kwargs
    )
