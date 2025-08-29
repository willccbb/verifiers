import os
import re
import asyncio
import json
from typing import List, Tuple, Optional

import numpy as np
import verifiers as vf
from datasets import Dataset
from openai import AsyncOpenAI

# Use the bundled question list; no external repo dependency
try:
    from .question_list import questions as LOCAL_QUESTIONS
except Exception:
    LOCAL_QUESTIONS = []  # type: ignore

# Fallback questions if the local list fails to import (full list)
DEFAULT_QUESTIONS: List[str] = [
    "Provide an explanation for Japan's Lost Decades.",
    "What is a cause of World War 1?",
    "Why might the United States government nationalize ASI development?",
    "How might you use a brick and a blanket?",
    "What architectural design features should be included in a tasteful home?",
    "What's one way to use oregano?",
    "How might we enable LLMs to spend more output tokens to get predictably better results?",
    "Propose a solution to Los Angeles traffic.",
    "What activities might I include at a party for firefighters?",
    "Why did Rome fall?",
    "How could we redesign the American education system to better prepare students for the 22nd century?",
    "What might be an unexpected consequence of achieving nuclear fusion?",
    "Describe a plausible alien life form that doesn't rely on carbon-based biology.",
    "How could we modify the rules of chess to make it more exciting for spectators?",
    "What would be the implications of a universal basic income on American society?",
    "Propose an alternative to democracy for successfully and fairly governing a country.",
    "How might we terraform Venus instead of Mars, and why?",
    "Design an original sport that combines elements of three existing sports.",
    "What could be a novel use for blockchain technology outside of cryptocurrency?",
    "How might human evolution be affected by long term space colonization?",
    "Invent a new musical instrument and describe how it would be played.",
    "What might be an unexpected solution to reducing plastic waste in oceans?",
    "How might we design a city that functions entirely underwater?",
    "What societal changes might occur if humans could communicate with animals?",
    "I have a fleet of 100 drones, how can I use them?",
    "Describe a sustainable farming method that could be used in a floating city.",
    "If all industrial buildings were required to be bioluminescent, what effects might this have?",
    "Invent a device that translates human dreams into tangible visualizations.",
    "How might daily life change if humans had the ability to breathe underwater?",
    "Create a recipe for a smoothie to have first thing in the morning that will give me energy.",
    "What new environmental challenges might arise if all vehicles were self-driving?",
    "Design a fashion line that incorporates smart clothing technology.",
    "Imagine a world where books are replaced by holographic storytelling; what impacts might this have?",
    "What might be the implications of having robots as therapists?",
    "Propose a system for energy-harvesting from natural disasters.",
    "How might the education system be revolutionized by virtual reality classrooms?",
    "What unique challenges might arise in a society where everyone lives to be 150 years old?",
    "Describe a mobile app that encourages acts of kindness.",
    "Give me a diet that a human should eat to best prepare them for a hypothetical hibernation.",
    "Imagine a competition where contestants build habitats for animals; what might be included?",
    "What might be the benefits of reintroducing dinosaurs into modern ecosystems?",
    "Propose a mechanism for reducing food waste through technological innovation.",
    "Design a city where all modes of transportation are vertically oriented.",
    "What is a useless ingredient for a baker to have in their kitchen?",
    "Imagine a civilization based entirely on underwater agriculture; what technology might be required?",
    "How might public health improve if all houses had healing gardens?",
    "Describe how to build a time travel machine assuming I can procure any required material.",
    "Create a concept for a museum that showcases possible futures.",
    "What would be the impact of a government mandating weekly mental health days?",
    "Invent a game that teaches players about sustainable living.",
    "How could we design a school that encourages lifelong learning from adults as well as children?",
    "Describe a new form of professional sports that focuses on non-physical competition.",
    "Devise a farming technique to harvest dinoflagellates and retain their bioluminescence.",
    "How can a perfumer increase the sillage of their scent?",
    "What can an artist who enjoys Basquiat’s art take inspiration from?",
    "Provide a proof for the Pythagorean theorem.",
    "A perfumer is creating a unique, unisex scent with benzoin and vanilla middle notes, what base and top notes should they add?",
    "What is a non poisonous recipe nobody has prepared before?",
    "Design an earring that would complement someone with a round face and small ears.",
    "Devise a strategy for me to always find gems when I mine in Webkinz World’s mines.",
    "What is a human value to align a large language model on?",
    "Give me a proof to Euclid's theorem of the infinitude of primes.",
    "Give a proof for the Cauchy-Schwarz Inequality.",
]


def _extract_tag(text: str, tag: str) -> str:
    pattern = f"<{tag}>(.*?)</{tag}>"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1].strip() if matches else text


def _build_prompt(
    question: str,
    prev_answers: List[str],
    reasoning_effort: Optional[str] = None,
) -> str:
    base_prompt = (
        "Answer the following question:.\n"
        "<question>" + question + "</question>\n"
        "Provide your answer in <answer></answer> XML tags.\n"
    )

    base_prompt += (
        "Your response should be one direct answer. Only provide one answer. "
        "DO NOT list multiple answers. Please try to be concise.\n"
    )

    if prev_answers:
        prev_str = "\n\n".join(
            [f"<previous_answer id='{i+1}'>\n{ans}\n</previous_answer>" for i, ans in enumerate(prev_answers)]
        )
        base_prompt += (
            "IMPORTANT: Provide an answer you *HAVE NOT* given previously.\n"
            "Your previous answers are inside of <previous_answers></previous_answers> XML tags.\n"
            "<previous_answers>\n" + prev_str + "\n</previous_answers>"
        )

    return base_prompt


class AidanBenchEnv(vf.MultiTurnEnv):
    """
    Multi-turn Environment that replicates AidanBench’s generation loop:
    - Repeatedly ask for novel answers to the same question
    - After each model response, compute coherence (o1-mini judge) and novelty (embeddings)
    - Stop when any threshold fails; reward = number of valid answers
    """

    def __init__(
        self,
        dataset: Dataset | None = None,
        questions: Optional[List[str]] = None,
        questions_path: Optional[str] = None,
        thresholds: dict | None = None,
        use_llm_similarity: bool = False,
        reasoning_effort: Optional[str] = None,
        max_turns: int = 20,
        num_questions: int | None = None,
        # Judge configuration
        judge_model: str = "o1-mini",
        judge_api_base_url: str = "https://openrouter.ai/api/v1",
        judge_api_key_var: str = "OPEN_ROUTER_KEY",
        # Embedding configuration
        embedding_model: str = "text-embedding-3-large",
        embedding_api_base_url: str = "https://api.openai.com/v1",
        embedding_api_key_var: str = "OPENAI_API_KEY",
        **kwargs,
    ):
        # Defaults mirror AidanBench’s interactive runner
        self.thresholds = thresholds or {
            "coherence_score": 15.0,
            "embedding_dissimilarity_score": 0.15,
            "llm_dissimilarity_score": 0.15,
        }
        self.use_llm_similarity = use_llm_similarity
        self.reasoning_effort = reasoning_effort
        # Judge client/config
        self.judge_model = judge_model
        judge_key = os.getenv(judge_api_key_var, "")
        if not judge_key:
            raise ValueError(
                f"Environment requires judge API key: set {judge_api_key_var} in your environment."
            )
        self.judge_client = AsyncOpenAI(api_key=judge_key, base_url=judge_api_base_url)
        # Embedding client/config
        self.embedding_model = embedding_model
        embed_key = os.getenv(embedding_api_key_var, "")
        if not embed_key:
            raise ValueError(
                f"Environment requires embedding API key: set {embedding_api_key_var} in your environment."
            )
        self.embed_client = AsyncOpenAI(api_key=embed_key, base_url=embedding_api_base_url)

        # Build dataset from (in priority): provided dataset > questions arg > questions_path > AidanBench local > fallback
        if dataset is None:
            qlist: List[str] = []
            if questions is not None and len(questions) > 0:
                qlist = questions
            elif questions_path is not None and os.path.exists(questions_path):
                try:
                    with open(questions_path, "r") as f:
                        text = f.read()
                    # Try JSON list first; fallback to line-delimited text
                    try:
                        obj = json.loads(text)
                        if isinstance(obj, list):
                            qlist = [str(x) for x in obj]
                        else:
                            raise ValueError("questions_path JSON must be a list of strings")
                    except json.JSONDecodeError:
                        qlist = [line.strip() for line in text.splitlines() if line.strip()]
                except Exception:
                    qlist = []
            elif LOCAL_QUESTIONS:
                qlist = LOCAL_QUESTIONS
            else:
                qlist = DEFAULT_QUESTIONS

            if num_questions is not None and num_questions > 0:
                qlist = qlist[: num_questions]

            def fmt_example(q: str):
                prompt = [{"role": "user", "content": _build_prompt(q, [], self.reasoning_effort)}]
                return {"prompt": prompt, "answer": "", "info": {"question": q}, "task": "aidanbench"}

            data = [fmt_example(q) for q in qlist]
            dataset = Dataset.from_list(data)
        else:
            # If user supplies a dataset with 'question' only, preformat to include AidanBench instructions
            if "prompt" not in dataset.column_names and "question" in dataset.column_names:
                def _map_row(x):
                    q = x["question"]
                    p = [{"role": "user", "content": _build_prompt(q, [], self.reasoning_effort)}]
                    y = dict(x)
                    y["prompt"] = p
                    return y
                dataset = dataset.map(_map_row)

        # Default rubric: count valid answers; also log average coherence/novelty as metrics
        def aidanbench_score(state, **kwargs) -> float:
            answers = state.get("aidanbench", {}).get("answers", [])
            return float(len(answers))

        def avg_coherence(state, **kwargs) -> float:
            scores = state.get("aidanbench", {}).get("coherence_scores", [])
            return float(sum(scores) / len(scores)) if scores else 0.0

        def avg_embedding_novelty(state, **kwargs) -> float:
            scores = state.get("aidanbench", {}).get("embedding_novelty_scores", [])
            return float(sum(scores) / len(scores)) if scores else 0.0

        def avg_llm_novelty(state, **kwargs) -> float:
            scores = state.get("aidanbench", {}).get("llm_novelty_scores", [])
            return float(sum(scores) / len(scores)) if scores else 0.0

        rubric = vf.Rubric(
            funcs=[aidanbench_score, avg_coherence, avg_embedding_novelty, avg_llm_novelty],
            weights=[1.0, 0.0, 0.0, 0.0],
        )

        super().__init__(
            dataset=dataset,
            rubric=rubric,
            message_type="chat",
            max_turns=max_turns,
            **kwargs,
        )

    def format_prompt(
        self,
        prompt_str: str,
        system_prompt: str | None = None,
        few_shot: list[dict] | None = None,
    ) -> list[dict]:
        # Override to inject AidanBench base prompt structure for the first turn
        return [{"role": "user", "content": _build_prompt(prompt_str, [], self.reasoning_effort)}]

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        # Initialize tracking for AidanBench metrics
        state.setdefault("aidanbench", {})
        state["aidanbench"].setdefault("answers", [])
        state["aidanbench"].setdefault("coherence_scores", [])
        state["aidanbench"].setdefault("embedding_novelty_scores", [])
        state["aidanbench"].setdefault("llm_novelty_scores", [])
        state["aidanbench"].setdefault("termination_reason", "")
        return state

    async def is_completed(self, messages: vf.Messages, state: vf.State, **kwargs) -> bool:
        # Wait until we have at least one model response
        if state["turn"] == 0:
            return False

        # Extract the last assistant answer
        assert isinstance(messages, list)
        last_assistant = None
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "assistant":
                last_assistant = m.get("content", "") or ""
                break
        if last_assistant is None:
            return False

        # Parse <answer>...</answer> if present
        new_answer = _extract_tag(last_assistant, "answer")

        # Grab question and previous accepted answers
        question = state.get("info", {}).get("question", "")
        prev_answers: List[str] = state["aidanbench"]["answers"]

        # Compute coherence via configurable judge
        coherence_score = float(await self._judge_coherence(question, new_answer))

        # Compute embedding novelty
        if not prev_answers:
            embedding_novelty = 1.0
        else:
            new_vec = await self._embed(new_answer)
            prev_vecs = await asyncio.gather(*[self._embed(ans) for ans in prev_answers])
            sims = [
                float(np.dot(new_vec, pv) / (np.linalg.norm(new_vec) * np.linalg.norm(pv)))
                for pv in prev_vecs
            ]
            embedding_novelty = 1.0 - max(sims)

        # Optional LLM similarity-based novelty
        llm_novelty = 1.0
        if self.use_llm_similarity and prev_answers:
            sims = await asyncio.gather(
                *[self._judge_similarity(question, new_answer, pa) for pa in prev_answers]
            )
            llm_novelty = 1.0 - max([float(s) for s in sims])

        # Record metrics for this attempt (accepted or not)
        passed = (
            coherence_score > self.thresholds["coherence_score"]
            and embedding_novelty >= self.thresholds["embedding_dissimilarity_score"]
            and (
                (not self.use_llm_similarity)
                or (llm_novelty >= self.thresholds["llm_dissimilarity_score"])
            )
        )

        if passed:
            state["aidanbench"]["answers"].append(new_answer)
            state["aidanbench"]["coherence_scores"].append(coherence_score)
            state["aidanbench"]["embedding_novelty_scores"].append(embedding_novelty)
            if self.use_llm_similarity:
                state["aidanbench"]["llm_novelty_scores"].append(llm_novelty)
            return False
        else:
            reason = []
            if coherence_score <= self.thresholds["coherence_score"]:
                reason.append("low_coherence")
            if embedding_novelty < self.thresholds["embedding_dissimilarity_score"]:
                reason.append("low_embedding_novelty")
            if self.use_llm_similarity and llm_novelty < self.thresholds["llm_dissimilarity_score"]:
                reason.append("low_llm_novelty")
            state["aidanbench"]["termination_reason"] = ",".join(reason) or "threshold"
            return True

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs
    ) -> Tuple[vf.Messages, vf.State]:
        # Provide a new user prompt instructing the model to avoid previous answers
        question = state.get("info", {}).get("question", "")
        prev_answers: List[str] = state["aidanbench"]["answers"]
        next_prompt = _build_prompt(question, prev_answers, self.reasoning_effort)
        return ([{"role": "user", "content": next_prompt}], state)

    # -----------------------
    # Internal helper methods
    # -----------------------

    async def _judge_coherence(self, question: str, answer: str) -> float:
        prompt = (
            "Your task is to evaluate the coherence and plausibility of an answer to a given question.\n\n"
            "Question: <question>" + question + "</question>\n"
            "Answer: <answer>" + answer + "</answer>\n\n"
            "Based on the following criteria, provide a Coherence and Plausibility Score on a scale of 0 - 100:\n\n"
            "0-20: INCOHERENT/NONSENSICAL\n"
            "- Answer is completely unrelated to the question\n"
            "- Contains logical impossibilities or contradictions\n"
            "- Makes claims that defy basic reality\n"
            "- Shows no understanding of the question's context\n\n"
            "21-40: SEVERELY FLAWED\n"
            "- Major logical gaps or inconsistencies\n"
            "- Significant misunderstanding of core concepts\n"
            "- Contains partially relevant information but mostly incorrect\n"
            "- May include some true statements but fails to form a coherent response\n\n"
            "41-60: PARTIALLY COHERENT\n"
            "- Shows basic understanding of the question\n"
            "- Contains some valid points mixed with errors\n"
            "- Logic is followable but may have weak connections\n"
            "- Answer is relevant but may miss key aspects\n\n"
            "61-80: MOSTLY COHERENT\n"
            "- Demonstrates clear understanding of the question\n"
            "- Logic is sound with minor gaps or inconsistencies\n"
            "- Most claims are plausible and well-supported\n"
            "- Forms a generally complete and relevant response\n\n"
            "81-100: HIGHLY COHERENT\n"
            "- Perfectly addresses the question\n"
            "- Demonstrates complete logical consistency\n"
            "- All claims are plausible and well-grounded\n"
            "- Forms a comprehensive and precise response\n\n"
            "IMPORTANT: Provide your final Coherence and Plausibility Score as a single integer between 0 and 100, "
            "enclosed in <coherence_score></coherence_score> XML tags. For example:\n"
            "<coherence_score>75</coherence_score>\n\n"
            "Do not include any additional text in your response."
        )
        resp = await self.judge_client.chat.completions.create(
            model=self.judge_model, messages=[{"role": "user", "content": prompt}]
        )
        content = resp.choices[0].message.content or ""
        try:
            return float(_extract_tag(content, "coherence_score"))
        except Exception:
            return 0.0

    async def _judge_similarity(self, question: str, answer1: str, answer2: str) -> float:
        prompt = (
            "Your task is to evaluate how semantically similar two answers are to the same question, "
            "focusing on core concepts and meaning rather than exact wording.\n\n"
            "Original Question: <question>" + question + "</question>\n"
            "First Answer: <answer1>" + answer1 + "</answer1>\n"
            "Second Answer: <answer2>" + answer2 + "</answer2>\n\n"
            "Based on the following criteria, provide a Similarity Score from 0 to 100:\n\n"
            "0-20: Completely different answers with no meaningful overlap\n"
            "21-40: Minimal similarity with few shared concepts\n"
            "41-60: Moderate similarity with some shared core ideas\n"
            "61-80: Substantial similarity with minor variations\n"
            "81-100: Nearly identical in meaning (may use different words)\n\n"
            "IMPORTANT: Provide your final Similarity Score as an integer between 0 and 100, "
            "enclosed in <similarity_score></similarity_score> XML tags. For example:\n"
            "<similarity_score>75</similarity_score>\n\n"
            "Do not include any additional text in your response."
        )
        resp = await self.judge_client.chat.completions.create(
            model=self.judge_model, messages=[{"role": "user", "content": prompt}]
        )
        content = resp.choices[0].message.content or ""
        try:
            score = float(_extract_tag(content, "similarity_score"))
        except Exception:
            score = 100.0
        return score / 100.0

    async def _embed(self, text: str) -> List[float]:
        resp = await self.embed_client.embeddings.create(
            model=self.embedding_model, input=[text]
        )
        return resp.data[0].embedding  # type: ignore


def load_environment(**kwargs) -> vf.Environment:
    """Factory for vf-install / vf-eval to load the environment."""
    return AidanBenchEnv(**kwargs)
