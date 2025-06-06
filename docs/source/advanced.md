# Advanced Topics

This guide covers advanced patterns and techniques for building sophisticated evaluation environments and training pipelines with the verifiers framework.

## Custom Environment Patterns

### Hierarchical Environments

Build environments that manage sub-environments:

```python
from verifiers.envs import MultiTurnEnv, SingleTurnEnv

class HierarchicalProblemSolver(MultiTurnEnv):
    """Solve complex problems by breaking into sub-problems."""
    
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        
        # Sub-environments for different problem types
        self.math_env = MathEnv(dataset=[])
        self.code_env = CodeEnv(dataset=[])
        self.logic_env = LogicEnv(dataset=[])
    
    def decompose_problem(self, problem):
        """Break problem into sub-problems."""
        # Use LLM to decompose
        decomposition_prompt = f"""Break this problem into sub-problems:
        
Problem: {problem}

<sub_problems>
List each sub-problem with its type (math/code/logic)
</sub_problems>"""
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": decomposition_prompt}]
        )
        
        return self.parse_sub_problems(response.choices[0].message.content)
    
    def solve_sub_problem(self, sub_problem, problem_type):
        """Route to appropriate sub-environment."""
        env_map = {
            "math": self.math_env,
            "code": self.code_env,
            "logic": self.logic_env
        }
        
        env = env_map.get(problem_type, self.math_env)
        solution, state = env.rollout(
            client=self.client,
            model=self.model,
            prompt=sub_problem,
            answer=""  # No ground truth for sub-problems
        )
        
        return solution, state
    
    def env_response(self, messages, state):
        """Orchestrate sub-problem solving."""
        last_message = messages[-1]["content"]
        
        if state.get("phase") == "decomposition":
            # Decompose the problem
            sub_problems = self.decompose_problem(state["problem"])
            state["sub_problems"] = sub_problems
            state["phase"] = "solving"
            state["current_sub"] = 0
            
            return f"I'll solve these sub-problems:\n{sub_problems}", state
        
        elif state.get("phase") == "solving":
            # Solve current sub-problem
            current = state["current_sub"]
            if current < len(state["sub_problems"]):
                sub = state["sub_problems"][current]
                solution, _ = self.solve_sub_problem(sub["problem"], sub["type"])
                
                state["solutions"].append(solution)
                state["current_sub"] += 1
                
                return f"Sub-problem {current+1} solution:\n{solution}", state
            else:
                # Combine solutions
                state["phase"] = "integration"
                return "Now I'll combine these solutions...", state
        
        return "Problem solved!", {"completed": True}
```

### Adversarial Environments

Create environments that actively challenge models:

```python
class AdversarialMathEnv(SingleTurnEnv):
    """Generate increasingly difficult problems based on errors."""
    
    def __init__(self, dataset, difficulty_model, **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        self.difficulty_model = difficulty_model
        self.error_patterns = defaultdict(list)
    
    def analyze_error(self, prompt, completion, answer):
        """Identify error patterns."""
        parsed = self.parser.parse(completion)
        
        error_type = self.classify_error(parsed.answer, answer)
        self.error_patterns[error_type].append({
            "prompt": prompt,
            "wrong_answer": parsed.answer,
            "correct_answer": answer,
            "reasoning": parsed.reasoning
        })
        
        return error_type
    
    def generate_adversarial_prompt(self, error_type):
        """Create prompt targeting specific weakness."""
        examples = self.error_patterns[error_type][-5:]  # Recent errors
        
        prompt = f"""Generate a math problem that tests the same concept as these:

{json.dumps(examples, indent=2)}

Make it slightly harder but testing the same skill."""
        
        response = self.difficulty_model.generate(prompt)
        return self.parse_generated_problem(response)
    
    def rollout(self, client, model, prompt, answer, **kwargs):
        """Standard rollout with error tracking."""
        completion, state = super().rollout(
            client, model, prompt, answer, **kwargs
        )
        
        # Analyze errors for adversarial generation
        scores = self.rubric.score_rollout_sync(
            prompt, completion, answer, state
        )
        
        if scores["reward"] < 0.5:  # Wrong answer
            error_type = self.analyze_error(prompt, completion, answer)
            state["error_type"] = error_type
        
        return completion, state
```

### Meta-Learning Environments

Environments that adapt their evaluation criteria:

```python
class MetaLearningEnv(SingleTurnEnv):
    """Environment that learns what to evaluate."""
    
    def __init__(self, dataset, meta_model, **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        self.meta_model = meta_model
        self.reward_history = []
        self.learned_criteria = []
    
    def learn_evaluation_criteria(self, n_samples=100):
        """Learn what distinguishes good from bad responses."""
        # Collect diverse responses
        good_responses = []
        bad_responses = []
        
        for sample in self.dataset[:n_samples]:
            # Generate multiple responses
            responses = []
            for temp in [0.3, 0.7, 1.0]:
                comp, _ = self.generate_completion(
                    sample["prompt"], 
                    temperature=temp
                )
                score = self.basic_scoring(comp, sample["answer"])
                responses.append((comp, score))
            
            # Sort by score
            responses.sort(key=lambda x: x[1], reverse=True)
            good_responses.append(responses[0][0])
            bad_responses.append(responses[-1][0])
        
        # Learn discriminative features
        criteria_prompt = f"""Analyze what distinguishes good from bad responses:

Good responses:
{json.dumps(good_responses[:5], indent=2)}

Bad responses:
{json.dumps(bad_responses[:5], indent=2)}

<criteria>
List specific criteria that good responses have
</criteria>"""
        
        criteria_response = self.meta_model.generate(criteria_prompt)
        self.learned_criteria = self.parse_criteria(criteria_response)
        
        # Create new reward functions
        self.update_rubric_with_learned_criteria()
    
    def create_criteria_reward_function(self, criterion):
        """Convert learned criterion to reward function."""
        def reward_func(completion, **kwargs):
            check_prompt = f"""Does this response meet the criterion: {criterion}

Response: {completion}

Answer yes (1.0) or no (0.0)."""
            
            score = self.meta_model.evaluate(check_prompt)
            return float(score)
        
        reward_func.__name__ = f"learned_{criterion[:20]}"
        return reward_func
```

## Advanced Rubric Patterns

### Compositional Rubrics

Build complex evaluations from simple components:

```python
from verifiers.rubrics import Rubric

class CompositionalRubric(Rubric):
    """Rubric with conditional and compositional logic."""
    
    def __init__(self, base_rubrics, composition_logic, **kwargs):
        super().__init__(**kwargs)
        self.base_rubrics = base_rubrics
        self.composition_logic = composition_logic
    
    def score_rollout_sync(self, prompt, completion, answer, state, **kwargs):
        """Apply compositional logic to base rubric scores."""
        # Get scores from all base rubrics
        base_scores = {}
        for name, rubric in self.base_rubrics.items():
            scores = rubric.score_rollout_sync(
                prompt, completion, answer, state, **kwargs
            )
            base_scores[name] = scores
        
        # Apply composition logic
        final_scores = self.composition_logic(base_scores, state)
        
        return final_scores

# Example usage
math_rubric = Rubric(funcs=[correct_answer], weights=[1.0])
reasoning_rubric = Rubric(funcs=[step_count, clarity], weights=[0.5, 0.5])
format_rubric = Rubric(funcs=[xml_format], weights=[1.0])

def composition_logic(base_scores, state):
    """Complex scoring logic."""
    scores = {}
    
    # Must have correct format
    if base_scores["format"]["xml_format"] < 0.5:
        scores["reward"] = 0.0
        return scores
    
    # Weight based on problem difficulty
    difficulty = state.get("difficulty", "medium")
    if difficulty == "hard":
        # Reasoning more important for hard problems
        scores["reward"] = (
            base_scores["math"]["reward"] * 0.6 +
            base_scores["reasoning"]["reward"] * 0.4
        )
    else:
        # Correctness more important for easy problems
        scores["reward"] = (
            base_scores["math"]["reward"] * 0.9 +
            base_scores["reasoning"]["reward"] * 0.1
        )
    
    # Copy individual scores
    for name, rubric_scores in base_scores.items():
        scores.update({f"{name}_{k}": v for k, v in rubric_scores.items()})
    
    return scores

compositional_rubric = CompositionalRubric(
    base_rubrics={
        "math": math_rubric,
        "reasoning": reasoning_rubric,
        "format": format_rubric
    },
    composition_logic=composition_logic
)
```

### Neural Rubrics

Use neural networks for evaluation:

```python
import torch
import torch.nn as nn

class NeuralRewardModel(nn.Module):
    """Neural network for reward prediction."""
    
    def __init__(self, encoder_model):
        super().__init__()
        self.encoder = encoder_model
        self.reward_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, prompt_ids, completion_ids):
        # Encode prompt and completion
        prompt_hidden = self.encoder(prompt_ids).last_hidden_state.mean(1)
        comp_hidden = self.encoder(completion_ids).last_hidden_state.mean(1)
        
        # Concatenate representations
        combined = torch.cat([prompt_hidden, comp_hidden], dim=1)
        
        # Predict reward
        reward = self.reward_head(combined)
        return reward

class NeuralRubric(Rubric):
    """Rubric using neural reward model."""
    
    def __init__(self, reward_model, tokenizer, device="cuda", **kwargs):
        super().__init__(**kwargs)
        self.reward_model = reward_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
    
    def neural_reward_func(self, prompt, completion, **kwargs):
        """Compute reward using neural model."""
        # Tokenize inputs
        prompt_ids = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).input_ids.to(self.device)
        
        comp_ids = self.tokenizer(
            completion,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).input_ids.to(self.device)
        
        # Get neural reward
        with torch.no_grad():
            reward = self.reward_model(prompt_ids, comp_ids)
        
        return float(reward.cpu().item())
    
    def train_reward_model(self, training_data):
        """Train the neural reward model."""
        # Prepare data
        dataset = RewardDataset(training_data, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Training loop
        optimizer = torch.optim.AdamW(self.reward_model.parameters(), lr=1e-5)
        
        for epoch in range(10):
            for batch in dataloader:
                prompts, completions, rewards = batch
                
                # Forward pass
                predicted_rewards = self.reward_model(prompts, completions)
                loss = nn.MSELoss()(predicted_rewards, rewards)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
```

## Advanced Training Techniques

### Progressive Unfreezing

Train models layer by layer:

```python
class ProgressiveUnfreezingTrainer(GRPOTrainer):
    """Gradually unfreeze model layers during training."""
    
    def __init__(self, *args, unfreeze_schedule, **kwargs):
        super().__init__(*args, **kwargs)
        self.unfreeze_schedule = unfreeze_schedule
        self.freeze_all_but_last_n_layers(0)
    
    def freeze_all_but_last_n_layers(self, n):
        """Freeze all layers except last n."""
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze last n layers
        if hasattr(self.model, 'transformer'):
            layers = self.model.transformer.h
        else:
            layers = self.model.model.layers
        
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Always keep head unfrozen
        if hasattr(self.model, 'lm_head'):
            for param in self.model.lm_head.parameters():
                param.requires_grad = True
    
    def on_epoch_begin(self, epoch):
        """Update unfreezing based on schedule."""
        for epoch_threshold, n_layers in self.unfreeze_schedule:
            if epoch >= epoch_threshold:
                self.freeze_all_but_last_n_layers(n_layers)
        
        # Log trainable parameters
        trainable_params = sum(
            p.numel() for p in self.model.parameters() 
            if p.requires_grad
        )
        print(f"Epoch {epoch}: {trainable_params:,} trainable parameters")

# Usage
trainer = ProgressiveUnfreezingTrainer(
    model=model,
    env=env,
    unfreeze_schedule=[
        (0, 2),   # First epoch: last 2 layers
        (1, 4),   # Second epoch: last 4 layers
        (2, 8),   # Third epoch: last 8 layers
        (3, -1),  # Fourth epoch: all layers
    ],
    config=config
)
```

### Mixture of Experts Training

Train specialized models for different task types:

```python
class MoEGRPOTrainer(GRPOTrainer):
    """Train mixture of experts with routing."""
    
    def __init__(self, expert_models, router_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expert_models = expert_models
        self.router_model = router_model
    
    def route_to_expert(self, prompt):
        """Determine which expert to use."""
        # Get routing logits
        routing_prompt = f"Classify this task: {prompt[:100]}..."
        route_probs = self.router_model.predict(routing_prompt)
        
        # Select expert
        expert_idx = torch.argmax(route_probs)
        return self.expert_models[expert_idx]
    
    def train_step(self, batch):
        """Train appropriate expert and router."""
        prompts = batch["prompts"]
        completions = batch["completions"]
        rewards = batch["rewards"]
        
        # Group by expert
        expert_batches = defaultdict(list)
        router_inputs = []
        router_labels = []
        
        for i, prompt in enumerate(prompts):
            # Determine best expert based on reward
            expert_rewards = []
            for expert in self.expert_models:
                with torch.no_grad():
                    expert_reward = self.evaluate_with_expert(
                        expert, prompt, completions[i]
                    )
                expert_rewards.append(expert_reward)
            
            best_expert = np.argmax(expert_rewards)
            expert_batches[best_expert].append(i)
            
            # Prepare router training data
            router_inputs.append(prompt)
            router_labels.append(best_expert)
        
        # Train each expert on its batch
        for expert_idx, indices in expert_batches.items():
            if indices:
                expert_batch = {
                    k: [v[i] for i in indices]
                    for k, v in batch.items()
                }
                self.train_expert(
                    self.expert_models[expert_idx],
                    expert_batch
                )
        
        # Train router
        self.train_router(router_inputs, router_labels)
```

### Constraint-Aware Training

Train with hard constraints:

```python
class ConstrainedGRPOTrainer(GRPOTrainer):
    """GRPO with constraint satisfaction."""
    
    def __init__(self, *args, constraints, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraints = constraints
    
    def compute_constraint_penalties(self, completions):
        """Compute penalties for constraint violations."""
        penalties = []
        
        for completion in completions:
            penalty = 0.0
            
            for constraint in self.constraints:
                if constraint["type"] == "length":
                    if len(completion) > constraint["max"]:
                        penalty += constraint["penalty"]
                
                elif constraint["type"] == "forbidden_words":
                    for word in constraint["words"]:
                        if word.lower() in completion.lower():
                            penalty += constraint["penalty"]
                
                elif constraint["type"] == "required_format":
                    if not constraint["check_func"](completion):
                        penalty += constraint["penalty"]
            
            penalties.append(penalty)
        
        return penalties
    
    def compute_rewards(self, completions, prompts):
        """Compute rewards with constraint penalties."""
        # Get base rewards
        base_rewards = super().compute_rewards(completions, prompts)
        
        # Apply constraint penalties
        penalties = self.compute_constraint_penalties(completions)
        
        # Combine
        final_rewards = [
            max(0, r - p) for r, p in zip(base_rewards, penalties)
        ]
        
        return final_rewards

# Usage
constraints = [
    {
        "type": "length",
        "max": 500,
        "penalty": 0.5
    },
    {
        "type": "forbidden_words",
        "words": ["I don't know", "I cannot"],
        "penalty": 1.0
    },
    {
        "type": "required_format",
        "check_func": lambda x: "<answer>" in x,
        "penalty": 0.8
    }
]

trainer = ConstrainedGRPOTrainer(
    model=model,
    env=env,
    constraints=constraints,
    config=config
)
```

## Performance Optimization

### Caching and Memoization

```python
from functools import lru_cache
import hashlib

class CachedEnvironment(SingleTurnEnv):
    """Environment with aggressive caching."""
    
    def __init__(self, *args, cache_size=10000, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_size = cache_size
        self.rollout_cache = {}
        self.score_cache = {}
    
    def get_cache_key(self, prompt, model_name, temperature):
        """Generate cache key for rollout."""
        key_str = f"{prompt}|{model_name}|{temperature}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @lru_cache(maxsize=1000)
    def parse_cached(self, completion):
        """Cache parsing results."""
        return self.parser.parse(completion)
    
    def rollout(self, client, model, prompt, answer, temperature=0.7, **kwargs):
        """Cached rollout execution."""
        cache_key = self.get_cache_key(prompt, model, temperature)
        
        if cache_key in self.rollout_cache:
            return self.rollout_cache[cache_key]
        
        # Execute rollout
        result = super().rollout(
            client, model, prompt, answer, 
            temperature=temperature, **kwargs
        )
        
        # Cache result
        if len(self.rollout_cache) < self.cache_size:
            self.rollout_cache[cache_key] = result
        
        return result
```

### Vectorized Operations

```python
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class VectorizedRubric(Rubric):
    """Rubric with vectorized reward computation."""
    
    def score_rollouts_vectorized(self, prompts, completions, answers, states):
        """Score multiple rollouts efficiently."""
        n_samples = len(prompts)
        n_funcs = len(self.reward_funcs)
        
        # Pre-allocate reward matrix
        reward_matrix = np.zeros((n_samples, n_funcs))
        
        # Vectorize where possible
        with ThreadPoolExecutor(max_workers=8) as executor:
            for func_idx, func in enumerate(self.reward_funcs):
                if hasattr(func, 'vectorized'):
                    # Use vectorized implementation
                    rewards = func.vectorized(
                        prompts, completions, answers, states
                    )
                    reward_matrix[:, func_idx] = rewards
                else:
                    # Parallel execution for non-vectorized
                    futures = []
                    for i in range(n_samples):
                        future = executor.submit(
                            self._call_reward_func,
                            func,
                            prompt=prompts[i],
                            completion=completions[i],
                            answer=answers[i],
                            state=states[i]
                        )
                        futures.append(future)
                    
                    # Collect results
                    for i, future in enumerate(futures):
                        reward_matrix[i, func_idx] = future.result()
        
        # Weighted aggregation
        weights = np.array(self.reward_weights)
        final_rewards = reward_matrix @ weights
        
        # Return structured results
        results = {
            func.__name__: reward_matrix[:, i].tolist()
            for i, func in enumerate(self.reward_funcs)
        }
        results["reward"] = final_rewards.tolist()
        
        return results
```

These advanced patterns enable building sophisticated evaluation and training systems that can handle complex requirements while maintaining the simplicity and modularity of the verifiers framework.