# PI Environments Hub - Private Beta

## Overview

Welcome to the Private Beta for the **Prime Intellect Environments Hub**! Our goal with the Environments Hub is to grow a community-powered platform for aggregating and showcasing environments, both for RL training and downstream evaluation. We’d love for you to check it out.

### Motivation

There are a few inter-related issues we see with the current ecosystem for both evals and RL environments which we’re aiming to address with this release:

- Despite the rapidly growing interest in training LLMs with RL, there is currently **no established community platform** for exploring and sharing train-ready environments.
- Environment implementations are often tied to a specific training RL stack and can be difficult to adapt to a new trainer.
- Popular evaluation suites (lm_eval, lighteval, openbench, simple-evals, HELM) offer convenient entrypoints into many single-turn Q&A evals, but these suites generally **lack support for tasks which are agentic in nature** or require complex infrastructure setups (TAU-bench, TerminalBench, SWE-bench), resulting in a proliferation of independent eval repos without shared entrypoints or specs.
- RL environments and agent evals are **basically the same thing** (dataset + harness + scoring rules), but current open-source efforts generally treat them as fundamentally separate.
- Realistic agent environments can be complex pieces of software requiring **dependencies and versioning**, and are ill-served by monorepo structures for environment collections which can quickly become unmaintainable.

With the Environments Hub, we’ve built a community platform that doubles as a proper Python package registry. Environments are modules which declare dependencies in a `pyproject.toml` and are distributed as wheels. By adopting the [`verifiers`](https://github.com/willccbb/verifiers) spec, development efforts can focus on task-specific components (datasets, tools or harnesses, reward functions) and automatically leverage existing infrastructure for running evaluations or [training models with RL](https://github.com/primeintellect-ai/prime-rl). We’re excited about what we’ve built so far, but this is also just the beginning — we’re rapidly iterating on new UI components, and have a full roadmap of enhancements in the works for streamlining the experience of training and evaluation with Environments. We’d greatly appreciate your feedback about any points of friction or features you’d love to see!

### Quick Links

- [Environments Hub](https://app.primeintellect.ai/dashboard/environments)
- [prime CLI github](https://github.com/PrimeIntellect-ai/prime-cli)
- [verifiers github](https://github.com/willccbb/verifiers)
- [verifiers docs](https://verifiers.readthedocs.io/en/latest/)
- [prime-rl github](https://github.com/primeintellect-ai/prime-rl)

### Q&A

Join the PI discord to discuss, share feedback, and ask any questions :)

- [# environment-hub-beta](https://discord.gg/MfHggUJA)
- [# prime-rl](https://discord.gg/DHBY3xGZ)
- [# verifiers](https://discord.gg/vCkh3VVa)

If you don’t see the channels by default, you may need to enable them under Channels & Roles:

![image.png](image.png)

## Getting Started

### Access

Please share the email associated with your Prime Intellect account with @willccbb on X or Discord to be granted Beta access. When logged into Prime Intellect app, you should then see an Environments tab in the Explore section (bug Will if you don’t). 

![image.png](image%201.png)

From here, you should be able to explore existing Environments and add your own:

![image.png](image%202.png)

### Setting Your User ID

To interact with the Environments Hub, you’ll want to set your username in the “Profile & SSH Keys” section, which will be the handle associated with any Environments you share.

![image.png](image%203.png)

![Screenshot 2025-08-15 at 12.19.39 PM.png](Screenshot_2025-08-15_at_12.19.39_PM.png)

### Using the Environments Hub

You can interact with the Environments Hub via the `prime` CLI:

```bash
$ uv tool install prime
```

You can browse and install Environments with:

```bash
$ prime env list # shows list of available envs
$ prime env --help # see options for push, pull, install, init, etc.
```

You can initialize a template for a new Environment by running:

```bash
$ prime env init my-env-name # creates template in ./environments by default
```

This creates a template for a Python module with:

- A [README.md](http://README.md) file (displayed on the Hub)
- A `pyproject.toml` file for managing dependencies, versioning, tags, description, etc.
- A Python file containing stub code for a `load_environment` function which returns a `vf.Environment` object — this will be the entrypoint for downstream applications to use your Environment, and should be used encapsulate any necessary preprocessing, resource provisioning, exposing configurable args, etc.

### Authentication

To push Environments to the hub, you’ll need to authenticate your session with an API which has Write permissions to the Environments Hub. You can create an API key in “Settings → API Keys” in the PI app, then set it with the CLI:

![Screenshot 2025-08-15 at 23.50.06.png](Screenshot_2025-08-15_at_23.50.06.png)

```bash
$ prime config set-api-key # paste API key when prompted
```

Once you’re authenticated, the development flow will look like: 

```bash
$ prime env init my-environment
$ cd ./environments/my_environment 
$ # edit README.md / pyproject.toml / my_environment.py
$ uv pip install -e . # local install
$ uv run vf-eval my-environment # test installed env with API models
$ prime env push # uploads .whl to hub
```

To install an Environment from the Hub (via `uv pip install`), do:

```bash
$ prime env install username/my-environment 
```

To see other install/add commands (e.g. `pip install`, `uv add`), do:

```bash
$ prime env info username/my-environment
```

## Developing and Sharing Environments

You can browse the source code for a number of `verifiers` Environments in the folders linked below for inspiration — in-app viewing of source code within the Hub will also be enabled shortly.

- https://github.com/willccbb/verifiers/tree/main/environments
- [GitHub prime-rl/environments at main · PrimeIntellect-ai/prime-rl](https://github.com/PrimeIntellect-ai/prime-environments)

We’d love for you to use the Hub to share any Environments or evals you’ve already built, or are inspired to build; they can be anything ranging from simple demos to complex agentic harnesses. Please let us know if you’d like help migrating an existing environment or eval implementation to `verifiers`. We’re happy to give you some compute credits as a thank-you for trying out the platform and sharing any Environment contributions, and we’re open to discussing additional bounties for porting established/in-demand evals to the Hub.

## RL Training

### prime-rl Trainer

- https://github.com/primeintellect-ai/prime-rl

`prime-rl` includes an async GRPO trainer built with torch FSDP + vLLM, designed for scalable multi-node full-parameter training. You can use any Environment from the Hub with `prime-rl` by installing it in the same environment as `prime-rl` and setting your environment module id + args in your `orch.toml`:

```bash
[environment]
id = "my-environment"
args = {}
```

### verifiers Trainer

- https://github.com/willccbb/verifiers
- [verifiers docs](https://verifiers.readthedocs.io/en/latest/)

`verifiers` includes a lightweight async GRPO trainer built with Transformers/Accelerate/vLLM, which supports LoRA + full-parameter training, best suited for experimentation with dense models on 1-2 nodes. 

### Other Trainers

The `verifiers` Environment spec should be relatively straightforward to adopt for any trainer which can expose an OpenAI-compatible inference worker for actors. We are happy to discuss any challenges you may face in getting Environments to work with your training stack, and we welcome PRs to `verifiers` demonstrating example usage with other training stacks. 

## FAQ:

We’ll use this space to collect questions/answers which come up over the course of the Private Beta period.

- Q: Question 1
    - Answer 1