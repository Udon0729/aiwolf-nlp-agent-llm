# aiwolf-nlp-agent-llm

[README in Japanese](/README.md)

A **self-improving** Werewolf (Mafia) AI agent for the AIWolf Competition (Natural Language Division).

## Purpose

The goal of this project is to build a Werewolf AI agent that reviews its own speech and strategy after every match and gets better as it plays more games. It is built on top of the official sample agent, runs on a local large language model (via vLLM), and tackles the following challenges.

In the competition, agents written by different participants connect one each and play against each other, rather than copies of your own agent. In such an environment—where outcomes depend heavily on the skill of others—an agent must behave coherently and stay on track. Analyzing logs from the initial implementation revealed these weaknesses:

- Weak role-specific strategy (e.g. the seer never uses its divination result, the werewolf makes losing attacks), so the agent fails to drive the game
- Speech drifts from the facts: confirmed information (alive/dead, executions, attacks, votes, divination results) is dropped, and events that never happened are talked about
- The agent is dragged along by other players' off-base statements and loses the consistency of its own claims
- It repeats empty pleasantries and fails to move the discussion forward

Equipping the agent to improve on these automatically, through play, is the central aim of this project.

## Design

The part that connects to the game server (WebSocket communication and match-progression handling) is kept exactly as in the official sample. A "growth layer" is added *inside* the agent instead. This improves the quality of speech and strategy without altering the competition-compliant connection method at all.

The growth layer covers:

- **Grounding in confirmed facts**: every turn's decision is explicitly tied to the confirmed information provided by the server (alive/dead, executions, attacks, votes, the agent's own divination results), preventing drift from reality.
- **Self-stance and critical reading**: the agent keeps its own prior claims, role, and obtained information to stay consistent, and evaluates others' statements against the confirmed facts rather than taking them at face value, so it is harder to drag off course.
- **Role-specific strategy**: standard play for each role and game size (the seer's claim and result disclosure, the werewolf's choice of attack target, the possessed's disruption, etc.) is given to the agent as reference knowledge.
- **Post-game self-review**: after each game the agent reflects on its own speech, extracts improvements, and feeds them into later matches—the self-improvement loop.

## Setup

> [!IMPORTANT]
> Python 3.11 or higher is required. Use [uv](https://docs.astral.sh/uv/) for dependency management and execution.

```bash
uv sync
cp config/.env.example config/.env
```

Prepare a config file for your preferred prompt language:

```bash
# Japanese prompts
cp config/config.jp.yml.example config/config.yml

# English prompts
cp config/config.en.yml.example config/config.yml
```

## Using a local LLM (vLLM)

The agent talks to a local LLM through an OpenAI-compatible endpoint. Serve a model with vLLM and configure `config/config.yml`:

```yaml
llm:
  type: openai
  sleep_time: 0

openai:
  model: gemma-4-31b-it          # match the vLLM served-model-name
  base_url: http://localhost:8000/v1
  temperature: 0.7
```

`config/.env` can hold any value (no auth is needed for a local LLM):

```
OPENAI_API_KEY=dummy
```

## Running

A game server ([aiwolf-nlp-server](https://github.com/aiwolfdial/aiwolf-nlp-server)) is required. With the server running, start the agents:

```bash
uv run src/main.py -c config/config.yml
```

As many agents as `agent.num` in `config/config.yml` are launched and connect to the game server. Match logs are written per agent under `log.output_dir` (default `./log`).

## Directory layout

```
src/
├─ main.py          Agent launch (process management)
├─ starter.py       Game-server connection and match-session handling
├─ agent/           Role-specific agent implementations
├─ utils/           Logging and helpers
└─ growth/          Growth layer (fact grounding, self-stance, role strategy, post-game review)
config/             Config files and prompts
```

## Others

For details on execution, settings, and the protocol, see [aiwolf-nlp-agent](https://github.com/aiwolfdial/aiwolf-nlp-agent).
