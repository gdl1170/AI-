"""
Agent Executor — ciclo di esecuzione agente con tool calling.
Ispirato a opencode, Claude Code, ChatGPT Code Interpreter.

Architettura:
1. Riceve un prompt + contesto (history, agent config)
2. Chiama il provider LLM (locale o online)
3. Se il provider ritorna tool_calls, li esegue
4. Invia i risultati al LLM per elaborazione
5. Ripete fino a risposta finale o max_iterations
"""

import json
import re
import time
import logging
from typing import Any, Generator

from .tools import execute_tool_call, list_tools, ToolResult

log = logging.getLogger("ai-plus.agent")

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)
TOOL_RESULT_RE = re.compile(
    r"<tool_result>\s*(\{.*?\})\s*</tool_result>",
    re.DOTALL,
)

MAX_ITERATIONS = 15
MAX_TOOL_OUTPUT_CHARS = 4000

AGENT_SYSTEM_PROMPT = """You are AI+, an AI coding assistant running in a terminal. You help users write, debug, and understand code, manage projects, and research technical topics.

You have access to tools that let you interact with the user's system:
{tools_description}

## How to use tools

To use a tool, output a JSON block inside <tool_call> tags:
<tool_call>
{{"name": "tool_name", "arguments": {{"key": "value"}}}}
</tool_call>

The tool will be executed and you'll receive the result in <tool_result> tags.

Example:
User: "Read the file main.py and tell me what it does"
Assistant: Let me read the file first.
<tool_call>
{{"name": "read", "arguments": {{"path": "main.py"}}}}
</tool_call>

## Guidelines
- You can use multiple tools in sequence to solve complex problems.
- When reading files, prefer reading specific sections with offset/limit for large files.
- For writing code, use the write or edit tool.
- Run commands with the run tool to test code or execute scripts.
- Search the web when you need current information.
- Be concise and direct in your responses.
- When you're done, just respond with your final answer (no tool_call tags).
- If a tool returns an error, try a different approach."""


def format_tools_for_prompt() -> str:
    tools = list_tools()
    lines = []
    for t in tools:
        params = t.get("parameters", {})
        props = params.get("properties", {})
        req = params.get("required", [])
        param_lines = []
        for name, info in props.items():
            req_mark = " (required)" if name in req else ""
            param_lines.append(f"  - {name}: {info.get('description', '')}{req_mark}")
        params_str = "\n".join(param_lines) if param_lines else "  (no parameters)"
        lines.append(f"\n📌 {t['name']}: {t['description']}\n{params_str}")
    return "\n".join(lines)


def build_agent_messages(
    prompt: str,
    history: list[dict] | None = None,
    system_prompt: str | None = None,
    agent_config: dict | None = None,
) -> list[dict]:
    messages = []

    tools_desc = format_tools_for_prompt()
    has_tools = agent_config and agent_config.get("permissions", {}).get("bash", False)

    if system_prompt:
        content = system_prompt
        if has_tools and "{tools_description}" not in content:
            content += f"\n\nAvailable tools:\n{tools_desc}"
        elif "{tools_description}" in content:
            content = content.replace("{tools_description}", tools_desc)
        messages.append({"role": "system", "content": content})
    elif agent_config:
        sys_content = agent_config.get("instructions", "") or AGENT_SYSTEM_PROMPT
        if "{tools_description}" in sys_content:
            sys_content = sys_content.replace("{tools_description}", tools_desc)
        elif has_tools or agent_config.get("tools", []):
            sys_content += f"\n\nAvailable tools:\n{tools_desc}"
        messages.append({"role": "system", "content": sys_content})
    else:
        messages.append({"role": "system", "content": AGENT_SYSTEM_PROMPT.replace("{tools_description}", tools_desc)})

    if history:
        for msg in history[-20:]:
            if msg["role"] in ("user", "assistant"):
                messages.append(msg)

    messages.append({"role": "user", "content": prompt})
    return messages


def parse_tool_calls(text: str) -> list[dict]:
    calls = []
    for match in TOOL_CALL_RE.finditer(text):
        try:
            call = json.loads(match.group(1))
            if "name" in call:
                calls.append(call)
        except json.JSONDecodeError:
            continue
    return calls


def format_tool_result(name: str, result: ToolResult) -> str:
    output = result.output[:MAX_TOOL_OUTPUT_CHARS] if result.output else ""
    error = result.error[:1000] if result.error else ""
    duration = round(result.duration, 2)

    parts = []
    if output:
        parts.append(f"Output:\n{output}")
    if error:
        parts.append(f"Error: {error}")
    if not output and not error and result.success:
        parts.append("(completed successfully)")

    summary = f"Tool '{name}' completed in {duration}s"
    if not result.success:
        summary = f"Tool '{name}' failed after {duration}s"

    body = "\n".join(parts)
    data = json.dumps({
        "name": name,
        "success": result.success,
        "duration": duration,
        "summary": summary,
        "output": output[:2000],
        "error": error[:500],
    })
    return f"""<tool_result>
{data}
</tool_result>

{summary}:

{body}"""


def execute_agent_turn(
    provider,
    messages: list[dict],
) -> Generator[dict, None, str]:
    """
    Single turn of the agent loop.
    Yields events: token, tool_call, tool_result, done, error
    Returns the final response text.
    """
    full_text = ""
    current_tool_calls = []
    in_tool_block = False

    try:
        for chunk in provider.generate_chat_stream(messages):
            if chunk.get("done"):
                result = chunk["result"]
                full_text = result.text
                yield {"type": "done", "text": result.text, "source": result.source,
                       "tokens": result.tokens_total, "time_s": result.time_s,
                       "model": result.model, "cached": result.cached}
                return result.text

            token = chunk.get("token", "")
            if token:
                full_text += token
                # Check if token contains tool_call markers
                if "<tool_call>" in full_text and not in_tool_block:
                    in_tool_block = True
                if "</tool_call>" in full_text and in_tool_block:
                    in_tool_block = False
                    # Parse and execute tool calls
                    calls = parse_tool_calls(full_text)
                    for call in calls:
                        yield {"type": "tool_call", "call": call}
                        tool_result = execute_tool_call(call)
                        yield {"type": "tool_result", "call": call, "result": tool_result["result"]}
                        # Append tool result to messages for next iteration
                        formatted = format_tool_result(call["name"], ToolResult(**tool_result["result"]))
                        messages.append({"role": "user", "content": formatted})
                    if calls:
                        # Clear tool call block and recurse
                        full_text = ""
                        yield from execute_agent_turn(provider, messages)
                        return ""

                yield {"type": "token", "text": token}

    except Exception as e:
        log.error(f"Agent execution error: {e}")
        yield {"type": "error", "message": str(e)}
        return f"[ERRORE] {e}"

    return full_text


class AgentExecutor:
    """
    Ciclo completo di esecuzione agente.
    """

    def __init__(self, provider, config: dict | None = None):
        self.provider = provider
        self.config = config or {}
        self.max_iterations = self.config.get("max_iterations", MAX_ITERATIONS)

    def execute(
        self,
        prompt: str,
        history: list[dict] | None = None,
        system_prompt: str | None = None,
        agent_config: dict | None = None,
    ) -> Generator[dict, None, str]:
        messages = build_agent_messages(prompt, history, system_prompt, agent_config)
        iteration = 0
        full_response = ""

        while iteration < self.max_iterations:
            iteration += 1
            log.info(f"Agent iteration {iteration}/{self.max_iterations}")

            result_text = ""
            for event in execute_agent_turn(self.provider, messages):
                if event["type"] == "done":
                    result_text = event.get("text", "")
                    full_response += result_text
                    yield event
                elif event["type"] == "token":
                    yield event
                elif event["type"] == "tool_call":
                    yield event
                elif event["type"] == "tool_result":
                    yield event
                elif event["type"] == "error":
                    yield event

            if not result_text:
                break
            if not parse_tool_calls(full_response):
                break

        if iteration >= self.max_iterations:
            yield {"type": "info", "message": f"Raggiunto limite di {self.max_iterations} iterazioni"}

        return full_response
