"""Simplified Execution Agent Runtime."""

import inspect
import json
from time import monotonic_ns
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .agent import ExecutionAgent
from .tools import get_tool_schemas, get_tool_registry
from ...config import get_settings
from ...services.evaluation_lab import LabToolPolicy, lab_tool_rejection
from ...services.evaluation_lab.models import TraceEventKind
from ...services.evaluation_lab.trace import emit_trace, trace_timing
from ...openrouter_client import request_chat_completion
from ...logging_config import logger


def _trace_tool_result(tool_name: str, result: Any) -> Any:
    """Keep exact generic results while reducing Gmail results to safe shape facts."""

    normalized_name = (tool_name or "").lower()
    if "gmail" not in normalized_name and tool_name != "task_email_search":
        return result
    try:
        if isinstance(result, list):
            result_count = len(result)
        elif isinstance(result, dict):
            items = result.get("items")
            result_count = len(items) if isinstance(items, list) else 0
        else:
            result_count = 0
        return {
            "result_type": type(result).__name__,
            "result_count": result_count,
        }
    except Exception:
        return {"result_type": "unavailable", "result_count": 0}


@dataclass
class ExecutionResult:
    """Result from an execution agent."""
    agent_name: str
    success: bool
    response: str
    error: Optional[str] = None
    tools_executed: List[str] = None


class ExecutionAgentRuntime:
    """Manages the execution of a single agent request."""

    MAX_TOOL_ITERATIONS = 8

    # Initialize execution agent runtime with settings, tools, and agent instance
    def __init__(
        self,
        agent_name: str,
        agent_id: Optional[str] = None,
        legacy_storage_key: Optional[str] = None,
    ):
        settings = get_settings()
        storage_key = agent_id or agent_name
        self.agent = ExecutionAgent(
            agent_name,
            storage_key=storage_key,
            agent_id=agent_id,
            legacy_storage_key=legacy_storage_key,
        )
        self.api_key = settings.openrouter_api_key
        self.model = settings.execution_agent_model
        self.lab_enabled = settings.lab_enabled
        self.tool_registry = get_tool_registry(agent_name=storage_key, settings=settings)
        self.tool_schemas = get_tool_schemas(settings=settings)

        if not self.api_key:
            raise ValueError("OpenRouter API key not configured. Set OPENROUTER_API_KEY environment variable.")

    # Main execution loop for running agent with LLM calls and tool execution
    async def execute(self, instructions: str) -> ExecutionResult:
        """Execute the agent with given instructions."""
        try:
            # Build system prompt with history
            system_prompt = self.agent.build_system_prompt_with_history()

            # Start conversation with the instruction
            messages = [{"role": "user", "content": instructions}]
            tools_executed: List[str] = []
            final_response: Optional[str] = None

            for iteration in range(self.MAX_TOOL_ITERATIONS):
                logger.info(
                    f"[{self.agent.name}] Requesting plan (iteration {iteration + 1})"
                )
                response = await self._make_llm_call(system_prompt, messages, with_tools=True)
                assistant_message = response.get("choices", [{}])[0].get("message", {})

                if not assistant_message:
                    raise RuntimeError("LLM response did not include an assistant message")

                raw_tool_calls = assistant_message.get("tool_calls", []) or []
                parsed_tool_calls = self._extract_tool_calls(raw_tool_calls)

                assistant_entry: Dict[str, Any] = {
                    "role": "assistant",
                    "content": assistant_message.get("content", "") or "",
                }
                if raw_tool_calls:
                    assistant_entry["tool_calls"] = raw_tool_calls
                messages.append(assistant_entry)

                if not parsed_tool_calls:
                    final_response = assistant_entry["content"] or "No action required."
                    break

                for tool_call in parsed_tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("arguments", {})
                    call_id = tool_call.get("id")

                    if not tool_name:
                        logger.warning("Tool call missing name: %s", tool_call)
                        failure = {"error": "Tool call missing name; unable to execute."}
                        tool_message = {
                            "role": "tool",
                            "tool_call_id": call_id or "unknown_tool",
                            "content": self._format_tool_result(
                                tool_name or "<unknown>", False, failure, tool_args
                            ),
                        }
                        messages.append(tool_message)
                        continue

                    tools_executed.append(tool_name)
                    logger.info(f"[{self.agent.name}] Executing tool: {tool_name}")

                    success, result = await self._execute_tool(tool_name, tool_args)

                    if success:
                        logger.info(f"[{self.agent.name}] Tool {tool_name} completed successfully")
                        record_payload = self._safe_json_dump(result)
                    else:
                        error_detail = result.get("error") if isinstance(result, dict) else str(result)
                        logger.warning(f"[{self.agent.name}] Tool {tool_name} failed: {error_detail}")
                        record_payload = error_detail

                    self.agent.record_tool_execution(
                        tool_name,
                        self._safe_json_dump(tool_args),
                        record_payload
                    )

                    tool_message = {
                        "role": "tool",
                        "tool_call_id": call_id or tool_name,
                        "content": self._format_tool_result(tool_name, success, result, tool_args),
                    }
                    messages.append(tool_message)

            else:
                raise RuntimeError("Reached tool iteration limit without final response")

            if final_response is None:
                raise RuntimeError("LLM did not return a final response")

            self.agent.record_response(final_response)
            emit_trace(
                TraceEventKind.FINAL_RESPONSE,
                {
                    "response": final_response,
                    "agent_name": self.agent.name,
                },
            )

            return ExecutionResult(
                agent_name=self.agent.name,
                success=True,
                response=final_response,
                tools_executed=tools_executed
            )

        except Exception as e:
            logger.error(f"[{self.agent.name}] Execution failed: {e}")
            error_msg = str(e)
            failure_text = f"Failed to complete task: {error_msg}"
            self.agent.record_response(f"Error: {error_msg}")
            emit_trace(
                TraceEventKind.ERROR,
                {
                    "boundary": "execution_runtime",
                    "error_type": type(e).__name__,
                },
            )
            emit_trace(
                TraceEventKind.FINAL_RESPONSE,
                {
                    "response": failure_text,
                    "agent_name": self.agent.name,
                },
            )

            return ExecutionResult(
                agent_name=self.agent.name,
                success=False,
                response=failure_text,
                error=error_msg
            )

    # Execute OpenRouter API call with system prompt, messages, and optional tool schemas
    async def _make_llm_call(self, system_prompt: str, messages: List[Dict], with_tools: bool) -> Dict:
        """Make an LLM call."""
        tools_to_send = self.tool_schemas if with_tools else None
        emit_trace(
            TraceEventKind.MODEL_CALL,
            {
                "runtime": "execution",
                "stage": "started",
                "model": self.model,
                "tool_schema_count": len(tools_to_send) if tools_to_send else 0,
            },
        )
        with trace_timing(monotonic_ns) as timing:
            logger.info(f"[{self.agent.name}] Calling LLM with model: {self.model}, tools: {len(tools_to_send) if tools_to_send else 0}")
            try:
                response = await request_chat_completion(
                    model=self.model,
                    messages=messages,
                    system=system_prompt,
                    api_key=self.api_key,
                    tools=tools_to_send
                )
            except Exception as exc:
                finished = timing.finish()
                emit_trace(
                    TraceEventKind.MODEL_CALL,
                    {
                        "runtime": "execution",
                        "stage": "failed",
                        "model": self.model,
                        "started_monotonic_ns": timing.started_monotonic_ns,
                        "finished_monotonic_ns": finished,
                        "elapsed_ns": timing.elapsed_ns,
                        "error_type": type(exc).__name__,
                    },
                )
                raise
            finished = timing.finish()
            emit_trace(
                TraceEventKind.MODEL_CALL,
                {
                    "runtime": "execution",
                    "stage": "completed",
                    "model": self.model,
                    "started_monotonic_ns": timing.started_monotonic_ns,
                    "finished_monotonic_ns": finished,
                    "elapsed_ns": timing.elapsed_ns,
                },
            )
            return response

    # Parse and validate tool calls from LLM response into structured format
    def _extract_tool_calls(self, raw_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract tool calls from an assistant message."""
        tool_calls: List[Dict[str, Any]] = []

        for tool in raw_tools:
            function = tool.get("function", {})
            name = function.get("name", "")
            args = function.get("arguments", "")

            if isinstance(args, str):
                try:
                    args = json.loads(args) if args else {}
                except json.JSONDecodeError:
                    args = {}

            if name:
                tool_calls.append({
                    "id": tool.get("id"),
                    "name": name,
                    "arguments": args,
                })

        return tool_calls

    # Safely convert objects to JSON with fallback to string representation
    def _safe_json_dump(self, payload: Any) -> str:
        """Serialize payload to JSON, falling back to string representation."""
        try:
            return json.dumps(payload, default=str)
        except TypeError:
            return str(payload)

    # Format tool execution results into JSON structure for LLM consumption
    def _format_tool_result(
        self,
        tool_name: str,
        success: bool,
        result: Any,
        arguments: Dict[str, Any],
    ) -> str:
        """Build a structured string for tool responses."""
        if success:
            payload: Dict[str, Any] = {
                "tool": tool_name,
                "status": "success",
                "arguments": arguments,
                "result": result,
            }
        else:
            error_detail = result.get("error") if isinstance(result, dict) else str(result)
            payload = {
                "tool": tool_name,
                "status": "error",
                "arguments": arguments,
                "error": error_detail,
            }
        return self._safe_json_dump(payload)

    # Execute tool function from registry with error handling and async support
    async def _execute_tool(self, tool_name: str, arguments: Dict) -> Tuple[bool, Any]:
        """Execute a tool. Returns (success, result)."""
        emit_trace(
            TraceEventKind.TOOL_CALL,
            {
                "runtime": "execution",
                "tool_name": tool_name,
                "stage": "started",
            },
        )
        with trace_timing(monotonic_ns) as timing:
            if getattr(self, "lab_enabled", False):
                decision = LabToolPolicy().decide_model_tool(tool_name)
                if not decision.allowed:
                    result = lab_tool_rejection(tool_name, decision)
                    finished = timing.finish()
                    emit_trace(
                        TraceEventKind.TOOL_CALL,
                        {
                            "runtime": "execution",
                            "tool_name": tool_name,
                            "stage": "rejected",
                            "success": False,
                            "result": _trace_tool_result(tool_name, result),
                            "started_monotonic_ns": timing.started_monotonic_ns,
                            "finished_monotonic_ns": finished,
                            "elapsed_ns": timing.elapsed_ns,
                        },
                    )
                    emit_trace(
                        TraceEventKind.GMAIL_EVIDENCE,
                        {
                            "boundary": "execution_runtime",
                            "operation_name": tool_name,
                            "stage": "rejected",
                            "allowed": False,
                            "policy_code": decision.code,
                            "callable_executed": False,
                        },
                    )
                    return False, result

            tool_func = self.tool_registry.get(tool_name)
            if not tool_func:
                result = {"error": f"Unknown tool: {tool_name}"}
                finished = timing.finish()
                emit_trace(
                    TraceEventKind.TOOL_CALL,
                    {
                        "runtime": "execution",
                        "tool_name": tool_name,
                        "stage": "rejected",
                        "success": False,
                        "result": _trace_tool_result(tool_name, result),
                        "started_monotonic_ns": timing.started_monotonic_ns,
                        "finished_monotonic_ns": finished,
                        "elapsed_ns": timing.elapsed_ns,
                    },
                )
                return False, result

            try:
                result = tool_func(**arguments)
                if inspect.isawaitable(result):
                    result = await result
                finished = timing.finish()
                emit_trace(
                    TraceEventKind.TOOL_CALL,
                    {
                        "runtime": "execution",
                        "tool_name": tool_name,
                        "stage": "completed",
                        "success": True,
                        "result": _trace_tool_result(tool_name, result),
                        "started_monotonic_ns": timing.started_monotonic_ns,
                        "finished_monotonic_ns": finished,
                        "elapsed_ns": timing.elapsed_ns,
                    },
                )
                return True, result
            except Exception as e:
                result = {"error": str(e)}
                finished = timing.finish()
                emit_trace(
                    TraceEventKind.TOOL_CALL,
                    {
                        "runtime": "execution",
                        "tool_name": tool_name,
                        "stage": "failed",
                        "success": False,
                        "result": _trace_tool_result(tool_name, result),
                        "started_monotonic_ns": timing.started_monotonic_ns,
                        "finished_monotonic_ns": finished,
                        "elapsed_ns": timing.elapsed_ns,
                        "error_type": type(e).__name__,
                    },
                )
                return False, result
