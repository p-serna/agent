import json
import logging
import re
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import openai
from dotenv import load_dotenv
from tenacity import (
    after_log,
    before_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from termcolor import colored

from src.tools import tool_names, tools

load_dotenv()

logger = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        openai_api_key: str,
        openai_base_url: str = "https://api.openai.com",
        available_tools: List[str] = tool_names,
        model: str = "gpt-4o",
        verbose: bool = True,
    ):
        self.client = openai.Client(api_key=openai_api_key, base_url=openai_base_url)
        self.openai_api_key = openai_api_key
        self.model = model

        self.tools = [tool for tool in tools if tool.name in available_tools]
        self.available_tools = available_tools
        self.verbose = verbose

        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """Create the system prompt describing available tools."""
        tools_description = "\n".join(
            f"- {tool.name}: {tool.description}" for tool in self.tools
        )

        return f"""You are a helpful AI assistant who will try to answer as accurately as possible, citing sources whenever they are available. You have access to the following tools:

{tools_description}

To use a tool, respond with a JSON object in the following format:
{{
    "thought": "your reasoning about what to do next",
    "action": "tool_name",
    "action_input": "input for the tool"
}}

If you have a final answer and don't need to use any tools, respond with:
{{
    "thought": "your reasoning",
    "final_answer": "your response to the user"
}}"""

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM's response from string to structured format."""

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            try:
                response = re.sub(r"```.*\n", "", response)
                response = re.sub(r"```", "", response)
                return json.loads(response)
            except Exception:
                pass
            # If the response isn't valid JSON, try to extract JSON from the text
            try:
                response = response.strip()
                start_idx = response.find("{")
                end_idx = response.rfind("}") + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx]
                    return json.loads(json_str)
            except Exception:
                return {
                    "thought": "Error parsing response",
                    "final_answer": "I encountered an error. Please try again.",
                }

    def run(self, query: str, max_steps: int = 5) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]

        if self.verbose:
            print(colored(f"User: {query}", "green"))
        for n_step in range(max_steps):
            if self.verbose:
                print(colored(f"Step {n_step}", "green"))
            response = self.client.chat.completions.create(
                model=self.model, messages=messages
            )

            # Parse the response
            agent_response = self._parse_llm_response(
                response.choices[0].message.content
            )
            print(agent_response)
            # Print thought in blue if present
            if self.verbose and "thought" in agent_response:
                print(colored(f"Thought: {agent_response['thought']}", "cyan"))

            # Add the assistant's response to the conversation
            messages.append(
                {"role": "assistant", "content": response.choices[0].message.content}
            )

            # If we have a final answer, return it
            if "final_answer" in agent_response:
                if self.verbose:
                    print(f"Final answer: {agent_response['final_answer']}")
                return agent_response["final_answer"]

            if self.verbose:
                print(
                    colored(
                        f"{agent_response.get('action','')}: {agent_response.get('action_input','')}",
                        "cyan",
                    )
                )

            # Otherwise, execute the tool and continue the conversation
            if "action" in agent_response:
                tool_name = agent_response["action"]
                tool_input = agent_response["action_input"]

                # Find the requested tool
                matched_tools = [t for t in self.tools if t.name == tool_name]
                if not matched_tools:
                    if self.verbose:
                        print(colored(f"Error: Tool '{tool_name}' not found.", "red"))

                    messages.append(
                        {
                            "role": "user",
                            "content": f"Error: Tool '{tool_name}' not found. Only use the available tools: {', '.join(self.available_tools)}",
                        }
                    )
                tool = matched_tools[0]

                # Execute the tool
                try:
                    tool_result = tool.func(tool_input)

                    messages.append(
                        {
                            "role": "user",
                            "content": f"Tool '{tool_name}' returned: {tool_result}",
                        }
                    )
                    if self.verbose:
                        print(colored(f"Observation:\n{tool_result}", "cyan"))
                except Exception as e:
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Tool '{tool_name}' failed with error: {str(e)}",
                        }
                    )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": f"Error: {agent_response.get('thought','No thought provided')}",
                    }
                )

        messages.append(
            {
                "role": "user",
                "content": f"Provide a final answer with the information retrieved so far.",
            }
        )
        response = self.client.chat.completions.create(
            model=self.model, messages=messages
        )
        agent_response = self._parse_llm_response(response.choices[0].message.content)
        if "final_answer" in agent_response:
            if self.verbose:
                print(f"Final answer: {agent_response['final_answer']}")
            return agent_response["final_answer"]

        return agent_response.get("thought", "No final answer provided")


class TaskType(Enum):
    SIMPLE_QUERY = "simple_query"
    MULTI_STEP = "multi_step"
    DATA_ANALYSIS = "data_analysis"
    RESEARCH = "research"


@dataclass
class TaskStep:
    id: str
    description: str
    tool_name: Optional[str] = None
    dependencies: List[str] = None
    completed: bool = False
    result: Any = None


class ImprovedAgent(Agent):
    def __init__(
        self,
        openai_api_key: str,
        openai_base_url: str = "https://api.openai.com",
        available_tools: List[str] = None,
        model: str = "gpt-4",
        verbose: bool = True,
        max_retries: int = 3,
        cache_duration: int = 3600,  # 1 hour cache
    ):
        super().__init__(
            openai_api_key, openai_base_url, available_tools, model, verbose
        )
        self.memory = deque(maxlen=100)  # Conversation memory
        self.cache = {}  # Result cache
        self.max_retries = max_retries
        self.cache_duration = cache_duration

    def _classify_task(self, query: str) -> TaskType:
        """Classify the incoming query to determine execution strategy."""
        # Implement classification logic using the LLM
        classification_prompt = f"""Classify the following query into one of these categories:
        - SIMPLE_QUERY: Direct questions requiring single tool use
        - MULTI_STEP: Complex questions requiring multiple tools
        - DATA_ANALYSIS: Queries involving data processing
        - RESEARCH: Queries requiring information gathering

        Query: {query}
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": classification_prompt}],
        )
        message = response.choices[0].message.content.strip().lower()
        for task_type in TaskType:
            if task_type.value in message:
                return TaskType(task_type.value)
        return TaskType.SIMPLE_QUERY

    def _create_task_plan(self, query: str, task_type: TaskType) -> List[TaskStep]:
        """Create a structured plan for executing the task."""
        planning_prompt = f"""Create a step-by-step plan for answering:
        Query: {query}
        Task Type: {task_type.value}
        Available Tools: {', '.join(self.available_tools)}

        Return the plan as a JSON array of steps, each with:
        - id: string
        - description: string
        - tool_name: string (optional)
        - dependencies: array of step ids (optional)
        """
        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": planning_prompt}]
        )
        plan_data = self._parse_llm_response(response.choices[0].message.content)
        return [TaskStep(**step) for step in plan_data]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        retry=retry_if_exception_type(Exception),
        before=before_log(logger, logging.DEBUG),
        after=after_log(logger, logging.DEBUG),
    )
    def _execute_tool(self, tool_name: str, tool_input: str) -> Any:
        """Execute a single tool with tenacity retry decorator."""
        tool = self.tools[tool_name]
        return tool.func(tool_input)

    def _execute_tool_with_retry(self, tool_name: str, tool_input: str) -> Any:
        """Execute a tool with retry logic and caching."""
        cache_key = f"{tool_name}:{tool_input}"
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if time.time() - cache_entry["timestamp"] < self.cache_duration:
                return cache_entry["result"]

        result = self._execute_tool(tool_name, tool_input)
        self.cache[cache_key] = {"result": result, "timestamp": time.time()}
        return result

    def _evaluate_result(self, result: str, query: str) -> Dict[str, Any]:
        """Evaluate the quality and relevance of the result."""
        eval_prompt = f"""Evaluate the following result for the query:
        Query: {query}
        Result: {result}

        Provide evaluation as JSON with:
        - relevance_score: float (0-1)
        - confidence_score: float (0-1)
        - completeness_score: float (0-1)
        - suggested_improvements: array of strings
        """
        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": eval_prompt}]
        )
        return self._parse_llm_response(response.choices[0].message.content)

    def run(self, query: str) -> Dict[str, Any]:
        """Execute the query with planning and evaluation."""
        # Classify the task
        task_type = self._classify_task(query)
        if self.verbose:
            print(colored(f"Task Type: {task_type.value}", "yellow"))

        # Create execution plan
        plan = self._create_task_plan(query, task_type)
        if self.verbose:
            print(colored(f"Plan: {plan}", "yellow"))

        # Execute plan
        for step in plan:
            if step.dependencies:
                # Check if dependencies are completed
                if not all(
                    any(s.id == dep and s.completed for s in plan)
                    for dep in step.dependencies
                ):
                    continue

            if step.tool_name:
                step.result = self._execute_tool_with_retry(
                    step.tool_name, step.description
                )
                if self.verbose:
                    print(
                        colored(
                            f"Step {step.id} ({step.description}): {step.result}",
                            "cyan",
                        )
                    )
            step.completed = True

        # Generate final answer
        final_answer = self._generate_final_answer(query, plan)
        if self.verbose:
            print(colored(f"Final Answer: {final_answer}", "yellow"))
        # Evaluate result
        evaluation = self._evaluate_result(final_answer, query)
        if self.verbose:
            print(colored(f"Evaluation: {evaluation}", "orange"))

        # Store in memory
        self.memory.append(
            {
                "query": query,
                "task_type": task_type,
                "plan": plan,
                "result": final_answer,
                "evaluation": evaluation,
            }
        )

        return {
            "answer": final_answer,
            "evaluation": evaluation,
            "task_type": task_type.value,
            "execution_plan": [
                {
                    "id": step.id,
                    "description": step.description,
                    "completed": step.completed,
                }
                for step in plan
            ],
        }

    def _generate_final_answer(self, query: str, plan: List[TaskStep]) -> str:
        """Generate final answer based on plan execution results."""
        context = "\n".join(
            f"Step {step.id} ({step.description}): {step.result}"
            for step in plan
            if step.completed and step.result
        )

        synthesis_prompt = f"""Based on the following execution results, provide a comprehensive answer to the query:
        Query: {query}

        Execution Results:
        {context}
        """

        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": synthesis_prompt}]
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    import os
    import sys

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        interactive = False
    else:
        interactive = True

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("MODEL")

    # Initialize the agent
    agent = Agent(
        openai_api_key,
        openai_base_url,
        available_tools=["DuckDuckGo search", "Fetch website"],
        model=model,
    )

    if not interactive:

        with open(file_path, "r") as file:
            task = file.read().strip()
        print(colored(f"AI Agent initialized. Running following task:{task}"))
        result = agent.run(task)
        # print(result)
        sys.exit()

    print(colored("AI Agent initialized. Type 'quit' to exit.", "green"))
    print(colored("Enter your task: ", "yellow"))

    while True:
        try:
            user_input = input(colored("> ", "green"))
            if user_input.lower() in ["quit", "exit", "q"]:
                break

            if user_input.strip():
                result = agent.run(user_input)
                print(colored("\nFinal Answer:", "yellow"), result, "\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(colored(f"Error: {str(e)}", "red"))
