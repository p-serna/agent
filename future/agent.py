import json
from typing import Any, Dict, List

import openai
from dotenv import load_dotenv
from termcolor import colored

from future.tools import tool_names, tools

load_dotenv()


class Agent:
    def __init__(
        self,
        openai_api_key: str,
        available_tools: List[str] = tool_names,
        verbose: bool = True,
    ):
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key

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
            response = openai.chat.completions.create(model="gpt-4o", messages=messages)

            # Parse the response
            agent_response = self._parse_llm_response(
                response.choices[0].message.content
            )

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
        response = openai.chat.completions.create(model="gpt-4o", messages=messages)
        agent_response = self._parse_llm_response(response.choices[0].message.content)
        if "final_answer" in agent_response:
            if self.verbose:
                print(f"Final answer: {agent_response['final_answer']}")
            return agent_response["final_answer"]

        return agent_response.get("thought", "No final answer provided")


if __name__ == "__main__":
    import os
    import sys

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        interactive = False
    else:
        interactive = True

    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Initialize the agent
    agent = Agent(openai_api_key, available_tools=["Google search", "Fetch website"])

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
