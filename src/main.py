"""
This script demonstrates a simple agent system using LangGraph.
It includes a manager, editor, and verifier to fix a bug in a Python function.
"""

import json
import os
import pathlib
import re
import tempfile
from enum import Enum
from typing import Any, Dict, List, TypedDict, Union

import docker
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, Graph, StateGraph

# Load environment variables from .env file
load_dotenv()


# Validate required environment variables
def validate_env_vars():
    """Validate that all required environment variables are set."""
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key",
    }

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Missing OpenAI API key")


# Initialize LLM without tracing
llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    api_key=os.getenv("OPENAI_API_KEY"),
)


# Define message types for type safety
class AgentState(TypedDict):
    """
    Represents the state of an agent in the system.
    In LangGraph, state is passed between nodes (agents) and must be immutable.
    Each agent receives this state and returns a modified copy.

    Attributes:
        messages: List of conversation messages between agents
        next_agent: Name of the next agent to handle the task (used for routing)
        task_status: Current status of the task (PLANNING, EDITING, etc.)
        code: Current state of the code being modified
    """

    messages: List[BaseMessage]
    next_agent: str
    task_status: str
    code: str


# Define possible states
class TaskStatus(str, Enum):
    """
    Enumeration of possible task states in the system.

    States:
        PLANNING: Task is being analyzed and planned
        EDITING: Code is being modified
        VERIFYING: Changes are being tested
        COMPLETE: Task has been completed
    """

    PLANNING = "PLANNING"
    EDITING = "EDITING"
    VERIFYING = "VERIFYING"
    COMPLETE = "COMPLETE"


def manager_agent(state: AgentState) -> AgentState:
    """
    Manager agent that plans and coordinates tasks.
    In LangGraph, each agent is a node in the graph that:
    1. Receives the current state
    2. Performs its task
    3. Returns a new state with next_agent set for routing
    """
    messages = state["messages"]

    system_prompt = SystemMessage(
        content="""You are the Manager agent in a Python code repair system.
        Your role is to:
        1. Review the issue description and verifier results
        2. Send instructions to the Editor agent if fixes are needed
        3. Send "COMPLETE" if all verifier tests have passed

        Rules:
        - Be concise in your responses
        - Only consider Python code
        - Send "COMPLETE" immediately if you see all tests have passed
        - Otherwise, give clear, focused instructions to the Editor

        Example response for failed tests:
        "Fix the addition operation in the add function by using + instead of *"

        Example response for passed tests:
        "COMPLETE - All tests have passed"
    """
    )
    llm_messages = [system_prompt, *messages]
    response = llm.invoke(llm_messages)
    new_messages = messages + [response]

    print(f"\nManager Agent:")
    print(f"Response: {response.content}\n")

    if "COMPLETE" in response.content:
        return {
            "messages": new_messages,
            "next_agent": "END",
            "task_status": TaskStatus.COMPLETE,
            "code": state["code"],
        }

    return {
        "messages": new_messages,
        "next_agent": "editor",
        "task_status": TaskStatus.EDITING,
        "code": state["code"],
    }


def editor_agent(state: AgentState) -> AgentState:
    """Editor agent that implements code changes."""
    messages = state["messages"]

    system_prompt = SystemMessage(
        content="""You are the Editor agent in a Python code repair system.
        Your role is to:
        1. Implement the exact changes requested by the Manager
        2. Return the complete fixed code in a Python code block
        3. Add a brief explanation of changes made

        Rules:
        - Only write Python code
        - Be concise in your explanations
        - Always include the complete function in a code block
        - Make minimal necessary changes

        Example response:
        "Fixed the addition operator.
        ```python
        def add(a, b):
            return a + b
        ```"
    """
    )
    llm_messages = [
        system_prompt,
        *messages,
        HumanMessage(content=f"Current code:\n```python\n{state['code']}\n```"),
    ]

    response = llm.invoke(llm_messages)

    # Extract new code from response
    code_match = re.search(r"```python\n(.*?)\n```", response.content, re.DOTALL)
    new_code = code_match.group(1) if code_match else state["code"]

    print(f"\nEditor Agent:")
    print(f"Response: {response.content}")
    print(f"New Code:\n{new_code}\n")

    return {
        "messages": messages + [response],
        "next_agent": "verifier",
        "task_status": TaskStatus.VERIFYING,
        "code": new_code,
    }


def _generate_test_cases(
    code: str, context: str, llm_obj: ChatOpenAI
) -> List[Dict[str, Any]]:
    """Generate test cases using LLM."""
    test_prompt = SystemMessage(
        content="""You are the Test Generator part of the Verifier agent.
            Given the code and context, generate comprehensive test cases.
            
            Return your response as a JSON array of test cases, where each test case has:
            - input_args: Array of input values
            - expected: Expected output
            - description: Brief description of what the test case verifies
            
            Example format:
            ```json
            [
                {
                    "input_args": [2, 3],
                    "expected": 5,
                    "description": "Basic addition of positive numbers"
                }
            ]
            ```
            """
    )

    test_messages = [
        test_prompt,
        HumanMessage(
            content=f"Code to test:\n```python\n{code}\n```\n\nContext: {context}"
        ),
    ]

    test_response = llm_obj.invoke(test_messages)

    # Extract test cases from response
    json_match = re.search(r"```json\n(.*?)\n```", test_response.content, re.DOTALL)
    if not json_match:
        json_match = re.search(r"\[(.*?)\]", test_response.content, re.DOTALL)

    try:
        return json.loads(json_match.group(1) if json_match else "[]")
    except (json.JSONDecodeError, AttributeError) as exc:
        raise ValueError("Failed to generate valid test cases") from exc


def _run_docker_tests(
    code: str, test_cases: List[Dict[str, Any]], test_runner_path: pathlib.Path
) -> List[Dict[str, Any]]:
    """Run tests in Docker container."""
    with (
        tempfile.NamedTemporaryFile(mode="w", suffix=".py") as code_file,
        tempfile.NamedTemporaryFile(mode="w", suffix=".json") as test_file,
    ):
        # Write code and test cases to temporary files
        code_file.write(code)
        code_file.flush()

        json.dump(test_cases, test_file)
        test_file.flush()

        try:
            # Create Docker client with optional environment configuration
            docker_client = docker.from_env(
                environment={
                    "DOCKER_HOST": os.getenv(
                        "DOCKER_HOST", "unix:///var/run/docker.sock"
                    )
                },
                timeout=60,
            )

            # Run tests in Docker container
            container = docker_client.containers.run(
                "python:3.9-slim",
                command=f"python /app/test_runner.py /app/code.py /app/tests.json",
                volumes={
                    str(test_runner_path): {
                        "bind": "/app/test_runner.py",
                        "mode": "ro",
                    },
                    code_file.name: {"bind": "/app/code.py", "mode": "ro"},
                    test_file.name: {"bind": "/app/tests.json", "mode": "ro"},
                },
                remove=True,
                stdout=True,
                stderr=True,
            )

            return json.loads(container.decode("utf-8"))

        except (docker.errors.ContainerError, docker.errors.ImageNotFound) as exc:
            return [{"error": f"Docker execution failed: {str(exc)}", "passed": False}]
        except Exception as exc:
            return [{"error": f"Unexpected error: {str(exc)}", "passed": False}]


def _format_verification_message(test_results: List[Dict[str, Any]]) -> str:
    """Format test results into a readable message."""
    verification_msg = "Code Verification Results:\n"
    all_passed = all(t.get("passed", False) for t in test_results)
    verification_msg += f"Overall Status: {'PASSED' if all_passed else 'FAILED'}\n\n"

    for test in test_results:
        verification_msg += f"Test: {test.get('description', 'Unnamed test')}\n"
        if "error" in test:
            verification_msg += f"  Result: ERROR - {test['error']}\n"
        else:
            verification_msg += f"  Input: {test['input']}\n"
            verification_msg += (
                f"  Result: {'PASSED' if test['passed'] else 'FAILED'}\n"
            )
            verification_msg += (
                f"  Got: {test['output']}, Expected: {test['expected']}\n"
            )
        verification_msg += "\n"

    return verification_msg


def verifier_agent(state: AgentState) -> AgentState:
    """Verifier agent that tests the code changes."""
    messages = state["messages"]
    code = state["code"]

    try:
        test_cases = _generate_test_cases(code, messages[-1].content, llm)
        test_runner_path = pathlib.Path(__file__).parent / "test_runner.py"
        test_results = _run_docker_tests(code, test_cases, test_runner_path)
        verification_msg = _format_verification_message(test_results)
        response = AIMessage(content=verification_msg)

        print(f"\nVerifier Agent:")
        print(f"Test Results: {verification_msg}\n")

        return {
            "messages": messages + [response],
            "next_agent": "manager",
            "task_status": TaskStatus.PLANNING,
            "code": code,
        }

    except Exception as e:
        error_msg = f"Verification failed: {str(e)}"
        print(f"\nVerifier Agent Error: {error_msg}\n")
        response = AIMessage(content=error_msg)
        return {
            "messages": messages + [response],
            "next_agent": "manager",
            "task_status": TaskStatus.PLANNING,
            "code": code,
        }


def _route_next(state: AgentState) -> Union[str, END]:
    """
    Routing function for LangGraph edges.
    LangGraph uses this to determine which node to visit next.

    Returns:
        - A node name (str) to continue execution
        - END to terminate the graph execution
    """
    if state["next_agent"] == "END":
        return END
    return state["next_agent"]


def create_agent_graph() -> Graph:
    """
    Creates the LangGraph workflow graph.

    A LangGraph consists of:
    1. Nodes (agents) that process the state
    2. Edges that define the flow between nodes
    3. A routing function that determines the next node
    4. Entry and exit points
    """
    # Initialize a new graph with our state type
    workflow = StateGraph(AgentState)

    # Add agent functions as nodes
    workflow.add_node("manager", manager_agent)
    workflow.add_node("editor", editor_agent)
    workflow.add_node("verifier", verifier_agent)

    # Add edges with routing logic
    # Each edge uses _route_next to determine the next node
    workflow.add_conditional_edges("manager", _route_next)
    workflow.add_conditional_edges("editor", _route_next)
    workflow.add_conditional_edges("verifier", _route_next)

    # Define the starting node
    workflow.set_entry_point("manager")

    # Compile the graph into an executable form
    return workflow.compile()


def main():
    """Main function to run the agent system."""
    try:
        # Create and compile the agent graph
        graph = create_agent_graph()

        # Define the initial code to fix
        buggy_code = """def add(a, b):
        # This function should add two numbers but has a bug
        return a * b  # Bug: using multiplication instead of addition
        """

        # Create initial state for the graph
        # This state will be passed to the first node (manager)
        initial_state = {
            "messages": [
                HumanMessage(
                    content="""There's a bug in the add() function. 
                It's supposed to add two numbers but it's performing multiplication instead. 
                Please fix this issue."""
                )
            ],
            "next_agent": "manager",  # First node to visit
            "task_status": TaskStatus.PLANNING,
            "code": buggy_code,
        }

        # Stream execution of the graph
        # Each iteration represents one node's execution
        for output in graph.stream(initial_state):
            if "manager" in output:  # Check manager's output
                if output["manager"]["next_agent"] == "END":
                    print("\nTask completed!")
                    break

    except Exception as exc:
        print(f"\nError: {str(exc)}")
        raise


if __name__ == "__main__":
    main()
