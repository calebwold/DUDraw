from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import openai
from chromadb import Client as ChromaClient
from chromadb.utils import embedding_functions
from chromadb import PersistentClient
import os
import json

# Import the DuDraw function data
from du_draw_functions_data import DU_DRAW_FUNCTIONS
from agent_tools import TOOLS_DEFINITIONS

# Calculator function without Streamlit
def calculate_expression(expression: str) -> str:
    """Calculates the result of a mathematical expression."""
    try:
        result = eval(expression)
        return f"Calculation Result: {result}"
    except Exception as e:
        return f"Error evaluating expression '{expression}': {e}"

app = Flask(__name__, static_folder='.')
CORS(app)

# --- Configuration ---
# Get API key from environment variable for security
YOUR_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

if not YOUR_OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required. Please set it before running the application.")

# Set these environment variables for the embedding function.
os.environ["OPENAI_API_KEY"] = YOUR_OPENAI_API_KEY
os.environ["CHROMA_OPENAI_API_KEY"] = YOUR_OPENAI_API_KEY

# Specify the LLM model to use
LLM_MODEL_NAME = "gpt-4o-mini"
MAX_AGENT_STEPS = 5

# --- Tool 1: DuDraw Function Retriever ---
class DuDrawFunctionRetriever:
    """
    A tool to retrieve relevant DuDraw function information from a ChromaDB vector store.
    """
    def __init__(self):
        self.chroma_client = PersistentClient(path="./chroma_db")
        self.collection_name = "du_draw_functions_collection"

        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )

        self.collection = self.chroma_client.get_or_create_collection(
            self.collection_name,
            embedding_function=self.embedding_function
        )
        self.populate_functions()

    def populate_functions(self):
        """
        Populates the ChromaDB collection with DuDraw function data from du_draw_functions_data.py.
        Only adds data if the collection is empty to prevent duplicates on restart.
        """
        if self.collection.count() == 0:
            print(f"Loading {len(DU_DRAW_FUNCTIONS)} DuDraw functions from data file...")
            documents = []
            metadatas = []
            ids = []

            for func in DU_DRAW_FUNCTIONS:
                doc_content = f"{func['description']}. Keywords: {', '.join(func.get('keywords', []))}"
                documents.append(doc_content)

                metadata_for_chroma = func.copy()
                if 'keywords' in metadata_for_chroma and isinstance(metadata_for_chroma['keywords'], list):
                    metadata_for_chroma['keywords'] = ', '.join(metadata_for_chroma['keywords'])
                if 'params' in metadata_for_chroma and isinstance(metadata_for_chroma['params'], list):
                    metadata_for_chroma['params'] = ', '.join(metadata_for_chroma['params'])

                metadatas.append(metadata_for_chroma)
                ids.append(func["id"])

            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
            print(f"Successfully loaded {len(DU_DRAW_FUNCTIONS)} DuDraw functions into ChromaDB")
        else:
            print(f"ChromaDB already contains {self.collection.count()} DuDraw functions")

    def retrieve_functions(self, query: str, n_results: int = 6):
        """
        Retrieves the top N most relevant DuDraw function metadata based on a natural language query.
        This pulls from the DU_DRAW_FUNCTIONS data loaded from du_draw_functions_data.py
        """
        try:
            # Query the ChromaDB collection which contains all DuDraw functions from the data file
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['metadatas']
            )
            retrieved_functions = []
            if results and results['metadatas'] and results['metadatas'][0]:
                for meta in results['metadatas'][0]:
                    retrieved_functions.append(meta)
            
            formatted_retrieval = self._format_retrieved_tools_for_llm_response(retrieved_functions)
            return formatted_retrieval
        except Exception as e:
            return f"Error: Could not retrieve functions: {e}"

    def _format_retrieved_tools_for_llm_response(self, tools_list):
        """Helper to format retrieved functions for LLM consumption as observation."""
        if not tools_list:
            return "No specific DuDraw functions were found for this request."
        formatted_list = []
        for tool in tools_list:
            formatted_list.append(
                f"Function ID: {tool.get('id', 'N/A')}\n"
                f"Description: {tool.get('description', 'N/A')}\n"
                f"Syntax: {tool.get('syntax', 'N/A')}\n"
                f"Parameters: {tool.get('params', 'N/A')}\n"
                f"Example: {tool.get('example', 'N/A')}\n"
                "---"
            )
        return "\n".join(formatted_list)


# --- Main Agent Class ---
class DuDrawAgent:
    """
    The core Agentic AI system for DuDraw code generation.
    """
    def __init__(self, openai_api_key):
        if not openai_api_key:
            raise ValueError("OpenAI API Key is required for the agent to function.")
        openai.api_key = openai_api_key

        self.retriever = DuDrawFunctionRetriever()
        self.conversation_history = []
        
        # Initialize available_tools
        self.available_tools = {"calculate_expression": calculate_expression}
        self.available_tools["retrieve_dudraw_functions"] = self.retriever.retrieve_functions

    def _add_to_history(self, role: str, content: str, tool_calls=None, tool_call_id=None, name=None):
        """Adds a message to the conversation history, supporting tool calls and responses."""
        message = {"role": role}
        if content:
            message["content"] = content
        if tool_calls:
            message["tool_calls"] = tool_calls
        if tool_call_id:
            message["tool_call_id"] = tool_call_id
        if name:
            message["name"] = name
        self.conversation_history.append(message)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    def _get_system_prompt(self):
        """Defines the sophisticated system prompt for the agent's behavior."""
        return """
        You are an expert Python programmer and DuDraw code generator with comprehensive knowledge of:
        - Python programming fundamentals (variables, functions, loops, conditionals, data structures)
        - Object-oriented programming concepts
        - Code organization and best practices
        - Proper commenting and documentation
        - DuDraw library functions and syntax
        
        Your primary task is to generate clean, well-commented, production-quality DuDraw code based on a user's natural language request. The code must be executable, properly formatted, and include helpful comments explaining what each section does.

        You operate in a ReAct (Reason-Act-Observe) loop. You have access to tools to gather information about DuDraw functions.

        **Your Process:**
        1.  **Think:** Before acting, always output a 'Thought:' explaining your current reasoning, what information you need, and which tool you plan to use (if any) or why you are providing a final answer.
        2.  **Act (Tool Use):** If you need information about DuDraw functions, ALWAYS use the `retrieve_dudraw_functions` tool to get the exact syntax, parameters, and examples. This ensures you use the correct function signatures.
        3.  **Observe:** The tool's output will be provided to you with exact function syntax and examples.
        4.  **Repeat:** Continue thinking, acting, and observing until you have enough information to generate the complete DuDraw code.
        5.  **Final Answer:** When you have sufficient information, provide the DuDraw code with proper comments and formatting. Do NOT call any tools when providing the final answer.

        **Code Quality Requirements:**
        - ALWAYS include a header comment at the top explaining what the program does
        - Add comments for each major section of code
        - Comment complex logic or calculations
        - Use descriptive variable names
        - Follow Python PEP 8 style guidelines
        - Include docstrings for functions when appropriate
        - Organize code logically (imports, constants, functions, main execution)

        **Output Format for Final Answer:**
        Your final response MUST be formatted as follows:
        ```python
        # Program Description: [Brief description of what this program does]
        # Author: DuDraw Code Companion
        # Date: Generated code
        
        # Import required libraries
        import dudraw
        
        # [Your well-commented DuDraw code here]
        # Make sure every significant line or block has a comment explaining what it does
        ```
        ---
        **Explanation:**
        - Provide a detailed, step-by-step explanation of the code
        - Explain the purpose of each function or major code block
        - Describe how the code achieves the user's request
        - For games/animations, clearly describe the `initialize_game`, `update_game`, `draw_game`, and `game_loop` functions.

        **Core DuDraw Principles (Always Apply when generating code):**
        - ALWAYS use the `retrieve_dudraw_functions` tool to get the exact syntax before writing any DuDraw function calls
        - Assume a canvas size of approximately 500x500 pixels unless specified otherwise
        - Default coordinate scale: 0-1 for common shapes (e.g., 0.5, 0.5 for center). Adjust scale using `dudraw.set_x_scale` or `dudraw.set_y_scale` if a pixel-based scale is clearly implied
        - Use sensible default values for coordinates, sizes (e.g., 0.1-0.2 for shapes, 20-30 for font), and colors (e.g., `dudraw.BLACK`, `dudraw.BLUE`) if not explicitly specified
        - CRITICAL: Ensure the generated code strictly follows the **exact syntax and parameter order** from the DuDraw function specifications retrieved from the tool
        - Pay close attention to whether a function expects a `dudraw.Color` constant (e.g., `dudraw.RED`) or RGB integer values
        - Always start with `dudraw.set_canvas_size()` to set up the canvas
        - For static drawings, end with `dudraw.show(0)` or `dudraw.show(delay_ms)` to display the result
        - Use proper Python coding practices: meaningful variable names, proper indentation, clear logic flow

        **Specific Guidelines for Interactive Animations or Games (apply when relevant):**
        If the user's request implies an interactive element, animation, or a game (e.g., "move", "control", "game", "score", "collision", "input"), you MUST structure the code as follows:
        1.  **Imports:** Always `import dudraw` and `import random` (for games) and potentially `import time`.
        2.  **Configuration/Constants:** Define key game parameters as constants at the top (e.g., `CANVAS_SIZE`, `GAME_SPEED`, `GRID_SIZE`, `PLAYER_SIZE`).
        3.  **Game State Variables:** Declare global variables to manage the game's dynamic state (e.g., `player_position`, `score`, `game_over`, `snake_segments`, `food_position`, `direction`). Initialize these in an `initialize_game()` function.
        4.  **`initialize_game()` Function:** Set up `dudraw.set_canvas_size`, `dudraw.set_x_scale`, `dudraw.set_y_scale`. Initialize all game state variables. **Call `dudraw.enable_keyboard_input()` if keyboard interaction is needed.**
        5.  **`update_game()` Function (Game Logic):** Contains all game logic. Player input handling (`dudraw.has_next_key_typed()`, `dudraw.next_key_typed()`), movement, collision detection, game rules (scoring, object spawning/removal), and game over/win conditions.
        6.  **`draw_game()` Function (Rendering):** Clear canvas (`dudraw.clear()`), set pen color, draw elements. Display score/messages. **Call `dudraw.show(milliseconds_delay)` at the end of this function.**
        7.  **`game_loop()` Function (Main Loop):** Calls `initialize_game()` once, then enters a `while True` loop calling `update_game()` and `draw_game()`.
        8.  **Main Execution Block:** Use `if __name__ == '__main__':` to call `game_loop()`.
        9.  **Restart Logic:** If game ends, allow a key press (e.g., 'R') to restart by calling `initialize_game()` again.

        **Available Tools:**
        You have access to the following tools:
        - `retrieve_dudraw_functions(query: str)`: **USE THIS TOOL FREQUENTLY** to get exact information about DuDraw functions, their syntax, parameters, and examples. This tool pulls from a comprehensive database of all DuDraw functions. ALWAYS use this before writing any DuDraw function call to ensure you use the correct syntax. Examples of queries: "draw circle", "set canvas size", "keyboard input", "draw text", "set color", etc.
        - `calculate_expression(expression: str)`: Use this to perform mathematical calculations. If the user asks a question that requires arithmetic (e.g., addition, subtraction, multiplication, division) or evaluating a numerical expression, use this tool.

        **IMPORTANT REMINDERS:**
        - Before writing ANY DuDraw function, use `retrieve_dudraw_functions` to get the exact syntax
        - Always include helpful comments explaining what each part of the code does
        - Write clean, readable, professional Python code
        - Make sure the code is complete and executable
        - Include proper error handling when appropriate
        - Use meaningful variable and function names
        """

    def run_agent(self, user_goal: str):
        """Run the agent and return messages as they're generated."""
        self.conversation_history = []
        self._add_to_history("user", user_goal)
        self.conversation_history.append({"role": "system", "content": self._get_system_prompt()})

        messages = []
        messages.append({"role": "user", "content": user_goal})
        
        current_thought_displayed = False
        final_output_generated = False
        steps = 0

        while not final_output_generated and steps < MAX_AGENT_STEPS:
            steps += 1

            try:
                response = openai.chat.completions.create(
                    model=LLM_MODEL_NAME,
                    messages=self.conversation_history,
                    tools=TOOLS_DEFINITIONS,
                    tool_choice="auto",
                    temperature=0.7,
                    max_tokens=1500
                )

                response_message = response.choices[0].message
                self._add_to_history("assistant", response_message.content if response_message.content else "", response_message.tool_calls)

                if response_message.content and response_message.content.strip().startswith("Thought:"):
                    messages.append({
                        "role": "assistant",
                        "type": "thought",
                        "content": response_message.content.replace('Thought:', '').strip()
                    })
                    current_thought_displayed = True
                elif response_message.content and not response_message.tool_calls:
                    messages.append({
                        "role": "assistant",
                        "type": "final",
                        "content": response_message.content
                    })
                    final_output_generated = True
                    break

                if response_message.tool_calls:
                    tool_messages = []
                    for tool_call in response_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)

                        messages.append({
                            "role": "assistant",
                            "type": "tool_call",
                            "content": f"Looking up DuDraw functions: {function_args.get('query', 'N/A')}" if function_name == "retrieve_dudraw_functions" else f"Calculating: {function_args.get('expression', 'N/A')}",
                            "tool_name": function_name,
                            "tool_args": function_args
                        })

                        if function_name in self.available_tools:
                            tool_to_call = self.available_tools[function_name]
                            try:
                                if function_name == "calculate_expression":
                                    tool_response = tool_to_call(function_args.get("expression", ""))
                                elif function_name == "retrieve_dudraw_functions":
                                    # This pulls from the actual data file
                                    tool_response = tool_to_call(function_args.get("query", ""))
                                else:
                                    tool_response = tool_to_call(**function_args)
                                
                                tool_messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": tool_response,
                                    "name": function_name
                                })
                                
                                # Show the retrieved information
                                if function_name == "retrieve_dudraw_functions":
                                    messages.append({
                                        "role": "assistant",
                                        "type": "tool",
                                        "content": f"Found DuDraw function information:\n{tool_response[:500]}..." if len(tool_response) > 500 else f"Found DuDraw function information:\n{tool_response}"
                                    })
                            except Exception as e:
                                error_message = f"Error calling tool '{function_name}': {str(e)}"
                                tool_messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": error_message,
                                    "name": function_name
                                })
                        else:
                            error_message = f"Error: Tool '{function_name}' not found."
                            tool_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": error_message,
                                "name": function_name
                            })
                    
                    if tool_messages:
                        self.conversation_history.extend(tool_messages)

                elif not current_thought_displayed:
                    messages.append({
                        "role": "assistant",
                        "type": "thought",
                        "content": response_message.content if response_message.content else "Processing..."
                    })

            except Exception as e:
                messages.append({
                    "role": "assistant",
                    "type": "error",
                    "content": f"An error occurred: {str(e)}"
                })
                final_output_generated = True

        if not final_output_generated:
            messages.append({
                "role": "assistant",
                "type": "error",
                "content": f"Agent failed to generate a final output after {MAX_AGENT_STEPS} steps."
            })

        return messages


# Global agent instance
agent = None

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    global agent
    try:
        if agent is None:
            agent = DuDrawAgent(YOUR_OPENAI_API_KEY)
        
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        messages = agent.run_agent(user_message)
        return jsonify({"messages": messages})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    global agent
    try:
        if agent is None:
            agent = DuDrawAgent(YOUR_OPENAI_API_KEY)
        # Verify data is loaded
        function_count = agent.retriever.collection.count()
        return jsonify({
            "status": "ready",
            "functions_loaded": function_count,
            "data_source": "du_draw_functions_data.py"
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

if __name__ == '__main__':
    # For local development
    app.run(debug=True, port=5000, host='0.0.0.0')
else:
    # For production (Gunicorn)
    pass

