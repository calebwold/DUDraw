import openai
from chromadb import Client as ChromaClient
from chromadb.utils import embedding_functions
from chromadb import PersistentClient
import streamlit as st
import os
import json

# Import the DuDraw function data
from du_draw_functions_data import DU_DRAW_FUNCTIONS

# Import tool definitions and functions from the new file
from agent_tools import TOOLS_DEFINITIONS, calculate_expression, AVAILABLE_TOOL_FUNCTIONS

# --- Configuration ---
# Get API key from environment variable for security
YOUR_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

if not YOUR_OPENAI_API_KEY:
    st.error("OPENAI_API_KEY environment variable is required. Please set it before running the application.")
    st.stop()

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
        Populates the ChromaDB collection with DuDraw function data.
        Only adds data if the collection is empty to prevent duplicates on restart.
        """
        if self.collection.count() == 0:
            st.info("Populating DuDraw functions into ChromaDB... (This happens once)")
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
            st.success(f"Added {len(DU_DRAW_FUNCTIONS)} DuDraw functions to ChromaDB.")
        else:
            st.info("ChromaDB already contains DuDraw functions. Skipping repopulation.")

    def retrieve_functions(self, query: str, n_results: int = 6):
        """
        Retrieves the top N most relevant DuDraw function metadata based on a natural language query.
        This method is designed to be called by the agent's LLM via tool calling.
        """
        st.write(f"**Tool: DuDraw Function Retriever** - Querying with: `{query}`")
        try:
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
            st.success(f"**Observation (from DuDraw Function Retriever):**\n```\n{formatted_retrieval}\n```")
            return formatted_retrieval # Return formatted string to be passed back to LLM
        except Exception as e:
            st.error(f"Error during ChromaDB retrieval: {e}")
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
    It orchestrates tool calls, LLM reasoning, and manages transparency.
    """
    def __init__(self, openai_api_key):
        if not openai_api_key:
            raise ValueError("OpenAI API Key is required for the agent to function.")
        openai.api_key = openai_api_key

        self.retriever = DuDrawFunctionRetriever()
        self.conversation_history = []
        
        # Initialize available_tools by combining the imported ones with instance-specific ones
        self.available_tools = AVAILABLE_TOOL_FUNCTIONS.copy() # Start with tools from agent_tools.py
        self.available_tools["retrieve_dudraw_functions"] = self.retriever.retrieve_functions # Add the retriever instance's method

    def _add_to_history(self, role: str, content: str, tool_calls=None, tool_call_id=None, name=None):
        """Adds a message to the conversation history, supporting tool calls and responses."""
        message = {"role": role}
        if content:
            message["content"] = content
        if tool_calls:
            message["tool_calls"] = tool_calls
        if tool_call_id:
            message["tool_call_id"] = tool_call_id
        if name: # For tool responses
            message["name"] = name
        self.conversation_history.append(message)
        # Keep history manageable, especially for multi-turn agents
        if len(self.conversation_history) > 20: # Increased slightly due to tool messages
            self.conversation_history = self.conversation_history[-20:]


    def _get_system_prompt(self):
        """Defines the sophisticated system prompt for the agent's behavior."""
        return f"""
        You are an expert DuDraw code generator and explainer, highly proficient in creating both static drawings AND complex interactive animations/games. Your primary task is to generate concise, correct, and executable DuDraw code based on a user's natural language request. You must also provide a clear, step-by-step explanation of the generated code.

        You operate in a ReAct (Reason-Act-Observe) loop. You have access to tools to gather information.

        **Your Process:**
        1.  **Think:** Before acting, always output a 'Thought:' explaining your current reasoning, what information you need, and which tool you plan to use (if any) or why you are providing a final answer.
        2.  **Act (Tool Use):** If you need information, call one of the available tools.
        3.  **Observe:** The tool's output will be provided to you.
        4.  **Repeat:** Continue thinking, acting, and observing until you have enough information to generate the complete DuDraw code and its explanation.
        5.  **Final Answer:** When you have sufficient information, provide the DuDraw code and its explanation in the specified format. Do NOT call any tools when providing the final answer.

        **Output Format for Final Answer:**
        Your final response MUST be formatted as follows:
        ```python
        # Your DuDraw Code
        # ...
        ```
        ---
        **Explanation:**
        - Your detailed, bulleted explanation of the code, line by line or by logical blocks.
        - For games/animations, clearly describe the `initialize_game`, `update_game`, `draw_game`, and `game_loop` functions.

        **Core DuDraw Principles (Always Apply when generating code):**
        - Assume a canvas size of approximately 500x500 pixels.
        - Default coordinate scale: 0-1 for common shapes (e.g., 0.5, 0.5 for center). Adjust scale using `dudraw.set_x_scale` or `dudraw.set_y_scale` if a pixel-based scale is clearly implied (e.g., "draw at 10,10 on a 500x500 canvas" suggests a 0-500 scale).
        - Invent sensible default values for coordinates, sizes (e.g., 0.1-0.2 for shapes, 20-30 for font), and colors (e.g., `dudraw.BLACK`, `dudraw.BLUE`) if not explicitly specified by the user.
        - Ensure the generated code strictly follows the **exact syntax and parameter order** from the DuDraw function specifications. Pay close attention to whether a function expects a `dudraw.Color` constant (e.g., `dudraw.RED`) or RGB integer values.

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
        - `retrieve_dudraw_functions(query: str)`: Use this to get information about DuDraw functions, their syntax, parameters, and examples. Always use this if you need to recall how to draw something specific in DuDraw.
        - `calculate_expression(expression: str)`: Use this to perform mathematical calculations. If the user asks a question that requires arithmetic (e.g., addition, subtraction, multiplication, division) or evaluating a numerical expression, use this tool. For example, if the user asks "What is 100 divided by 4 and then draw a circle with that radius?", you should first use `calculate_expression` to get the result of the division, then use `retrieve_dudraw_functions` if you need to know how to draw a circle, and finally generate the DuDraw code.

        """

    def run_agent(self, user_goal: str):
        self.conversation_history = [] # Reset history for a new user goal
        self._add_to_history("user", user_goal)

        # Display user message in chat format
        with st.chat_message("user"):
            st.write(user_goal)

        # The initial message from the system to prime the ReAct loop
        self.conversation_history.append({"role": "system", "content": self._get_system_prompt()})

        current_thought_displayed = False
        final_output_generated = False
        steps = 0

        while not final_output_generated and steps < MAX_AGENT_STEPS:
            steps += 1

            try:
                # Call LLM
                with st.spinner("Thinking..."):
                    response = openai.chat.completions.create(
                        model=LLM_MODEL_NAME,
                        messages=self.conversation_history,
                        tools=TOOLS_DEFINITIONS, # Use the imported TOOLS_DEFINITIONS
                        tool_choice="auto", # Let the LLM decide if it wants to use a tool
                        temperature=0.7,
                        max_tokens=1500 # Increased for potentially longer thought processes and code
                    )

                response_message = response.choices[0].message
                self._add_to_history("assistant", response_message.content if response_message.content else "", response_message.tool_calls)

                # Display agent's thought if available
                if response_message.content and response_message.content.strip().startswith("Thought:"):
                    with st.chat_message("assistant"):
                        st.write("**Thinking:** " + response_message.content.replace('Thought:', '').strip())
                    current_thought_displayed = True
                elif response_message.content and not response_message.tool_calls:
                    # If it's not a tool call and contains content, it's likely the final answer
                    generated_output = response_message.content
                    with st.chat_message("assistant"):
                        st.write("**Generated DuDraw Code & Explanation:**")
                        st.code(generated_output, language='python')
                    self._add_to_history("assistant", f"Final Output:\n{generated_output}")
                    final_output_generated = True
                    break # Exit loop if final answer is generated

                # Handle tool calls
                if response_message.tool_calls:
                    tool_call_successful = False
                    tool_messages = [] # Collect tool outputs for the next turn
                    for tool_call in response_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)

                        with st.chat_message("assistant"):
                            st.write(f"**Calling Tool:** `{function_name}` with arguments: `{function_args}`")

                        if function_name in self.available_tools:
                            tool_to_call = self.available_tools[function_name]
                            tool_response = tool_to_call(**function_args)
                            tool_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_response,
                                "name": function_name
                            })
                            tool_call_successful = True
                        else:
                            error_message = f"Error: Tool '{function_name}' not found."
                            st.error(error_message)
                            tool_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": error_message,
                                "name": function_name
                            })
                    
                    # Add all tool messages to history at once
                    if tool_messages:
                        self.conversation_history.extend(tool_messages)

                    if not tool_call_successful:
                        st.error("No tool calls were successful in this step. Agent might be stuck or unable to find a suitable tool.")
                        break

                elif not current_thought_displayed: # If no tool call and no explicit thought, means LLM didn't follow prompt well
                    with st.chat_message("assistant"):
                        st.write("**Thinking:** " + (response_message.content if response_message.content else "Processing..."))

            except openai.APIError as e:
                st.error(f"An OpenAI API error occurred: {e}. Please check your API key and network connection.")
                st.error(f"Ensure your API key has access to the `{LLM_MODEL_NAME}` model.")
                self._add_to_history("assistant", f"Error: OpenAI API Error: {e}")
                final_output_generated = True # End loop on API error
            except json.JSONDecodeError as e:
                st.error(f"Error parsing tool arguments from LLM: {e}. This usually means the LLM generated malformed JSON for the tool call.")
                self._add_to_history("assistant", f"Error: Malformed tool arguments JSON: {e}")
                final_output_generated = True
            except Exception as e:
                st.error(f"An unexpected error occurred during agent's reasoning loop: {e}")
                self._add_to_history("assistant", f"Error: Unexpected Error in loop: {e}")
                final_output_generated = True

        if not final_output_generated:
            with st.chat_message("assistant"):
                st.error(f"Agent failed to generate a final output after {MAX_AGENT_STEPS} steps or encountered an unrecoverable error.")
                st.info("Please try rephrasing your request or check the agent's internal state.")


# --- Streamlit User Interface ---
st.set_page_config(page_title="DuDraw Code Companion Agent", layout="wide")

# Custom CSS for chatbot-style interface with DU branding
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
    /* Apply modern chatbot font globally */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
    }
    
    /* DU Brand Colors */
    :root {
        --du-crimson: #C8102E;
        --du-crimson-dark: #8B0000;
        --du-gold: #FFB81C;
        --du-white: #FFFFFF;
        --du-gray: #F5F5F5;
        --du-light-gray: #FAFAFA;
        --du-dark-gray: #333333;
        --du-border: #E0E0E0;
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 900px;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 12px;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
    }
    
    /* Chat message text */
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] div,
    [data-testid="stChatMessage"] span {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
        font-weight: 400;
        line-height: 1.6;
    }
    
    /* User message bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageUser"]) {
        background-color: var(--du-crimson);
        color: var(--du-white);
        margin-left: auto;
        max-width: 80%;
        border-bottom-right-radius: 4px;
    }
    
    /* Assistant message bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAssistant"]) {
        background-color: var(--du-light-gray);
        color: var(--du-dark-gray);
        margin-right: auto;
        max-width: 80%;
        border-bottom-left-radius: 4px;
        border-left: 3px solid var(--du-gold);
    }
    
    /* Button styling */
    .stButton > button {
        background-color: var(--du-crimson) !important;
        color: var(--du-white) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.01em;
        transition: all 0.2s ease !important;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: var(--du-crimson-dark) !important;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border: 1px solid var(--du-border) !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 400;
        line-height: 1.5;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--du-crimson) !important;
        box-shadow: 0 0 0 2px rgba(200, 16, 46, 0.1) !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: var(--du-white);
        border-right: 1px solid var(--du-border);
    }
    
    [data-testid="stSidebar"] .stSuccess {
        background-color: rgba(200, 16, 46, 0.1) !important;
        border-left: 3px solid var(--du-crimson) !important;
        color: var(--du-dark-gray) !important;
    }
    
    [data-testid="stSidebar"] .stInfo {
        background-color: rgba(255, 184, 28, 0.1) !important;
        border-left: 3px solid var(--du-gold) !important;
        color: var(--du-dark-gray) !important;
    }
    
    /* Code blocks */
    .stCodeBlock {
        border: 1px solid var(--du-border) !important;
        border-radius: 8px !important;
        background-color: #1e1e1e !important;
        margin: 0.5rem 0 !important;
    }
    
    /* Info/Success/Warning/Error boxes - minimal styling */
    .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-color: var(--du-crimson) transparent transparent transparent !important;
    }
    
    /* Clean header */
    .chat-header {
        text-align: center;
        padding: 1.5rem 0;
        border-bottom: 2px solid var(--du-border);
        margin-bottom: 2rem;
    }
    
    .chat-header h1 {
        color: var(--du-crimson) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
        font-weight: 600 !important;
        font-size: 1.75rem !important;
        letter-spacing: -0.02em;
        margin: 0 !important;
        border: none !important;
        padding: 0 !important;
    }
    
    .chat-header p {
        color: var(--du-dark-gray) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 400;
        margin: 0.5rem 0 0 0 !important;
        opacity: 0.7;
    }
    
    /* Sidebar font */
    [data-testid="stSidebar"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] div {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
    }
    
    /* All text elements */
    body, p, div, span, h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif !important;
    }
    
    /* Input area styling */
    .input-container {
        position: sticky;
        bottom: 0;
        background: var(--du-white);
        padding: 1.5rem 0;
        border-top: 1px solid var(--du-border);
        margin-top: 2rem;
    }
    
    /* Chat message improvements */
    [data-testid="stChatMessage"] {
        margin-bottom: 1.5rem;
    }
    
    /* Cleaner spacing */
    .main .block-container {
        padding-left: 2rem;
        padding-right: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Clean header
st.markdown("""
<div class="chat-header">
    <h1>DuDraw Code Companion</h1>
    <p>AI-powered code generation for DuDraw</p>
</div>
""", unsafe_allow_html=True)

agent = None
try:
    agent = DuDrawAgent(YOUR_OPENAI_API_KEY)
    st.sidebar.markdown("### DuDraw Agent")
    st.sidebar.markdown("---")
    st.sidebar.success("Agent initialized successfully")
    st.sidebar.info("ChromaDB database ready for function retrieval")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    This AI-powered agent helps you generate DuDraw code by understanding your natural language descriptions.
    
    **Features:**
    - Static drawings
    - Interactive games
    - Animations
    - Complex shapes
    """)

    # Chatbot-style input area
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    user_input = st.text_input(
        "Type your message",
        placeholder="Describe what you want to draw in DuDraw...",
        key="user_input_area",
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("Send", key="generate_button", use_container_width=True):
            if user_input:
                agent.run_agent(user_input)
            else:
                st.warning("Please enter a description for your DuDraw code.")
    
    st.markdown('</div>', unsafe_allow_html=True)

except ValueError as e:
    st.error(f"Initialization Error: {e}. Please ensure the hardcoded OpenAI API Key is correct.")
except Exception as e:
    st.error(f"An unexpected error occurred during agent initialization: {e}")