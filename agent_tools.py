# --- Tool 2: Calculator Function ---
def calculate_expression(expression: str) -> str:
    """Calculates the result of a mathematical expression.
    Args:
        expression (str): The mathematical expression to evaluate (e.g., "2 + 2", "15 * 3 / 2").
    Returns:
        str: The result of the calculation, or an error message if invalid.
    """
    try:
        # Evaluate the expression safely using eval() or a more robust parser
        result = eval(expression)
        return f"Calculation Result: {result}"
    except Exception as e:
        return f"Error evaluating expression '{expression}': {e}"

# --- All Tool Definitions (for LLM function calling) ---
TOOLS_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_dudraw_functions",
            "description": "Retrieve relevant DuDraw function information based on a natural language query. Use this tool when you need to understand specific DuDraw drawing, input, or setting functions to generate code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A concise natural language query describing the type of DuDraw functions needed (e.g., 'functions for drawing circles', 'keyboard input functions', 'setting canvas size')."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_expression",
            "description": "Calculates the result of a mathematical expression. Use this tool when you need to perform arithmetic operations or evaluate mathematical formulas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate (e.g., '2 + 2', '15 * 3 / 2', 'math.sqrt(9)')."
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# A dictionary mapping tool names to their actual Python functions
AVAILABLE_TOOL_FUNCTIONS = {
    "calculate_expression": calculate_expression
    # Note: 'retrieve_dudraw_functions' will be added in main_app.py
    # because it requires an instance of DuDrawFunctionRetriever
}