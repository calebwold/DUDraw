# DuDraw Code Companion

A chatbot-style web application for generating DuDraw code using AI.

## Setup

1. **Install dependencies:**
```bash
pip install flask flask-cors openai chromadb
```

2. **Set up your OpenAI API key:**
   - Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
   - Create a `.env` file in the project root:
   ```bash
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```
   - Or export it as an environment variable:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

3. **Run the backend API server:**
```bash
python api.py
```

The API will start on `http://localhost:5000`

4. **Open the frontend:**
   - Open `index.html` in your web browser, or serve it using a simple HTTP server:
   ```bash
   # Python 3
   python -m http.server 8000
   
   # Then open http://localhost:8000 in your browser
   ```

## Usage

1. Start the backend API (`python api.py`)
2. Open the `index.html` file in your browser
3. Type your request in the input field (e.g., "Draw a red square in the center")
4. Click "Send" or press Enter
5. The AI agent will generate DuDraw code for you

## Files

- `api.py` - Flask backend API server
- `index.html` - Frontend HTML/CSS/JavaScript application
- `du_draw_functions_data.py` - DuDraw function definitions
- `agent_tools.py` - Tool definitions for the AI agent
- `chroma_db/` - Vector database for function retrieval

## Features

- Clean chatbot interface
- Real-time code generation
- Step-by-step agent reasoning
- DU brand colors (Crimson and Gold)
- Modern Inter font
- Responsive design


