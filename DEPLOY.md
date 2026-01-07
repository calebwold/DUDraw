# Deployment Guide

## Recommended Setup: Netlify (Frontend) + Render (Backend)

### Step 1: Deploy Backend on Render

1. **Go to [Render.com](https://render.com)** and sign up/login
2. **Create a new Web Service:**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the `DUDraw` repository

3. **Configure the service:**
   - **Name:** `dudraw-api` (or your preferred name)
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn api:app`
   - **Plan:** Free tier is fine to start

4. **Add Environment Variable:**
   - Go to "Environment" tab
   - Add: `OPENAI_API_KEY` = `your_openai_api_key_here`

5. **Deploy:**
   - Click "Create Web Service"
   - Wait for deployment (takes 2-3 minutes)
   - Copy your service URL (e.g., `https://dudraw-api.onrender.com`)

### Step 2: Update Frontend with Backend URL

1. **Edit `index.html`:**
   - Find the line: `const RENDER_API_URL = 'https://your-app-name.onrender.com/api';`
   - Replace `your-app-name` with your actual Render service name
   - Example: `const RENDER_API_URL = 'https://dudraw-api.onrender.com/api';`

2. **Commit and push:**
   ```bash
   git add index.html
   git commit -m "Update API URL for Render backend"
   git push origin main
   ```

### Step 3: Deploy Frontend on Netlify

1. **Go to [Netlify.com](https://netlify.com)** and sign up/login
2. **Add new site:**
   - Click "Add new site" → "Import an existing project"
   - Connect your GitHub repository
   - Select the `DUDraw` repository

3. **Configure:**
   - **Build command:** (leave empty or `echo "No build needed"`)
   - **Publish directory:** `.` (root)
   - Click "Deploy site"

4. **Done!** Your site will be live at `https://your-site.netlify.app`

## Alternative: Deploy Everything on Render

If you prefer to keep everything in one place:

1. **Deploy backend on Render** (same as Step 1 above)
2. **Update `api.py` to serve static files:**
   ```python
   @app.route('/')
   def index():
       return send_from_directory('.', 'index.html')
   ```
3. **Deploy as web service** - Render will serve both frontend and backend

## Local Development

```bash
# Terminal 1: Start backend
cd final_project
export OPENAI_API_KEY=your_key_here
python api.py

# Terminal 2: Serve frontend (optional, or just open index.html)
python -m http.server 8000
```

Then open `http://localhost:8000` in your browser.

