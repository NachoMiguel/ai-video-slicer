# Google API Setup for Real Data Testing

## Required Credentials

Create a `.env` file in the project root with these variables:

```bash
GOOGLE_API_KEY=your_actual_api_key_here
GOOGLE_CSE_ID=your_actual_search_engine_id_here
```

## Step 1: Get Google API Key

1. **Go to Google Cloud Console**: https://console.cloud.google.com/
2. **Create/Select Project**: Create a new project or select an existing one
3. **Enable Custom Search API**:
   - Go to "APIs & Services" → "Library"
   - Search for "Custom Search API" 
   - Click on it and press "Enable"
4. **Create API Key**:
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "API Key"
   - Copy the generated API key
   - (Optional) Restrict the key to only Custom Search API for security

## Step 2: Create Custom Search Engine

1. **Go to Custom Search**: https://cse.google.com/cse/
2. **Create New Search Engine**:
   - Click "Add" or "New search engine"
   - In "Sites to search", enter `*` (asterisk to search entire web)
   - Give it a name like "Celebrity Image Search"
   - Click "Create"
3. **Configure Search Engine**:
   - Click "Control Panel" for your new search engine
   - Go to "Setup" → "Basic"
   - Turn **ON** "Image search"
   - Turn **ON** "Search the entire web"
   - Copy the "Search engine ID" (looks like: `017576662512468239146:omuauf_lfve`)

## Step 3: Create .env File

Create a file named `.env` (exactly, no extension) in your project root:

```bash
# .env file content
GOOGLE_API_KEY=AIzaSyA1234567890abcdefghijklmnopqrstuvwxyz
GOOGLE_CSE_ID=017576662512468239146:omuauf_lfve
```

Replace with your actual values!

## Step 4: Test the Setup

Run the real-world test:

```bash
python test_phase_b_real.py
```

**With Google API working**, you should see:
```
✓ Found 8 image URLs from Google Custom Search
✓ Downloaded real celebrity photos from live search results
```

**Without Google API**, you see:
```
Web search failed - using curated celebrity database
Using curated database for: leonardo dicaprio
```

## Verification

The code in `backend/main.py` automatically loads these environment variables:

```python
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
```

If both are set, it uses live Google search. If not, it falls back to the curated database.

## Cost Information

- **Google Custom Search API**: Free tier includes 100 searches per day
- **Additional searches**: $5 per 1000 queries
- **For testing**: 100 free searches per day is plenty

## Security Note

- Add `.env` to your `.gitignore` file (already done in this project)
- Never commit API keys to version control
- Consider using environment variables in production instead of .env files 