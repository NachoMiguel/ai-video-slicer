# DuckDuckGo Image Search Setup

🆓 **Great News!** We've switched to DuckDuckGo for image searching - **completely FREE** and no API key required!

## Why DuckDuckGo?

- ✅ **Completely Free** - No API keys or subscription required
- ✅ **No Rate Limits** - Search as much as you need
- ✅ **Privacy Focused** - DuckDuckGo doesn't track users
- ✅ **Easy Setup** - Works out of the box
- ✅ **Reliable** - Good quality image results

## How It Works

Our AI Video Slicer now uses DuckDuckGo's image search to find celebrity photos automatically. The system:

1. 🔍 **Searches DuckDuckGo** for celebrity images based on character names and age context
2. 📥 **Downloads Images** automatically to create a face recognition database
3. 🎯 **Matches Faces** in your videos using the downloaded reference images
4. ✂️ **Creates Clips** with the identified celebrities

## Installation

No special setup required! Just make sure you have the required dependencies:

```bash
cd backend
pip install -r requirements.txt
```

The dependencies now include:
- `beautifulsoup4` - For parsing web content
- `requests` - For making HTTP requests to DuckDuckGo

## Testing

Test the DuckDuckGo image search functionality:

```bash
cd ai-video-slicer
python test_bing_api.py
```

**Note:** The test file is still named `test_bing_api.py` for compatibility, but it now tests DuckDuckGo functionality.

### Sample Test Output

```
🚀 DuckDuckGo Image Search Test Suite
============================================================
🆓 Using FREE DuckDuckGo image search - no API key needed!

🔍 Testing DuckDuckGo Image Search
==================================================
✅ No API key required - DuckDuckGo is free to use!

🎯 Testing DuckDuckGo Image Search
------------------------------
Searching for: Robert De Niro actor
Searching DuckDuckGo for images: Robert De Niro actor
DuckDuckGo returned 3 image URLs
✅ Found 3 image URLs:
  1. https://example.com/image1.jpg...
  2. https://example.com/image2.jpg...
  3. https://example.com/image3.jpg...

📥 Testing full image collection...
✅ Collected 2 images successfully!
  📁 duckduckgo_image_1.jpg: 15420 bytes
  📁 duckduckgo_image_2.jpg: 18350 bytes
```

## Integration Example

Here's how to use the DuckDuckGo image search in your code:

```python
from main import search_bing_images

# Search for celebrity images (function name kept for compatibility)
image_urls = search_bing_images("Robert De Niro young actor", 5)

print(f"Found {len(image_urls)} image URLs:")
for url in image_urls:
    print(f"  - {url}")
```

## Troubleshooting

### No Images Found
If DuckDuckGo returns no results:
- ✅ Check your internet connection
- ✅ Try different search terms
- ✅ The system will automatically fall back to mock images for testing

### Connection Issues
- ✅ Ensure you're not behind a restrictive firewall
- ✅ DuckDuckGo might temporarily rate-limit requests - try again later
- ✅ The system includes automatic fallback mechanisms

## Next Steps

1. ✅ **Run the test** to verify everything works
2. ✅ **Process your videos** - the system will automatically download celebrity images
3. ✅ **Enjoy free image search** with no API costs!

## Benefits Over Paid APIs

| Feature | DuckDuckGo (Free) | Bing API (Paid) |
|---------|-------------------|------------------|
| Cost | 🆓 FREE | 💰 Paid after free tier |
| Setup | ✅ No setup | ❌ Requires Azure account |
| API Key | ✅ Not needed | ❌ Required |
| Rate Limits | ✅ Generous | ❌ Strict limits |
| Privacy | ✅ Privacy-focused | ❌ Tracks usage |

---

**🎉 Ready to slice videos with FREE image search!** 