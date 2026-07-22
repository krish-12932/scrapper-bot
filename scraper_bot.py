import os
import re
import io
import time
import logging
import threading
import urllib.request
import httpx
from PIL import Image
from flask import Flask
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load env variables - force load from THIS folder only
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path, override=True)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

# -------------------------------------------------------------
# FLASK WEB SERVER & PINGER FOR RENDER
# -------------------------------------------------------------
flask_app = Flask(__name__)

@flask_app.route('/')
def home():
    return "Instagram Scraper Bot is running on Render!"

def run_flask():
    port = int(os.environ.get("PORT", 8080))
    flask_app.run(host="0.0.0.0", port=port)

def keep_awake_pinger():
    """Pings its own Render URL every 10 minutes so it never sleeps."""
    my_url = os.environ.get("RENDER_EXTERNAL_URL")
    if not my_url:
        logger.warning("No RENDER_EXTERNAL_URL found. Auto-ping disabled.")
        return
    while True:
        try:
            time.sleep(10 * 60)
            logger.info(f"🔄 Auto-Ping: Pinging {my_url} to stay awake...")
            urllib.request.urlopen(my_url)
        except Exception as e:
            logger.error(f"❌ Auto-Ping Failed: {e}")
# -------------------------------------------------------------

# -------------------------------------------------------------
# 4K AI UPSCALER (via Hugging Face Spaces API)
# -------------------------------------------------------------
HF_API_URL = os.getenv("HF_API_URL")

from gradio_client import Client, handle_file
import tempfile

async def upscale_image_ai(image_bytes: bytes) -> bytes:
    """
    Sends image to Hugging Face Gradio API for RealESRGAN 4K upscaling.
    Returns the upscaled image bytes.
    """
    if not HF_API_URL:
        raise ValueError("HF_API_URL is not set in .env")
        
    try:
        # Save bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_in:
            temp_in.write(image_bytes)
            temp_in_path = temp_in.name
            
        # Increase timeout to 5 minutes (300 seconds) because 4K AI takes time
        # gradio_client automatically picks up HF_TOKEN from environment
        client = Client(HF_API_URL, httpx_kwargs={"timeout": 300.0})
        
        result_path = None
        for attempt in range(10):
            try:
                import asyncio
                # Run the blocking gradio client in a separate thread!
                result_path = await asyncio.to_thread(
                    client.predict,
                    handle_file(temp_in_path)
                )
                break  # Success
            except Exception as e:
                if "No GPU was available" in str(e) and attempt < 9:
                    logger.warning(f"GPU busy, retrying... (Attempt {attempt+1}/10)")
                    import asyncio
                    await asyncio.sleep(10)
                else:
                    raise e
                    
        if not result_path:
            raise Exception("Failed to process image after 3 attempts.")
            
        # Read the resulting image bytes
        with open(result_path, "rb") as f:
            out_bytes = f.read()
            
        # Clean up
        os.remove(temp_in_path)
        
        logger.info("✅ AI Upscaling via Hugging Face Gradio successful!")
        return out_bytes
        
    except Exception as e:
        logger.error(f"Gradio API Error: {e}")
        raise Exception(f"HF API failed: {e}")
# -------------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    welcome_msg = (
        "🤖 **Welcome to the Scraping Bot!**\n\n"
        "Send me an Instagram Reel or Post link, and I will download it for you."
    )
    await update.message.reply_text(welcome_msg, parse_mode="Markdown")

async def handle_instagram_link(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Detects Instagram links, fetches media via RapidAPI, and sends it to user."""
    text = update.message.text
    
    # Regex to find Instagram URLs
    urls = re.findall(r'(https?://(?:www\.)?instagram\.com/[^\s]+)', text)
    
    if not urls:
        await update.message.reply_text("❌ No valid Instagram link found.")
        return
        
    url = urls[0]  # Take the first link
    
    # Clean the URL - strip query params like ?img_index=3&igsh=... that cause API errors
    # Keep only the base path: https://www.instagram.com/p/SHORTCODE/ or /reel/SHORTCODE/
    from urllib.parse import urlparse, urlunparse
    parsed = urlparse(url)
    # Remove query string and fragment, keep only scheme+netloc+path
    clean_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip('/') + '/', '', '', ''))
    logger.info(f"Original URL: {url} → Cleaned URL: {clean_url}")
    
    msg = await update.message.reply_text("⏳ Processing Instagram link...")
    
    try:
        api_url = "https://instagram-reels-downloader-api.p.rapidapi.com/download"
        querystring = {"url": clean_url}
        headers = {
            "x-rapidapi-key": RAPIDAPI_KEY,
            "x-rapidapi-host": "instagram-reels-downloader-api.p.rapidapi.com",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(api_url, headers=headers, params=querystring)
            
        if response.status_code != 200:
            logger.error(f"API Error: {response.status_code} - {response.text}")
            await msg.edit_text(f"❌ API Error: Failed to fetch media (Status {response.status_code}).")
            return
            
        data = response.json()
        logger.info(f"API Response keys: {list(data.get('data', {}).keys()) if data.get('data') else data}")
        
        inner = data.get("data", {})
        medias = inner.get("medias", [])

        
        # Collect all downloadable media (skip audio-only)
        all_media = [
            m for m in medias
            if m.get("url") and m.get("type") in ("video", "image") and not m.get("is_audio")
        ]
        
        if not all_media:
            # Last resort fallback
            fallback_url = inner.get("videoUrl") or inner.get("displayUrl")
            if fallback_url and "instagram.com/reel" not in fallback_url:
                all_media = [{"url": fallback_url, "type": "video" if "video" in fallback_url.lower() else "image"}]
        
        if not all_media:
            await msg.edit_text("❌ Could not extract any downloadable media from this post.")
            return
        
        total = len(all_media)
        caption = "📥 Downloaded via Scraper Bot"
        
        await msg.edit_text(f"📤 Sending {total} media item(s) to you...")
        
        async def send_item(item, cap=None):
            """Download, upscale (if image), and send as document."""
            if item["type"] == "video":
                # Videos: send directly via URL
                try:
                    await update.message.reply_video(video=item["url"], caption=cap)
                except Exception as e:
                    logger.warning(f"Video URL send failed, trying bytes: {e}")
                    async with httpx.AsyncClient(follow_redirects=True, timeout=180.0) as client:
                        r = await client.get(item["url"])
                        if r.status_code == 200:
                            await update.message.reply_video(video=r.content, caption=cap)
            else:
                # Images: send directly as a document without AI enhancement
                try:
                    await update.message.reply_document(document=item["url"], filename="image.jpg", caption=cap)
                except Exception as e:
                    logger.warning(f"Document send failed: {e}")
        if total == 1:
            await send_item(all_media[0], caption)
        else:
            # Multiple media - send one by one
            await msg.edit_text(f"✨ Found {len(all_media)} media items. Sending them one by one...")
            
            for idx, item in enumerate(all_media):
                cap = caption if idx == 0 else None
                if item["type"] == "video":
                    try:
                        await update.message.reply_video(video=item["url"], caption=cap)
                    except Exception as e:
                        logger.error(f"Failed to send video {idx}: {e}")
                else:
                    # Send image directly as document
                    try:
                        await update.message.reply_document(document=item["url"], filename=f"image_{idx+1}.jpg", caption=cap)
                    except Exception as e:
                        logger.error(f"Failed to send image {idx+1}: {e}")
        
        await msg.delete()
        
    except Exception as e:
        logger.error(f"Error handling IG link: {e}")
        await msg.edit_text(f"❌ An error occurred:\n`{e}`", parse_mode="Markdown")

async def handle_direct_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles direct image uploads sent to the bot."""
    msg = await update.message.reply_text("✨ AI is enhancing your image to 4K... (Please wait)")
    try:
        # Get the highest resolution photo, or document
        if update.message.photo:
            file_id = update.message.photo[-1].file_id
        else:
            file_id = update.message.document.file_id
            
        file = await context.bot.get_file(file_id)
        
        # Download image bytes directly using httpx
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.get(file.file_path)
            
        if r.status_code == 200:
            image_bytes = r.content
            upscaled_bytes = await upscale_image_ai(image_bytes)
            
            # Delete the "Please wait" message now that we have the result
            await msg.delete()
            
            await update.message.reply_document(
                document=upscaled_bytes,
                filename="4k_enhanced.png",
                caption="📥 4K Enhanced · Direct Upload"
            )
        else:
            await msg.edit_text("⚠️ Failed to download your image.")
    except Exception as e:
        logger.error(f"Failed to enhance direct image: {e}")
        await msg.edit_text(f"⚠️ AI Enhancer failed or is busy (Error: {e}). Please try again.")

def main():
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "WAITING_FOR_TOKEN":
        logger.error("TELEGRAM_BOT_TOKEN is missing or not set!")
        return
        
    logger.info("🚀 Starting Scraping Bot...")
    
    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .read_timeout(120)
        .write_timeout(120)
        .connect_timeout(120)
        .pool_timeout(120)
        .build()
    )
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex(r'instagram\.com'), handle_instagram_link))
    application.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, handle_direct_image))
    
    logger.info("🎧 Listening for Instagram links...")

    # Start Flask Web Server (for Render health check)
    threading.Thread(target=run_flask, daemon=True).start()

    # Start Keep-Awake Pinger (prevents Render free tier sleep)
    threading.Thread(target=keep_awake_pinger, daemon=True).start()

    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
