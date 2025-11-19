# scripts/run_telegram_bot.py
import logging
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters
from app import bot
from app.config import settings

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

async def start(update, context):
    await update.message.reply_text(
        "¡Hola! Soy tu bot de recomendaciones de música, pelis y series 🎧🎬\n"
        "Contame qué vas a hacer o cómo te sentís."
    )

async def handle_text(update, context):
    user_id = str(update.effective_user.id)
    text = update.message.text
    response = bot.handle_message(user_id, text)
    await update.message.reply_text(response)

def main():
    application = ApplicationBuilder().token(settings.telegram_bot_token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    application.run_polling()

if __name__ == "__main__":
    logger.info("Iniciando bot MoodFlix...")
    main()
    logger.info("Bot MoodFlix detenido.")