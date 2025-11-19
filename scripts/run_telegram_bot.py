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

# scripts/run_telegram_bot.py

def handle_message(self, user_id: str, text: str) -> str:
    raw_text = text.strip()
    lower = raw_text.lower()

    # 1) Saludo simple → no hago lógica de recomendaciones
    if any(g in lower for g in ["hola", "holis", "buenas", "buen día", "buen dia", "hey", "hello"]):
        return (
            "¡Hola! 👋 Soy *MoodFlix*.\n\n"
            "Puedo recomendarte:\n"
            "• 🎬 Películas\n"
            "• 📺 Series\n"
            "• 🎧 Música\n\n"
            "Usando cómo te sentís, lo que estás haciendo (correr, estudiar, etc.) "
            "o algún género que te guste.\n\n"
        ) 
    
async def handle_text(update, context):
    user_id = str(update.effective_user.id)
    text = update.message.text
    response = bot.handle_message(user_id, text)
    await update.message.reply_text(response)

def main():
    application = ApplicationBuilder().token(settings.telegram_bot_token).build()

    application.add_handler(CommandHandler("start", handle_message))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    application.run_polling()

if __name__ == "__main__":
    logger.info("Iniciando bot MoodFlix...")
    main()
    logger.info("Bot MoodFlix detenido.")