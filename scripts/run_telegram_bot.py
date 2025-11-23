import logging

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    filters,
)

from app.config import settings
from app.chat import ChatManager

# --------------------------------
# Logging básico
# --------------------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logger = logging.getLogger("moodflix")

# --------------------------------
# Instancia global del ChatManager
# --------------------------------
chat = ChatManager()


# --------------------------------
# Handlers
# --------------------------------

async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Maneja el comando /start y /reset.
    Le pasamos el texto '/start' al ChatManager
    para que use su propia lógica de bienvenida.
    """
    user_id = str(update.effective_user.id)
    logger.info("📲 /start de user_id=%s", user_id)

    response = chat.handle_message(user_id, "/start")
    await update.message.reply_text(response, parse_mode="Markdown")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Maneja cualquier mensaje de texto que NO sea comando.
    """
    user_id = str(update.effective_user.id)
    text = update.message.text or ""
    logger.info("📩 Mensaje de %s: %s", user_id, text)

    response = chat.handle_message(user_id, text)
    await update.message.reply_text(response, parse_mode="Markdown")


# --------------------------------
# Función principal
# --------------------------------

def main() -> None:
    logger.info("🚀 Iniciando bot MoodFlix...")

    application = ApplicationBuilder().token(settings.telegram_bot_token).build()

    # Comandos
    application.add_handler(CommandHandler("start", handle_start))
    application.add_handler(CommandHandler("reset", handle_start))
    application.add_handler(CommandHandler("reiniciar", handle_start))

    # Mensajes de texto "normales"
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)
    )

    logger.info("🤖 Bot MoodFlix listo. Esperando mensajes...")
    application.run_polling()


if __name__ == "__main__":
    main()
