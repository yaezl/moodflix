# app/__init__.py
from .config import settings
from .chat import RecommenderChatBot

bot = RecommenderChatBot(settings=settings)
