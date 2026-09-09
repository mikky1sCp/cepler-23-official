# core/logger.py
import logging
import os
from datetime import datetime

LOG_FILE = "wallet.log"

def setup_logger():
    """Настраивает логирование в файл и в консоль"""
    logger = logging.getLogger('WKCWallet')
    logger.setLevel(logging.DEBUG)

    # Удаляем старые обработчики, чтобы не дублировать
    if logger.handlers:
        logger.handlers.clear()

    # Обработчик для файла
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Обработчик для консоли (только ошибки и выше)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

logger = setup_logger()