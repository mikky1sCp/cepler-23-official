# core/storage.py
import json
import os
import sys
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
from core.logger import logger

def get_data_dir():
    if getattr(sys, 'frozen', False):
        # Запущено как .exe — сохраняем в папку AppData
        data_dir = os.path.join(os.environ.get('APPDATA', os.path.expanduser('~')), 'WorldKitCoin')
    else:
        # Запущено как скрипт — сохраняем рядом с файлом
        data_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(data_dir, exist_ok=True)
    return data_dir

WALLET_FILE = os.path.join(get_data_dir(), "wallet_data.json")

def encrypt(data: str, password: str) -> str:
    key = get_data_dir(password)
    f = Fernet(key)
    return f.encrypt(data.encode()).decode()

def decrypt(encrypted: str, password: str) -> str:
    key = get_data_dir(password)
    f = Fernet(key)
    return f.decrypt(encrypted.encode()).decode()

def save_wallet(mnemonic: str, password: str):
    try:
        encrypted = encrypt(mnemonic, password)
        with open(WALLET_FILE, 'w') as f:
            json.dump({"mnemonic_encrypted": encrypted}, f)
        logger.info("Кошелёк сохранён")
    except Exception as e:
        logger.error(f"Ошибка сохранения кошелька: {e}")

def load_wallet(password: str) -> str:
    if not os.path.exists(WALLET_FILE):
        logger.warning("Файл кошелька не найден")
        return None
    try:
        with open(WALLET_FILE, 'r') as f:
            data = json.load(f)
        encrypted = data.get('mnemonic_encrypted')
        if not encrypted:
            return None
        return decrypt(encrypted, password)
    except Exception as e:
        logger.error(f"Ошибка загрузки кошелька: {e}")
        return None