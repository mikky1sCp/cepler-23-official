# core/wallet.py
from .crypto import *
import hashlib
import json
import os
import threading
import time
from core.api import NodeAPI

class WalletManager:
    """Управляет несколькими кошельками (мнемониками)"""
    def __init__(self, storage_file="wallets.json"):
        self.storage_file = storage_file
        self.wallets = []  # список словарей {name, mnemonic, index}
        self.current_wallet_index = 0
        self.load()

    def load(self):
        if os.path.exists(self.storage_file):
            with open(self.storage_file, 'r') as f:
                data = json.load(f)
                self.wallets = data.get('wallets', [])
                self.current_wallet_index = data.get('current', 0)
        else:
            self.wallets = []
            self.current_wallet_index = 0

    def save(self):
        with open(self.storage_file, 'w') as f:
            json.dump({
                'wallets': self.wallets,
                'current': self.current_wallet_index
            }, f, indent=2)

    def add_wallet(self, name: str, mnemonic: str):
        for w in self.wallets:
            if w['name'] == name:
                raise ValueError(f"Кошелёк с именем '{name}' уже существует")
        self.wallets.append({'name': name, 'mnemonic': mnemonic, 'index': 0})
        self.current_wallet_index = len(self.wallets) - 1
        self.save()

    def remove_wallet(self, index: int):
        if len(self.wallets) <= 1:
            raise ValueError("Нельзя удалить единственный кошелёк")
        del self.wallets[index]
        if self.current_wallet_index >= len(self.wallets):
            self.current_wallet_index = len(self.wallets) - 1
        self.save()

    def get_current_wallet(self) -> dict:
        if self.wallets:
            return self.wallets[self.current_wallet_index]
        return None

    def get_wallet_by_index(self, index: int) -> dict:
        if 0 <= index < len(self.wallets):
            return self.wallets[index]
        return None

    def get_wallet_names(self) -> list:
        return [w['name'] for w in self.wallets]

    def set_current_index(self, index: int):
        if 0 <= index < len(self.wallets):
            self.current_wallet_index = index
            self.save()

class HDWallet:
    def __init__(self, mnemonic: str = None):
        self.mnemonic = mnemonic
        self.seed = None
        self.addresses = []
        self.public_keys = {}  # index -> pubkey_hex
        self.current_index = 0
        self._auto_update_thread = None
        self._stop_auto_update = False
        self._update_callbacks = []  # список функций, вызываемых при обновлении
        if mnemonic:
            self.load(mnemonic)

    def load(self, mnemonic: str):
        self.mnemonic = mnemonic
        self.seed = mnemonic_to_seed(mnemonic)
        self.current_index = 0
        self.addresses = self.get_addresses(10)
        # Заполняем публичные ключи
        for i in range(len(self.addresses)):
            self.public_keys[i] = self.get_public_key_hex(i)

    def get_private_key(self, index: int) -> bytes:
        master = hashlib.sha256(self.seed).digest()
        return derive_child_key(master, index)

    def get_public_key_hex(self, index: int) -> str:
        priv = self.get_private_key(index)
        pub = private_to_public(priv)
        return pub.hex()

    def get_address(self, index: int) -> str:
        priv = self.get_private_key(index)
        pub = private_to_public(priv)
        return public_to_address(pub)

    def get_addresses(self, count: int = 10) -> list:
        addrs = []
        for i in range(count):
            addrs.append(self.get_address(i))
        return addrs

    def get_current_address(self) -> str:
        return self.get_address(self.current_index)

    def create_new_address(self) -> str:
        self.current_index += 1
        addr = self.get_address(self.current_index)
        self.addresses.append(addr)
        self.public_keys[self.current_index] = self.get_public_key_hex(self.current_index)
        return addr

    def get_private_key_hex(self, index: int = None) -> str:
        if index is None:
            index = self.current_index
        return self.get_private_key(index).hex()

    def export_private_key(self, index: int = None) -> str:
        return self.get_private_key_hex(index)

    def import_private_key(self, private_key_hex: str) -> str:
        try:
            priv_bytes = bytes.fromhex(private_key_hex)
            pub = private_to_public(priv_bytes)
            addr = public_to_address(pub)
            return addr
        except:
            raise ValueError("Неверный формат приватного ключа")

    def sign_message(self, message: str, index: int = None) -> str:
        if index is None:
            index = self.current_index
        priv_hex = self.get_private_key_hex(index)
        return sign_message(priv_hex, message)

    def verify_message(self, address: str, message: str, signature_hex: str) -> bool:
        # Ищем публичный ключ по адресу
        pubkey_hex = None
        for idx, addr in enumerate(self.addresses):
            if addr == address:
                pubkey_hex = self.public_keys.get(idx)
                break
        if not pubkey_hex:
            # Если не нашли, пробуем найти по индексам (может быть адрес из другого кошелька)
            return False
        return verify_signature_with_pubkey(pubkey_hex, message, signature_hex)

    # ===================== АВТООБНОВЛЕНИЕ =====================
    def start_auto_update(self, interval=10, callback=None):
        """Запускает фоновый поток для автоматического обновления баланса и истории"""
        if callback:
            self._update_callbacks.append(callback)
        self._stop_auto_update = False
        if self._auto_update_thread is None or not self._auto_update_thread.is_alive():
            self._auto_update_thread = threading.Thread(target=self._auto_update_loop, args=(interval,), daemon=True)
            self._auto_update_thread.start()

    def stop_auto_update(self):
        self._stop_auto_update = True
        if self._auto_update_thread:
            self._auto_update_thread.join(timeout=2)

    def _auto_update_loop(self, interval):
        while not self._stop_auto_update:
            time.sleep(interval)
            try:
                # Обновляем баланс и историю для текущего адреса
                addr = self.get_current_address()
                balance = NodeAPI.get_balance(addr)
                history = NodeAPI.get_history(addr)
                # Вызываем колбэки
                for cb in self._update_callbacks:
                    try:
                        cb(balance, history)
                    except Exception as e:
                        print(f"Ошибка в колбэке обновления: {e}")
            except Exception as e:
                print(f"Ошибка автообновления: {e}")

    def get_balance(self) -> int:
        """Получает баланс через API (без кэширования)"""
        addr = self.get_current_address()
        return NodeAPI.get_balance(addr)

    def get_history(self) -> list:
        addr = self.get_current_address()
        return NodeAPI.get_history(addr)