# gui/main_window.py
import tkinter as tk
from tkinter import ttk, messagebox, Menu
import logging
import os
import time
import threading
from datetime import datetime
from core.wallet import HDWallet, WalletManager
from core.api import NodeAPI
from core.storage import save_wallet, load_wallet
from gui.dialogs import (
    CreateWalletDialog, RestoreWalletDialog, SendDialog,
    WalletManagerDialog, ExportPrivateKeyDialog, ImportPrivateKeyDialog,
    SignMessageDialog, VerifyMessageDialog
)
from resources.style import apply_style
import qrcode
from PIL import Image, ImageTk
import io

# Настройка логирования
LOG_FILE = "wallet.log"
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("WorldKitCoin Wallet")
        self.root.geometry("950x750")
        self.current_theme = "dark"
        self.wallet_manager = WalletManager()
        self.wallet = None
        self.addresses = []
        self.history = []
        self.last_balance = 0
        self.last_history_hash = ""
        self.update_timer_id = None
        self.node_status = "unknown"  # unknown, online, offline

        self.create_widgets()
        self.refresh_wallet()
        self.start_auto_update()
        logger.info("Кошелёк запущен")

    def create_widgets(self):
        # Меню
        menubar = Menu(self.root)
        self.root.config(menu=menubar)

        wallet_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Кошелёк", menu=wallet_menu)
        wallet_menu.add_command(label="Управление кошельками", command=self.manage_wallets)
        wallet_menu.add_command(label="Создать новый", command=self.create_wallet)
        wallet_menu.add_command(label="Восстановить", command=self.restore_wallet)
        wallet_menu.add_separator()
        wallet_menu.add_command(label="Экспорт приватного ключа", command=self.export_private_key)
        wallet_menu.add_command(label="Импорт приватного ключа", command=self.import_private_key)
        wallet_menu.add_separator()
        wallet_menu.add_command(label="Подписать сообщение", command=self.sign_message)
        wallet_menu.add_command(label="Проверить подпись", command=self.verify_message)
        wallet_menu.add_separator()
        wallet_menu.add_command(label="Выход", command=self.on_exit)

        tools_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Инструменты", menu=tools_menu)
        tools_menu.add_command(label="Показать QR-код адреса", command=self.show_qr)
        tools_menu.add_command(label="Переключить тему", command=self.toggle_theme)
        tools_menu.add_command(label="Обновить баланс", command=self.update_balance)

        # Основной фрейм
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Верхняя панель: адрес и баланс
        top_frame = ttk.Frame(self.main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))

        self.addr_label = ttk.Label(top_frame, text="Адрес: ", font=("Courier", 10))
        self.addr_label.pack(anchor=tk.W)

        btn_frame = ttk.Frame(top_frame)
        btn_frame.pack(anchor=tk.W, pady=2)
        ttk.Button(btn_frame, text="Копировать", command=self.copy_address).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="QR-код", command=self.show_qr).pack(side=tk.LEFT, padx=2)

        self.balance_label = ttk.Label(top_frame, text="Баланс: 0 WKC", font=("Segoe UI", 14))
        self.balance_label.pack(anchor=tk.W, pady=5)
        ttk.Button(top_frame, text="Обновить баланс", command=self.update_balance).pack(anchor=tk.W)
        ttk.Button(top_frame, text="Отправить", command=self.send_dialog).pack(anchor=tk.W, pady=5)

        # Основной контент: адреса слева, история справа
        content_frame = ttk.Frame(self.main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Левая часть: адреса
        left_frame = ttk.LabelFrame(content_frame, text="Адреса", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.addr_listbox = tk.Listbox(left_frame, height=8, font=("Courier", 9))
        self.addr_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.addr_listbox.bind('<<ListboxSelect>>', self.on_address_select)
        ttk.Button(left_frame, text="Создать новый адрес", command=self.create_new_address).pack(anchor=tk.W)

        # Правая часть: история
        right_frame = ttk.LabelFrame(content_frame, text="История транзакций", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.history_listbox = tk.Listbox(right_frame, height=12, font=("Courier", 9))
        self.history_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        ttk.Button(right_frame, text="Обновить историю", command=self.update_history).pack(anchor=tk.W)

        # Статус бар
        self.status_var = tk.StringVar(value="Готово")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Индикатор статуса ноды (в статус-баре)
        self.node_status_label = ttk.Label(status_bar, text="⚪ Нода: проверка...", foreground="gray")
        self.node_status_label.pack(side=tk.RIGHT, padx=5)

        # Инициализация таймера автообновления
        self.start_auto_update()
        logger.info("Интерфейс создан")

    def set_status(self, msg):
        self.status_var.set(msg)
        self.root.update()
        logger.info(f"Статус: {msg}")

    def update_node_status(self):
        """Проверяет доступность ноды и обновляет индикатор"""
        try:
            import requests
            resp = requests.get("http://localhost:5000/info", timeout=2)
            if resp.status_code == 200:
                self.node_status = "online"
                self.node_status_label.config(text="🟢 Нода: онлайн", foreground="green")
            else:
                self.node_status = "offline"
                self.node_status_label.config(text="🔴 Нода: офлайн", foreground="red")
        except:
            self.node_status = "offline"
            self.node_status_label.config(text="🔴 Нода: офлайн", foreground="red")

    # ===================== АВТООБНОВЛЕНИЕ =====================
    def start_auto_update(self):
        """Запускает таймер автоматического обновления (каждые 10 секунд)"""
        if self.update_timer_id:
            self.root.after_cancel(self.update_timer_id)
        self.update_timer_id = self.root.after(10000, self.auto_update)

    def auto_update(self):
        """Автоматическое обновление баланса и истории"""
        if self.wallet:
            # Обновляем статус ноды
            self.update_node_status()
            # Получаем текущий баланс и историю
            old_balance = self.last_balance
            old_history_len = len(self.history)
            self.update_balance(silent=True)
            self.update_history(silent=True)
            # Если баланс изменился или появились новые транзакции — показываем уведомление
            if self.last_balance != old_balance:
                self.show_notification(f"Баланс изменился: {self.last_balance} WKC")
                self.last_balance = self.last_balance
            # Запускаем следующий таймер
            self.update_timer_id = self.root.after(10000, self.auto_update)

    def show_notification(self, message):
        """Показывает всплывающее уведомление (тост)"""
        # Простое решение: выводим в статус и в лог
        self.set_status(f"🔔 {message}")
        logger.info(f"Уведомление: {message}")
        # Можно добавить всплывающее окно (messagebox), но это раздражает.
        # Лучше сделать отдельный виджет, но пока так.

    # ===================== УПРАВЛЕНИЕ КОШЕЛЬКОМ =====================
    def refresh_wallet(self):
        wallet_data = self.wallet_manager.get_current_wallet()
        if wallet_data:
            mnemonic = wallet_data['mnemonic']
            self.wallet = HDWallet(mnemonic)
            self.addresses = self.wallet.get_addresses(10)
            self.update_ui()
            self.set_status(f"Активный кошелёк: {wallet_data['name']}")
            logger.info(f"Загружен кошелёк '{wallet_data['name']}', адрес: {self.wallet.get_current_address()}")
        else:
            self.create_wallet()

    def create_wallet(self):
        dialog = CreateWalletDialog(self.root)
        if dialog.result:
            mnemonic = dialog.result
            name = tk.simpledialog.askstring("Имя кошелька", "Введите имя для кошелька:", parent=self.root)
            if not name:
                name = f"Кошелёк {len(self.wallet_manager.wallets) + 1}"
            self.wallet_manager.add_wallet(name, mnemonic)
            self.refresh_wallet()
            self.set_status(f"Создан кошелёк '{name}'")
            logger.info(f"Создан новый кошелёк '{name}'")

    def restore_wallet(self):
        dialog = RestoreWalletDialog(self.root)
        if dialog.result:
            mnemonic = dialog.result
            name = tk.simpledialog.askstring("Имя кошелька", "Введите имя для кошелька:", parent=self.root)
            if not name:
                name = f"Восстановленный {len(self.wallet_manager.wallets) + 1}"
            self.wallet_manager.add_wallet(name, mnemonic)
            self.refresh_wallet()
            self.set_status(f"Восстановлен кошелёк '{name}'")
            logger.info(f"Восстановлен кошелёк '{name}'")

    def manage_wallets(self):
        WalletManagerDialog(self.root, self.wallet_manager, self.refresh_wallet)

    # ===================== КЛЮЧИ И ПОДПИСИ =====================
    def export_private_key(self):
        if self.wallet:
            ExportPrivateKeyDialog(self.root, self.wallet)

    def import_private_key(self):
        if self.wallet:
            ImportPrivateKeyDialog(self.root, self.wallet)

    def sign_message(self):
        if self.wallet:
            SignMessageDialog(self.root, self.wallet)

    def verify_message(self):
        if self.wallet:
            VerifyMessageDialog(self.root, self.wallet)

    # ===================== ОБНОВЛЕНИЕ UI =====================
    def update_ui(self):
        if not self.wallet:
            return
        addr = self.wallet.get_current_address()
        self.addr_label.config(text=f"Адрес: {addr}")
        self.update_balance()
        self.update_addresses()
        self.update_history()

    def update_balance(self, silent=False):
        if not self.wallet:
            return
        addr = self.wallet.get_current_address()
        balance = NodeAPI.get_balance(addr)
        self.last_balance = balance
        self.balance_label.config(text=f"Баланс: {balance} WKC")
        if not silent:
            self.set_status("Баланс обновлён")

    def update_addresses(self):
        self.addr_listbox.delete(0, tk.END)
        for i, addr in enumerate(self.addresses):
            self.addr_listbox.insert(tk.END, f"[{i}] {addr}")

    def update_history(self, silent=False):
        if not self.wallet:
            return
        addr = self.wallet.get_current_address()
        self.history = NodeAPI.get_history(addr)
        self.history_listbox.delete(0, tk.END)
        for tx in self.history[:20]:
            self.history_listbox.insert(tk.END, f"{tx.get('amount')} WKC  {tx.get('timestamp')}")
        if not silent:
            self.set_status("История обновлена")

    def on_address_select(self, event):
        selection = self.addr_listbox.curselection()
        if selection:
            index = selection[0]
            addr = self.addresses[index]
            self.wallet.current_index = index
            self.addr_label.config(text=f"Адрес: {addr}")
            self.update_balance()
            self.update_history()
            self.set_status(f"Выбран адрес {index}")

    def create_new_address(self):
        if not self.wallet:
            return
        addr = self.wallet.create_new_address()
        self.addresses.append(addr)
        self.update_addresses()
        self.set_status(f"Новый адрес создан: {addr}")
        logger.info(f"Создан новый адрес: {addr}")

    def copy_address(self):
        addr = self.wallet.get_current_address()
        self.root.clipboard_clear()
        self.root.clipboard_append(addr)
        self.set_status("Адрес скопирован")

    def send_dialog(self):
        if not self.wallet:
            return
        dialog = SendDialog(self.root, self.wallet, refresh_callback=self.refresh_ui)
        if dialog.result:
            self.refresh_ui()
            self.set_status("Транзакция отправлена")
            logger.info(f"Отправлена транзакция с {self.wallet.get_current_address()}")

    def refresh_ui(self):
        """Обновляет баланс и историю после отправки или других изменений"""
        self.update_balance()
        self.update_history()
        self.update_addresses()

    # ===================== QR-КОД =====================
    def show_qr(self):
        if not self.wallet:
            messagebox.showwarning("Ошибка", "Кошелёк не загружен")
            return
        addr = self.wallet.get_current_address()
        qr = qrcode.QRCode(box_size=6, border=2)
        qr.add_data(addr)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        img_tk = ImageTk.PhotoImage(img)
        qr_window = tk.Toplevel(self.root)
        qr_window.title("QR-код адреса")
        qr_window.geometry("300x350")
        label = ttk.Label(qr_window, image=img_tk)
        label.image = img_tk
        label.pack(pady=10)
        ttk.Label(qr_window, text=addr, font=("Courier", 8), wraplength=280).pack(pady=5)
        ttk.Button(qr_window, text="Закрыть", command=qr_window.destroy).pack(pady=10)

    # ===================== ТЕМА =====================
    def toggle_theme(self):
        if self.current_theme == "dark":
            self.current_theme = "light"
            apply_style(self.root, "light")
        else:
            self.current_theme = "dark"
            apply_style(self.root, "dark")
        self.set_status(f"Тема: {self.current_theme}")
        logger.info(f"Переключена тема на {self.current_theme}")

    # ===================== ЗАВЕРШЕНИЕ РАБОТЫ =====================
    def on_exit(self):
        if self.update_timer_id:
            self.root.after_cancel(self.update_timer_id)
        logger.info("Кошелёк закрыт")
        self.root.quit()