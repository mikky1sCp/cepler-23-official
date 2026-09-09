# resources/locales.py
import json
import os

LOCALES_DIR = os.path.join(os.path.dirname(__file__), "locales")

# Загружаем переводы
_translations = {}

def load_locale(lang="ru"):
    global _translations
    file_path = os.path.join(LOCALES_DIR, f"{lang}.json")
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            _translations = json.load(f)
    else:
        # fallback
        _translations = {}

def get_text(key, **kwargs):
    text = _translations.get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except:
            pass
    return text

# Предустановленные переводы
# Если файлы не существуют, создадим их при первом вызове
def init_locales():
    os.makedirs(LOCALES_DIR, exist_ok=True)
    # Русский
    ru_file = os.path.join(LOCALES_DIR, "ru.json")
    if not os.path.exists(ru_file):
        ru_data = {
            "app_title": "WorldKitCoin Кошелёк",
            "menu_wallet": "Кошелёк",
            "menu_create": "Создать новый",
            "menu_restore": "Восстановить",
            "menu_manage": "Управление кошельками",
            "menu_export_private": "Экспорт приватного ключа",
            "menu_import_private": "Импорт приватного ключа",
            "menu_sign": "Подписать сообщение",
            "menu_verify": "Проверить подпись",
            "menu_tools": "Инструменты",
            "menu_qr": "Показать QR-код адреса",
            "menu_theme": "Переключить тему",
            "menu_language": "Язык",
            "menu_exit": "Выход",
            "address": "Адрес",
            "copy": "Копировать",
            "qr": "QR-код",
            "balance": "Баланс",
            "update_balance": "Обновить баланс",
            "send": "Отправить",
            "addresses": "Адреса",
            "create_new_address": "Создать новый адрес",
            "history": "История транзакций",
            "update_history": "Обновить историю",
            "status_ready": "Готово",
            "status_balance_updated": "Баланс обновлён",
            "status_history_updated": "История обновлена",
            "status_address_copied": "Адрес скопирован",
            "status_transaction_sent": "Транзакция отправлена",
            "status_wallet_loaded": "Активный кошелёк: {name}",
            "status_wallet_created": "Создан кошелёк '{name}'",
            "status_wallet_restored": "Восстановлен кошелёк '{name}'",
            "status_new_address": "Новый адрес создан: {addr}",
            "status_address_selected": "Выбран адрес {index}",
            "error_no_wallet": "Кошелёк не загружен",
            "error_invalid_address": "Неверный адрес (должен быть 40 hex)",
            "error_amount": "Сумма должна быть целым числом",
            "error_positive": "Сумма должна быть положительной",
            "confirm_send": "Отправить {amount} WKC на адрес {to}?",
            "send_success": "Транзакция отправлена! Хэш: {tx_hash}",
            "send_fail": "Не удалось отправить: {error}",
            "qr_title": "QR-код адреса",
            "theme_switched": "Тема: {theme}",
            "language_switched": "Язык: {lang}",
            "wallet_manager_title": "Управление кошельками",
            "wallet_manager_list": "Список кошельков",
            "wallet_manager_create": "Создать новый",
            "wallet_manager_delete": "Удалить",
            "wallet_manager_activate": "Сделать активным",
            "wallet_manager_close": "Закрыть",
            "create_wallet_title": "Создание кошелька",
            "create_wallet_mnemonic": "Ваша мнемоническая фраза:",
            "create_wallet_save": "Сохраните эти слова в надёжном месте!",
            "create_wallet_confirm": "Я сохранил, продолжить",
            "restore_wallet_title": "Восстановление кошелька",
            "restore_wallet_prompt": "Введите мнемоническую фразу (12 слов):",
            "restore_wallet_button": "Восстановить",
            "send_title": "Отправить WKC",
            "send_to": "Кому (адрес 40 hex):",
            "send_amount": "Сумма (WKC):",
            "send_button": "Отправить",
            "export_private_title": "Экспорт приватного ключа",
            "export_private_label": "Приватный ключ для текущего адреса:",
            "export_private_warning": "Этот ключ позволяет управлять средствами! Храните его в безопасности.",
            "export_private_copy": "Копировать",
            "import_private_title": "Импорт приватного ключа",
            "import_private_prompt": "Введите приватный ключ (hex):",
            "import_private_button": "Импортировать",
            "import_private_success": "Ключ импортирован. Адрес: {addr}",
            "sign_title": "Подпись сообщения",
            "sign_prompt": "Введите сообщение для подписи:",
            "sign_button": "Подписать",
            "sign_copy": "Копировать подпись",
            "verify_title": "Верификация сообщения",
            "verify_address": "Адрес:",
            "verify_message": "Сообщение:",
            "verify_signature": "Подпись (hex):",
            "verify_button": "Проверить",
            "verify_result_invalid": "❌ Подпись недействительна",
            "verify_result_valid": "✅ Подпись действительна"
        }
        with open(ru_file, 'w', encoding='utf-8') as f:
            json.dump(ru_data, f, ensure_ascii=False, indent=2)

    # Английский
    en_file = os.path.join(LOCALES_DIR, "en.json")
    if not os.path.exists(en_file):
        en_data = {k: v for k, v in ru_data.items()}  # пока копия
        # Можно вручную перевести ключевые фразы
        en_data.update({
            "app_title": "WorldKitCoin Wallet",
            "menu_wallet": "Wallet",
            "menu_create": "Create new",
            "menu_restore": "Restore",
            "menu_manage": "Manage wallets",
            "menu_export_private": "Export private key",
            "menu_import_private": "Import private key",
            "menu_sign": "Sign message",
            "menu_verify": "Verify message",
            "menu_tools": "Tools",
            "menu_qr": "Show address QR code",
            "menu_theme": "Toggle theme",
            "menu_language": "Language",
            "menu_exit": "Exit",
            "status_wallet_loaded": "Active wallet: {name}",
            "status_wallet_created": "Wallet '{name}' created",
            "status_wallet_restored": "Wallet '{name}' restored",
            "status_new_address": "New address created: {addr}",
            "status_address_selected": "Address {index} selected",
            "error_no_wallet": "No wallet loaded",
            "error_invalid_address": "Invalid address (must be 40 hex)",
            "send_title": "Send WKC",
            "send_to": "To (address 40 hex):",
            "send_amount": "Amount (WKC):",
            "send_button": "Send",
            "confirm_send": "Send {amount} WKC to {to}?",
            "send_success": "Transaction sent! Hash: {tx_hash}",
            "send_fail": "Failed to send: {error}",
            "create_wallet_title": "Create wallet",
            "create_wallet_mnemonic": "Your mnemonic phrase:",
            "create_wallet_save": "Save these words in a safe place!",
            "create_wallet_confirm": "I saved, continue",
            "restore_wallet_title": "Restore wallet",
            "restore_wallet_prompt": "Enter mnemonic phrase (12 words):",
            "restore_wallet_button": "Restore",
            "export_private_title": "Export private key",
            "import_private_title": "Import private key",
            "sign_title": "Sign message",
            "verify_title": "Verify message",
            "theme_switched": "Theme: {theme}",
            "language_switched": "Language: {lang}",
            "qr_title": "Address QR code"
        })
        with open(en_file, 'w', encoding='utf-8') as f:
            json.dump(en_data, f, ensure_ascii=False, indent=2)

    load_locale("ru")  # по умолчанию русский

init_locales()