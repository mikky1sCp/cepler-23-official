# gui/dialogs.py
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from core.crypto import generate_mnemonic, sign_message, verify_signature_with_pubkey
from core.api import NodeAPI
from core.wallet import WalletManager

# ===================== МУЛЬТИЯЗЫЧНОСТЬ =====================
LANG = {
    'ru': {
        'title_create': 'Создание кошелька',
        'mnemonic_label': 'Ваша мнемоническая фраза:',
        'save_warning': 'Сохраните эти слова в надёжном месте!',
        'confirm_btn': 'Я сохранил, продолжить',
        'title_restore': 'Восстановление кошелька',
        'enter_mnemonic': 'Введите мнемоническую фразу (12 слов):',
        'restore_btn': 'Восстановить',
        'error_12_words': 'Мнемоника должна содержать 12 слов',
        'title_send': 'Отправить WKC',
        'to_label': 'Кому (адрес 40 hex):',
        'amount_label': 'Сумма (WKC):',
        'send_btn': 'Отправить',
        'confirm_send': 'Подтверждение',
        'send_confirm_text': 'Отправить {amount} WKC на адрес {to}?',
        'send_success': 'Транзакция отправлена!',
        'send_fail': 'Не удалось отправить: {error}',
        'title_manage': 'Управление кошельками',
        'wallet_list': 'Список кошельков',
        'btn_create': 'Создать новый',
        'btn_delete': 'Удалить',
        'btn_activate': 'Сделать активным',
        'btn_close': 'Закрыть',
        'new_wallet_name': 'Введите имя кошелька:',
        'delete_confirm': 'Удалить кошелёк "{name}"?',
        'cannot_delete_last': 'Нельзя удалить единственный кошелёк',
        'wallet_exists': 'Кошелёк с именем "{name}" уже существует',
        'activated': 'Кошелёк "{name}" стал активным',
        'title_export': 'Экспорт приватного ключа',
        'privkey_label': 'Приватный ключ для текущего адреса:',
        'privkey_warning': 'Этот ключ позволяет управлять средствами! Храните его в безопасности.',
        'copy_btn': 'Копировать',
        'copied': 'Приватный ключ скопирован',
        'title_import': 'Импорт приватного ключа',
        'enter_privkey': 'Введите приватный ключ (hex):',
        'import_btn': 'Импортировать',
        'import_success': 'Ключ импортирован. Адрес: {addr}',
        'title_sign': 'Подпись сообщения',
        'msg_label': 'Введите сообщение для подписи:',
        'sign_btn': 'Подписать',
        'signature_label': 'Подпись:',
        'copy_sig': 'Копировать подпись',
        'title_verify': 'Верификация сообщения',
        'address_label': 'Адрес:',
        'message_label': 'Сообщение:',
        'signature_label2': 'Подпись (hex):',
        'verify_btn': 'Проверить',
        'verify_fail': '❌ Неверная подпись',
        'verify_success': '✅ Подпись верна!',
        'fill_all': 'Заполните все поля',
        'invalid_address': 'Неверный адрес (должен быть 40 hex)',
        'enter_amount': 'Сумма должна быть целым числом',
        'positive_amount': 'Сумма должна быть положительной',
        'invalid_privkey': 'Неверный формат приватного ключа',
        'enter_message': 'Введите сообщение',
    },
    'en': {
        'title_create': 'Create Wallet',
        'mnemonic_label': 'Your mnemonic phrase:',
        'save_warning': 'Save these words in a safe place!',
        'confirm_btn': 'I have saved, continue',
        'title_restore': 'Restore Wallet',
        'enter_mnemonic': 'Enter mnemonic phrase (12 words):',
        'restore_btn': 'Restore',
        'error_12_words': 'Mnemonic must contain 12 words',
        'title_send': 'Send WKC',
        'to_label': 'To (address 40 hex):',
        'amount_label': 'Amount (WKC):',
        'send_btn': 'Send',
        'confirm_send': 'Confirmation',
        'send_confirm_text': 'Send {amount} WKC to address {to}?',
        'send_success': 'Transaction sent!',
        'send_fail': 'Failed to send: {error}',
        'title_manage': 'Manage Wallets',
        'wallet_list': 'Wallet List',
        'btn_create': 'Create New',
        'btn_delete': 'Delete',
        'btn_activate': 'Activate',
        'btn_close': 'Close',
        'new_wallet_name': 'Enter wallet name:',
        'delete_confirm': 'Delete wallet "{name}"?',
        'cannot_delete_last': 'Cannot delete the only wallet',
        'wallet_exists': 'Wallet "{name}" already exists',
        'activated': 'Wallet "{name}" activated',
        'title_export': 'Export Private Key',
        'privkey_label': 'Private key for current address:',
        'privkey_warning': 'This key controls funds! Keep it secure.',
        'copy_btn': 'Copy',
        'copied': 'Private key copied',
        'title_import': 'Import Private Key',
        'enter_privkey': 'Enter private key (hex):',
        'import_btn': 'Import',
        'import_success': 'Key imported. Address: {addr}',
        'title_sign': 'Sign Message',
        'msg_label': 'Enter message to sign:',
        'sign_btn': 'Sign',
        'signature_label': 'Signature:',
        'copy_sig': 'Copy Signature',
        'title_verify': 'Verify Message',
        'address_label': 'Address:',
        'message_label': 'Message:',
        'signature_label2': 'Signature (hex):',
        'verify_btn': 'Verify',
        'verify_fail': '❌ Invalid signature',
        'verify_success': '✅ Signature is valid!',
        'fill_all': 'Fill all fields',
        'invalid_address': 'Invalid address (must be 40 hex)',
        'enter_amount': 'Amount must be an integer',
        'positive_amount': 'Amount must be positive',
        'invalid_privkey': 'Invalid private key format',
        'enter_message': 'Enter a message',
    }
}

CURRENT_LANG = 'ru'  # можно переключать

def _(key):
    return LANG.get(CURRENT_LANG, {}).get(key, key)

# ===================== ВСЕ ДИАЛОГИ =====================
class CreateWalletDialog:
    def __init__(self, parent):
        self.result = None
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(_('title_create'))
        self.dialog.geometry("500x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        ttk.Label(self.dialog, text=_('mnemonic_label'), font=("Segoe UI", 12)).pack(pady=10)
        self.mnemonic = generate_mnemonic()
        self.mnemonic_text = tk.Text(self.dialog, height=3, font=("Courier", 10))
        self.mnemonic_text.insert(tk.END, self.mnemonic)
        self.mnemonic_text.config(state=tk.DISABLED)
        self.mnemonic_text.pack(padx=20, pady=10, fill=tk.BOTH)

        ttk.Label(self.dialog, text=_('save_warning'), foreground="red").pack()

        ttk.Button(self.dialog, text=_('confirm_btn'), command=self.confirm).pack(pady=10)

    def confirm(self):
        self.result = self.mnemonic
        self.dialog.destroy()

class RestoreWalletDialog:
    def __init__(self, parent):
        self.result = None
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(_('title_restore'))
        self.dialog.geometry("400x150")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        ttk.Label(self.dialog, text=_('enter_mnemonic')).pack(pady=5)
        self.entry = ttk.Entry(self.dialog, width=50)
        self.entry.pack(pady=5)
        ttk.Button(self.dialog, text=_('restore_btn'), command=self.restore).pack(pady=10)

    def restore(self):
        mnemonic = self.entry.get().strip()
        if len(mnemonic.split()) != 12:
            messagebox.showerror(_('error_12_words'), _('error_12_words'))
            return
        self.result = mnemonic
        self.dialog.destroy()

class SendDialog:
    def __init__(self, parent, wallet, refresh_callback=None):
        self.result = None
        self.wallet = wallet
        self.refresh_callback = refresh_callback
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(_('title_send'))
        self.dialog.geometry("450x250")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        ttk.Label(self.dialog, text=_('to_label')).grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.to_entry = ttk.Entry(self.dialog, width=45)
        self.to_entry.grid(row=0, column=1, padx=10, pady=5)

        ttk.Label(self.dialog, text=_('amount_label')).grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.amount_entry = ttk.Entry(self.dialog, width=20)
        self.amount_entry.grid(row=1, column=1, sticky=tk.W, padx=10, pady=5)

        ttk.Button(self.dialog, text=_('send_btn'), command=self.send).grid(row=2, column=0, columnspan=2, pady=20)

    def send(self):
        to = self.to_entry.get().strip()
        amount_str = self.amount_entry.get().strip()
        if not to or not amount_str:
            messagebox.showwarning(_('fill_all'), _('fill_all'))
            return
        if len(to) != 40:
            messagebox.showwarning(_('invalid_address'), _('invalid_address'))
            return
        try:
            amount = int(amount_str)
        except ValueError:
            messagebox.showwarning(_('enter_amount'), _('enter_amount'))
            return
        if amount <= 0:
            messagebox.showwarning(_('positive_amount'), _('positive_amount'))
            return

        if not messagebox.askyesno(_('confirm_send'), _('send_confirm_text').format(amount=amount, to=to)):
            return

        priv_key = self.wallet.get_private_key_hex()
        result = NodeAPI.send_transaction(self.wallet.get_current_address(), to, amount, priv_key)
        if "error" in result:
            messagebox.showerror(_('send_fail').format(error=result['error']), _('send_fail').format(error=result['error']))
        else:
            messagebox.showinfo(_('send_success'), f"{_('send_success')}\nХэш: {result.get('tx_hash', '')}")
            self.result = True
            self.dialog.destroy()
            # Автообновление баланса и истории
            if self.refresh_callback:
                self.refresh_callback()

class WalletManagerDialog:
    def __init__(self, parent, wallet_manager, refresh_callback=None):
        self.parent = parent
        self.wallet_manager = wallet_manager
        self.refresh_callback = refresh_callback
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(_('title_manage'))
        self.dialog.geometry("500x400")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        ttk.Label(self.dialog, text=_('wallet_list'), font=("Segoe UI", 12)).pack(pady=10)

        self.listbox = tk.Listbox(self.dialog, height=6)
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        self.update_list()

        btn_frame = ttk.Frame(self.dialog)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text=_('btn_create'), command=self.create_wallet).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text=_('btn_delete'), command=self.delete_wallet).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text=_('btn_activate'), command=self.set_active).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text=_('btn_close'), command=self.dialog.destroy).pack(side=tk.LEFT, padx=5)

    def update_list(self):
        self.listbox.delete(0, tk.END)
        for i, name in enumerate(self.wallet_manager.get_wallet_names()):
            marker = "▶ " if i == self.wallet_manager.current_wallet_index else "  "
            self.listbox.insert(tk.END, f"{marker}{name}")

    def create_wallet(self):
        name = simpledialog.askstring(_('new_wallet_name'), _('new_wallet_name'), parent=self.dialog)
        if name:
            mnemonic = generate_mnemonic()
            messagebox.showinfo(_('mnemonic_label'), f"{_('mnemonic_label')}\n\n{mnemonic}\n\n{_('save_warning')}")
            try:
                self.wallet_manager.add_wallet(name, mnemonic)
                self.update_list()
                if self.refresh_callback:
                    self.refresh_callback()
            except ValueError as e:
                messagebox.showerror(_('wallet_exists').format(name=name), str(e))

    def delete_wallet(self):
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showwarning(_('fill_all'), _('fill_all'))
            return
        idx = selection[0]
        name = self.wallet_manager.get_wallet_names()[idx]
        if messagebox.askyesno(_('delete_confirm').format(name=name), _('delete_confirm').format(name=name)):
            try:
                self.wallet_manager.remove_wallet(idx)
                self.update_list()
                if self.refresh_callback:
                    self.refresh_callback()
            except ValueError as e:
                messagebox.showerror(_('cannot_delete_last'), str(e))

    def set_active(self):
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showwarning(_('fill_all'), _('fill_all'))
            return
        idx = selection[0]
        name = self.wallet_manager.get_wallet_names()[idx]
        self.wallet_manager.set_current_index(idx)
        self.update_list()
        messagebox.showinfo(_('activated').format(name=name), _('activated').format(name=name))
        if self.refresh_callback:
            self.refresh_callback()

class ExportPrivateKeyDialog:
    def __init__(self, parent, wallet):
        self.wallet = wallet
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(_('title_export'))
        self.dialog.geometry("500x200")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        ttk.Label(self.dialog, text=_('privkey_label'), font=("Segoe UI", 10)).pack(pady=10)
        priv_key = self.wallet.get_private_key_hex()
        self.key_entry = tk.Text(self.dialog, height=3, font=("Courier", 9))
        self.key_entry.insert(tk.END, priv_key)
        self.key_entry.config(state=tk.DISABLED)
        self.key_entry.pack(fill=tk.X, padx=20, pady=10)

        ttk.Label(self.dialog, text=_('privkey_warning'), foreground="red").pack()
        ttk.Button(self.dialog, text=_('copy_btn'), command=self.copy_key).pack(pady=5)
        ttk.Button(self.dialog, text=_('btn_close'), command=self.dialog.destroy).pack(pady=5)

    def copy_key(self):
        self.dialog.clipboard_clear()
        self.dialog.clipboard_append(self.key_entry.get("1.0", tk.END).strip())
        messagebox.showinfo(_('copied'), _('copied'))

class ImportPrivateKeyDialog:
    def __init__(self, parent, wallet):
        self.wallet = wallet
        self.result = None
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(_('title_import'))
        self.dialog.geometry("500x200")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        ttk.Label(self.dialog, text=_('enter_privkey')).pack(pady=5)
        self.entry = ttk.Entry(self.dialog, width=60)
        self.entry.pack(pady=5)
        ttk.Button(self.dialog, text=_('import_btn'), command=self.import_key).pack(pady=10)

    def import_key(self):
        priv_hex = self.entry.get().strip()
        if not priv_hex:
            messagebox.showwarning(_('fill_all'), _('fill_all'))
            return
        try:
            addr = self.wallet.import_private_key(priv_hex)
            self.result = addr
            messagebox.showinfo(_('import_success').format(addr=addr), _('import_success').format(addr=addr))
            self.dialog.destroy()
        except ValueError as e:
            messagebox.showerror(_('invalid_privkey'), str(e))

class SignMessageDialog:
    def __init__(self, parent, wallet):
        self.wallet = wallet
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(_('title_sign'))
        self.dialog.geometry("600x400")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        ttk.Label(self.dialog, text=_('msg_label')).pack(pady=5)
        self.msg_entry = tk.Text(self.dialog, height=5, font=("Courier", 9))
        self.msg_entry.pack(fill=tk.X, padx=20, pady=5)

        ttk.Button(self.dialog, text=_('sign_btn'), command=self.sign).pack(pady=5)

        self.result_label = ttk.Label(self.dialog, text="", wraplength=500)
        self.result_label.pack(pady=10)

    def sign(self):
        msg = self.msg_entry.get("1.0", tk.END).strip()
        if not msg:
            messagebox.showwarning(_('enter_message'), _('enter_message'))
            return
        sig = self.wallet.sign_message(msg)
        self.result_label.config(text=f"{_('signature_label')}\n{sig}")
        ttk.Button(self.dialog, text=_('copy_sig'), command=lambda: self.copy(sig)).pack(pady=5)

    def copy(self, text):
        self.dialog.clipboard_clear()
        self.dialog.clipboard_append(text)
        messagebox.showinfo(_('copied'), _('copied'))

class VerifyMessageDialog:
    def __init__(self, parent, wallet):
        self.wallet = wallet
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(_('title_verify'))
        self.dialog.geometry("600x400")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        ttk.Label(self.dialog, text=_('address_label')).pack(anchor=tk.W, padx=20)
        self.addr_entry = ttk.Entry(self.dialog, width=50)
        self.addr_entry.pack(fill=tk.X, padx=20, pady=2)

        ttk.Label(self.dialog, text=_('message_label')).pack(anchor=tk.W, padx=20)
        self.msg_entry = tk.Text(self.dialog, height=3, font=("Courier", 9))
        self.msg_entry.pack(fill=tk.X, padx=20, pady=2)

        ttk.Label(self.dialog, text=_('signature_label2')).pack(anchor=tk.W, padx=20)
        self.sig_entry = ttk.Entry(self.dialog, width=70)
        self.sig_entry.pack(fill=tk.X, padx=20, pady=2)

        # Кнопка "Вставить публичный ключ" (для верификации)
        ttk.Button(self.dialog, text="Вставить публичный ключ", command=self.insert_pubkey).pack(pady=2)

        ttk.Button(self.dialog, text=_('verify_btn'), command=self.verify).pack(pady=10)

        self.result_label = ttk.Label(self.dialog, text="", font=("Segoe UI", 10))
        self.result_label.pack(pady=5)

    def insert_pubkey(self):
        """Вставляет публичный ключ текущего адреса в поле адреса (если не заполнено)"""
        if not self.addr_entry.get().strip():
            try:
                pubkey_hex = self.wallet.get_public_key_hex()
                self.addr_entry.insert(0, pubkey_hex)
            except:
                messagebox.showwarning("Ошибка", "Не удалось получить публичный ключ")

    def verify(self):
        addr = self.addr_entry.get().strip()
        msg = self.msg_entry.get("1.0", tk.END).strip()
        sig = self.sig_entry.get().strip()
        if not addr or not msg or not sig:
            messagebox.showwarning(_('fill_all'), _('fill_all'))
            return

        # Проверяем, является ли адрес публичным ключом (66 hex) или 40 hex
        if len(addr) == 40:
            # Это хэш адрес, нужно найти соответствующий публичный ключ
            # Временно: просим пользователя ввести публичный ключ или используем текущий
            if messagebox.askyesno("Публичный ключ", "Вы ввели хэш адрес (40 символов). Для верификации нужен полный публичный ключ. Использовать публичный ключ текущего адреса?"):
                try:
                    pubkey_hex = self.wallet.get_public_key_hex()
                    addr = pubkey_hex
                except:
                    messagebox.showerror("Ошибка", "Не удалось получить публичный ключ")
                    return
            else:
                messagebox.showwarning("Ошибка", "Введите полный публичный ключ (130 hex)")
                return
        elif len(addr) != 130:
            messagebox.showwarning(_('invalid_address'), _('invalid_address') + " (должен быть 40 или 130 hex)")
            return

        try:
            result = verify_signature_with_pubkey(addr, msg, sig)
            if result:
                self.result_label.config(text=_('verify_success'), foreground="green")
            else:
                self.result_label.config(text=_('verify_fail'), foreground="red")
        except Exception as e:
            self.result_label.config(text=f"Ошибка: {e}", foreground="red")