# gui/widgets.py
import tkinter as tk
from tkinter import ttk
import qrcode
from PIL import Image, ImageTk
import io

class QRCodeDialog:
    def __init__(self, parent, address):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("QR-код адреса")
        self.dialog.geometry("300x350")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Генерируем QR-код
        qr = qrcode.QRCode(box_size=8, border=2)
        qr.add_data(address)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Конвертируем в формат для Tkinter
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        self.tk_img = ImageTk.PhotoImage(Image.open(img_bytes))

        label = ttk.Label(self.dialog, image=self.tk_img)
        label.pack(pady=10)

        ttk.Label(self.dialog, text=address, font=("Courier", 8), wraplength=250).pack(pady=5)
        ttk.Button(self.dialog, text="Закрыть", command=self.dialog.destroy).pack(pady=10)