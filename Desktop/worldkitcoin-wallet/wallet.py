# wallet.py
import tkinter as tk
from gui.main_window import MainWindow
import sys
import os

# Добавляем путь к папке, чтобы работали импорты
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()