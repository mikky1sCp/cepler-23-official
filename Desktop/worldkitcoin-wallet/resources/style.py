# resources/style.py
import tkinter as tk
from tkinter import ttk

DARK_STYLE = {
    'bg': '#1e1e1e',
    'fg': 'white',
    'selectbg': '#3a3a3a',
    'selectfg': 'white',
    'button_bg': '#f0b90b',
    'button_fg': 'black',
    'entry_bg': '#2d2d2d',
    'entry_fg': 'white',
}

LIGHT_STYLE = {
    'bg': '#f0f0f0',
    'fg': 'black',
    'selectbg': '#c0c0c0',
    'selectfg': 'black',
    'button_bg': '#f0b90b',
    'button_fg': 'black',
    'entry_bg': 'white',
    'entry_fg': 'black',
}

def apply_style(root, theme="dark"):
    style = ttk.Style()
    style.theme_use('clam')
    colors = DARK_STYLE if theme == "dark" else LIGHT_STYLE

    root.configure(bg=colors['bg'])
    style.configure('TButton', padding=6, relief='flat', background=colors['button_bg'], foreground=colors['button_fg'])
    style.map('TButton', background=[('active', '#d49c0a')])
    style.configure('TLabel', background=colors['bg'], foreground=colors['fg'])
    style.configure('TFrame', background=colors['bg'])
    style.configure('TListbox', background=colors['entry_bg'], foreground=colors['entry_fg'])
    style.configure('TEntry', fieldbackground=colors['entry_bg'], foreground=colors['entry_fg'])
    style.configure('TLabelframe', background=colors['bg'], foreground=colors['fg'])
    style.configure('TLabelframe.Label', background=colors['bg'], foreground=colors['fg'])
    # Listbox напрямую
    root.option_add('*Listbox.background', colors['entry_bg'])
    root.option_add('*Listbox.foreground', colors['entry_fg'])
    root.option_add('*Listbox.selectBackground', colors['selectbg'])
    root.option_add('*Listbox.selectForeground', colors['selectfg'])