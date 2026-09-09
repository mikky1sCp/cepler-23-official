# core/theme.py
import json
import os

THEME_FILE = "theme_settings.json"

THEMES = {
    "light": {
        "bg": "#f0f0f0",
        "fg": "#000000",
        "entry_bg": "#ffffff",
        "entry_fg": "#000000",
        "button_bg": "#e0e0e0",
        "button_fg": "#000000",
        "listbox_bg": "#ffffff",
        "listbox_fg": "#000000",
        "status_bg": "#d0d0d0",
        "status_fg": "#000000",
    },
    "dark": {
        "bg": "#1e1e1e",
        "fg": "#ffffff",
        "entry_bg": "#2d2d2d",
        "entry_fg": "#ffffff",
        "button_bg": "#3d3d3d",
        "button_fg": "#ffffff",
        "listbox_bg": "#2d2d2d",
        "listbox_fg": "#ffffff",
        "status_bg": "#2d2d2d",
        "status_fg": "#ffffff",
    }
}

def load_theme():
    if os.path.exists(THEME_FILE):
        with open(THEME_FILE, 'r') as f:
            data = json.load(f)
            return data.get('theme', 'dark')
    return 'dark'

def save_theme(theme_name):
    with open(THEME_FILE, 'w') as f:
        json.dump({'theme': theme_name}, f)

def get_theme_colors(theme_name):
    return THEMES.get(theme_name, THEMES['dark'])