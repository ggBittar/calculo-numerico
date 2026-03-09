from pathlib import Path
import sys

from PyQt6.QtWidgets import QApplication

from app_window import MainWindow
from ui_controls import apply_dark_theme


def main():
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    window = MainWindow(Path(__file__).resolve().parent)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
