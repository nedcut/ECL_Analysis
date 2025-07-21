import logging
import sys
from PyQt5 import QtWidgets

from src.ui import VideoAnalyzer


def main():
    app = QtWidgets.QApplication(sys.argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    win = VideoAnalyzer()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
