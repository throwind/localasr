from __future__ import annotations

from pathlib import Path
import sys

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QLinearGradient, QPainter, QPen


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: make_dmg_background.py /path/to/install-background.png", file=sys.stderr)
        return 2
    app = QGuiApplication.instance() or QGuiApplication([])
    path = Path(args[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    draw_background(path)
    app.quit()
    print(f"已生成 DMG 背景图：{path}")
    return 0


def draw_background(path: Path) -> None:
    width = 720
    height = 430
    image = QImage(width, height, QImage.Format_ARGB32)
    image.fill(QColor("#f8f8f5"))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, True)

    gradient = QLinearGradient(QPointF(0, 0), QPointF(width, height))
    gradient.setColorAt(0, QColor("#fbfbf8"))
    gradient.setColorAt(0.52, QColor("#f5f5f0"))
    gradient.setColorAt(1, QColor("#eef3ef"))
    painter.fillRect(0, 0, width, height, gradient)

    draw_arrow(painter)

    painter.end()
    image.save(str(path))


def draw_arrow(painter: QPainter) -> None:
    center_y = 190
    arrow_color = QColor("#42463f")
    shadow_pen = QPen(QColor(0, 0, 0, 38), 8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    painter.setPen(shadow_pen)
    painter.drawLine(QPointF(248, center_y + 3), QPointF(316, center_y + 3))
    painter.drawLine(QPointF(316, center_y + 3), QPointF(284, center_y - 29))
    painter.drawLine(QPointF(316, center_y + 3), QPointF(284, center_y + 35))

    arrow_pen = QPen(arrow_color, 7, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    painter.setPen(arrow_pen)
    painter.drawLine(QPointF(248, center_y), QPointF(316, center_y))
    painter.drawLine(QPointF(316, center_y), QPointF(284, center_y - 32))
    painter.drawLine(QPointF(316, center_y), QPointF(284, center_y + 32))


if __name__ == "__main__":
    raise SystemExit(main())
