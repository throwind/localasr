from __future__ import annotations

import gc
import multiprocessing
import os
import platform
from pathlib import Path
import subprocess
import sys
import threading

from .config import AppConfig
from .diagnostics import append_memory_log, memory_log_path, memory_summary
from .pipeline import format_elapsed, output_stem
from . import __version__
from .resources import (
    DEFAULT_MODEL,
    DEFAULT_ONNX_MODEL,
    bundled_onnx_models_dir,
    default_model_cache_dir,
    download_default_models,
    install_bundled_onnx_models,
    list_cached_models,
    model_is_cached,
    resolve_asset,
)
from .service import JobManager, TranscriptionRequest


def main(argv: list[str] | None = None) -> int:
    multiprocessing.freeze_support()
    args = list(sys.argv[1:] if argv is None else argv)
    if "-h" in args or "--help" in args:
        print("usage: localasr-gui [--smoke-test]")
        print("启动 localasr 桌面 GUI。")
        return 0
    if "--smoke-test" in args:
        return smoke_test()

    try:
        from PySide6.QtCore import QPointF, QRectF, QSize, QSettings, QTimer, Qt
        from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QTextCursor
        from PySide6.QtWidgets import (
            QApplication,
            QAbstractSpinBox,
            QCheckBox,
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QDoubleSpinBox,
            QFileDialog,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMenu,
            QMessageBox,
            QPlainTextEdit,
            QPushButton,
            QSpinBox,
            QStyle,
            QTabWidget,
            QToolButton,
            QVBoxLayout,
            QWidget,
        )
    except ImportError:
        print("错误：缺少 GUI 依赖，请安装：pip install 'localasr[gui]'", file=sys.stderr)
        return 1

    class ProgressButton(QPushButton):
        """将任务进度直接绘制在主操作按钮内，避免额外占据一行。"""

        def __init__(self, text: str, parent: QWidget | None = None) -> None:
            super().__init__(text, parent)
            self._progress: float | None = None
            self._normal_text = text
            self.setMinimumHeight(40)
            self.setCursor(Qt.PointingHandCursor)

        def set_progress(self, progress: float | None) -> None:
            self._progress = None if progress is None else max(0.0, min(1.0, progress))
            self.update()

        def set_action_text(self, text: str) -> None:
            self._normal_text = text
            self.setText(text)

        def paintEvent(self, event) -> None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
            radius = 9.0
            if self._progress is None:
                if not self.isEnabled():
                    background = QColor("#e7ece9")
                    text_color = QColor("#8b9894")
                elif self.isDown():
                    background = QColor("#cddfda")
                    text_color = QColor("#124f54")
                elif self.underMouse():
                    background = QColor("#d9ebe5")
                    text_color = QColor("#105f68")
                else:
                    background = QColor("#e5efeb")
                    text_color = QColor("#1d5f58")
            else:
                background = QColor("#dcebe7")
                text_color = QColor("#164d50")
            painter.setPen(Qt.NoPen)
            painter.setBrush(background)
            painter.drawRoundedRect(rect, radius, radius)
            if self._progress is not None:
                path = QPainterPath()
                path.addRoundedRect(rect, radius, radius)
                painter.save()
                painter.setClipPath(path)
                painter.setBrush(QColor("#105f68"))
                painter.drawRect(QRectF(rect.left(), rect.top(), rect.width() * self._progress, rect.height()))
                painter.restore()
            painter.setPen(QPen(text_color))
            painter.setFont(self.font())
            painter.drawText(rect, Qt.AlignCenter, self.text())
            if self._progress is not None:
                painter.save()
                painter.setClipRect(QRectF(rect.left(), rect.top(), rect.width() * self._progress, rect.height()))
                painter.setPen(QPen(QColor("#ffffff")))
                painter.drawText(rect, Qt.AlignCenter, self.text())
                painter.restore()

    class MainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.settings = QSettings("LocalASR", "LocalASR")
            self.manager = JobManager()
            self.current_job_id: str | None = None
            self.rendered_log_count = 0
            self.progress_logs: list[str] = []
            self.result_status_text: str | None = None
            self.model_prompt_shown = False
            self.selected_audio_path: Path | None = None
            self.result_txt_path: Path | None = None
            self.result_folder_path: Path | None = None
            self.apply_default_migration()
            self.engine_value = self.setting_text("options/engine", os.environ.get("LOCALASR_ENGINE", "funasr"))
            self.model_value = self.setting_text("options/model", self.default_model_for_engine(self.engine_value))
            if self.engine_value == "sherpa-onnx" and self.model_value == DEFAULT_MODEL:
                self.model_value = DEFAULT_ONNX_MODEL
            self.model_dir_value = self.setting_text("options/model_dir", str(default_model_cache_dir()))
            self.input_dir_value = self.setting_text("paths/input_dir", str(Path.home()))
            self.output_dir_value = self.input_dir_value
            self.recursive_value = False
            self.overwrite_value = self.setting_bool("options/overwrite", False)
            self.language_value = self.setting_text("options/language", "auto")
            self.device_value = self.setting_text("options/device", "cpu")
            self.speaker_diarization_value = self.setting_bool("options/speaker_diarization", True)
            self.chunk_seconds_value = self.setting_int("options/chunk_seconds", 600)
            self.boundary_search_seconds_value = self.setting_int("options/boundary_search_seconds", 30)
            self.overlap_seconds_value = self.setting_int("options/overlap_seconds", 5)
            self.silence_threshold_db_value = self.setting_int("options/silence_threshold_db", -35)
            self.silence_min_duration_value = self.setting_float("options/silence_min_duration", 0.5)

            self.setWindowTitle("音频转文本")
            icon_path = resolve_asset("localasr-icon.png")
            if icon_path is not None:
                self.setWindowIcon(QIcon(str(icon_path)))
            self.resize(570, 250)
            self.setMinimumSize(520, 235)

            root = QWidget()
            self.setCentralWidget(root)
            layout = QVBoxLayout(root)
            layout.setContentsMargins(18, 13, 18, 13)
            layout.setSpacing(8)

            header_layout = QHBoxLayout()
            title = QLabel("音频转文本")
            title.setObjectName("Title")
            subtitle = QLabel("本地离线转写")
            subtitle.setObjectName("Subtitle")
            self.settings_button = QPushButton("设置")
            self.settings_button.setObjectName("SettingsButton")
            self.settings_button.setIcon(self.make_sliders_icon("#1f5f5b"))
            self.settings_button.setIconSize(QSize(18, 18))
            self.settings_button.clicked.connect(self.open_settings)
            header_layout.addWidget(title)
            header_layout.addWidget(subtitle, 1)
            header_layout.addWidget(self.settings_button)
            layout.addLayout(header_layout)

            file_panel = QWidget()
            file_panel.setObjectName("FilePanel")
            file_layout = QVBoxLayout(file_panel)
            file_layout.setContentsMargins(12, 8, 12, 8)
            file_row = QHBoxLayout()
            self.choose_input_button = QPushButton("打开音频")
            self.choose_input_button.setObjectName("ChooseButton")
            self.choose_input_button.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
            self.choose_input_button.clicked.connect(self.choose_audio_file)
            self.file_name_label = QLabel("还没有选择文件")
            self.file_name_label.setObjectName("FileName")
            self.file_name_label.setWordWrap(True)
            self.file_meta_label = QLabel("选择一个音频文件开始转写")
            self.file_meta_label.setObjectName("FileMeta")
            self.file_meta_label.setWordWrap(True)
            file_info = QVBoxLayout()
            file_info.setSpacing(3)
            file_info.addWidget(self.file_name_label)
            file_info.addWidget(self.file_meta_label)
            file_row.addWidget(self.choose_input_button)
            file_row.addLayout(file_info, 1)
            file_layout.addLayout(file_row)
            layout.addWidget(file_panel)

            self.start_button = ProgressButton("开始转写")
            self.start_button.setObjectName("PrimaryAction")
            self.start_button.clicked.connect(self.start_job)
            layout.addWidget(self.start_button)
            self.status_label = QLabel()

            result_panel = QWidget()
            result_panel.setObjectName("ResultPanel")
            result_layout = QHBoxLayout(result_panel)
            result_layout.setContentsMargins(12, 4, 12, 4)
            result_layout.setSpacing(7)
            self.result_status_label = QLabel("生成文本")
            self.result_status_label.setObjectName("ResultStatus")
            self.result_file_button = QPushButton("文本文件将在转写完成后显示")
            self.result_file_button.setObjectName("ResultFileButton")
            self.result_file_button.setToolTip("转写完成后，点击文件名打开 TXT")
            self.result_file_button.clicked.connect(self.open_text_result)
            self.result_menu_button = QToolButton()
            self.result_menu_button.setText("…")
            self.result_menu_button.setObjectName("ResultMenuButton")
            self.result_menu_button.setToolTip("文本文件操作")
            self.result_menu = QMenu(self.result_menu_button)
            self.result_menu.addAction("在 Finder 中显示", self.open_result_folder)
            self.result_menu.addAction("删除", self.delete_text_result)
            self.result_menu_button.setMenu(self.result_menu)
            self.result_menu_button.setPopupMode(QToolButton.InstantPopup)
            result_layout.addWidget(self.result_status_label)
            result_layout.addWidget(self.result_file_button, 1)
            result_layout.addWidget(self.result_menu_button)
            layout.addWidget(result_panel)

            self.timer = QTimer(self)
            self.timer.setInterval(800)
            self.timer.timeout.connect(self.refresh_job)
            self.memory_timer = QTimer(self)
            self.memory_timer.setInterval(60_000)
            self.memory_timer.timeout.connect(self.record_idle_memory)
            self.memory_timer.start()

            chevron_path = resolve_asset("chevron-down.png")
            checkmark_path = resolve_asset("checkmark.svg")
            chevron_rule = (
                f"""
                QComboBox::down-arrow {{
                    image: url("{chevron_path.as_posix()}");
                    width: 10px;
                    height: 10px;
                }}
                """
                if chevron_path is not None
                else ""
            )
            checkmark_rule = (
                f'''
                QCheckBox::indicator:checked {{
                    image: url("{checkmark_path.as_posix()}");
                }}
                '''
                if checkmark_path is not None
                else ""
            )
            self.setStyleSheet(
                """
                QWidget { font-size: 13px; color: #23302e; }
                QMainWindow { background: #f4f6f3; }
                QDialog, QMessageBox { background: #f4f6f3; color: #23302e; }
                QDialog QLabel, QMessageBox QLabel, QTabWidget QLabel { color: #23302e; }
                QTabWidget::pane { border: 1px solid #d7e1dc; background: #fcfdfb; }
                QTabBar::tab {
                    color: #52635e;
                    background: #e9eeeb;
                    border: 1px solid #d7e1dc;
                    border-bottom: none;
                    padding: 7px 13px;
                    margin-right: 3px;
                }
                QTabBar::tab:selected { color: #124f54; background: #fcfdfb; font-weight: 700; }
                #Title { font-size: 23px; font-weight: 700; color: #152b28; }
                #Subtitle { color: #71807c; padding-left: 8px; }
                #FilePanel, #ResultPanel {
                    border: 1px solid #d7e1dc;
                    border-radius: 10px;
                    background: #fcfdfb;
                }
                QGroupBox {
                    border: 1px solid #dedbd2;
                    border-radius: 8px;
                    margin-top: 10px;
                    padding: 9px;
                    background: #fffefa;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 4px;
                    color: #344054;
                    font-weight: bold;
                }
                QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
                    border: 1px solid #d0d5dd;
                    border-radius: 7px;
                    padding: 6px;
                    background: white;
                    color: #23302e;
                }
                QComboBox {
                    padding-right: 34px;
                }
                QComboBox::drop-down {
                    subcontrol-origin: padding;
                    subcontrol-position: top right;
                    width: 30px;
                    border: none;
                    border-left: 1px solid #eef0f2;
                    border-top-right-radius: 7px;
                    border-bottom-right-radius: 7px;
                    background: #f2f4f7;
                }
                QComboBox::drop-down:hover {
                    background: #e6eaee;
                }
                QComboBox QAbstractItemView {
                    border: 1px solid #d0d5dd;
                    border-radius: 7px;
                    padding: 4px;
                    background: white;
                    selection-background-color: #1f5f5b;
                    selection-color: white;
                }
                QPushButton {
                    border: 1px solid #19665f;
                    border-radius: 8px;
                    padding: 7px 12px;
                    color: white;
                    background: #19665f;
                    font-weight: bold;
                }
                QPushButton:hover { background: #24766e; border-color: #24766e; }
                QPushButton:pressed { background: #124a45; border-color: #124a45; }
                QPushButton:disabled { color: #85918e; background: #e6ece9; border-color: #dce4e0; }
                #SettingsButton {
                    color: #1d5f58;
                    background: #f8fbf9;
                    border: 1px solid #bdd4cc;
                    border-radius: 7px;
                    padding: 5px 7px;
                }
                #SettingsButton:hover { background: #e4eeea; border-color: #c9dad4; }
                #SettingsButton:pressed { background: #d2e2dc; }
                #ChooseButton { padding: 7px 11px; min-height: 22px; }
                #PrimaryAction { min-height: 30px; font-size: 14px; border: none; background: transparent; }
                #SecondaryButton {
                    color: #1f5f5b;
                    background: #fffefa;
                    border: 1px solid #bad1cc;
                }
                #SecondaryButton:hover { background: #eef4f2; }
                #SecondaryButton:pressed {
                    color: #174b47;
                    background: #dcebe7;
                    border-color: #8fb8b0;
                }
                #SmallButton {
                    padding: 5px 8px;
                    min-height: 24px;
                }
                #SmallSecondaryButton {
                    color: #1f5f5b;
                    background: #fffefa;
                    border: 1px solid #bad1cc;
                    padding: 5px 8px;
                    min-height: 24px;
                }
                #SmallSecondaryButton:hover { background: #eef4f2; }
                #SmallSecondaryButton:pressed {
                    color: #174b47;
                    background: #dcebe7;
                    border-color: #8fb8b0;
                }
                #TinySecondaryButton {
                    color: #1f5f5b;
                    background: #fffefa;
                    border: 1px solid #bad1cc;
                    padding: 3px 6px;
                    min-height: 20px;
                    font-size: 12px;
                }
                #TinySecondaryButton:hover { background: #eef4f2; }
                #TinySecondaryButton:pressed {
                    color: #174b47;
                    background: #dcebe7;
                    border-color: #8fb8b0;
                }
                #ResultButton, #SmallSecondaryButton, #TinySecondaryButton {
                    color: #1d5f58;
                    background: #f8fbf9;
                    border: 1px solid #bdd4cc;
                }
                #ResultFileButton {
                    color: #92a09c;
                    background: transparent;
                    border: none;
                    padding: 4px 0;
                    text-align: left;
                    font-weight: normal;
                }
                #ResultFileButton:enabled { color: #1b64a0; font-weight: 700; }
                #ResultFileButton:enabled:hover { color: #124a7f; text-decoration: underline; }
                #ResultFileButton:disabled { color: #92a09c; background: transparent; border: none; }
                #ResultStatus { color: #1d4e49; font-weight: 700; }
                #ResultMenuButton {
                    color: #1d5f58;
                    background: transparent;
                    border: 1px solid #bdd4cc;
                    border-radius: 7px;
                    min-width: 25px;
                    min-height: 25px;
                    font-size: 17px;
                    padding: 0;
                }
                #ResultMenuButton:hover { background: #e8f2ee; }
                #ResultMenuButton::menu-indicator { image: none; width: 0px; }
                #FileName { color: #213a36; font-weight: 700; }
                #FileMeta { color: #71807c; }
                #SettingsLogView { color: #52635e; background: #f1f4f2; border: 1px solid #e0e7e3; border-radius: 7px; padding: 5px; }
                #SettingsLogHint { color: #71807c; font-size: 11px; }
                QMenu { background: #ffffff; border: 1px solid #cfdad5; padding: 4px; }
                QMenu::item { padding: 6px 20px 6px 10px; border-radius: 4px; }
                QMenu::item:selected { background: #e8f2ee; color: #174b47; }
                QCheckBox { color: #23302e; spacing: 7px; }
                QCheckBox::indicator {
                    width: 16px;
                    height: 16px;
                    border: 1px solid #65847c;
                    border-radius: 4px;
                    background: #ffffff;
                }
                QCheckBox::indicator:hover { border-color: #105f68; background: #edf6f3; }
                QCheckBox::indicator:checked { border-color: #105f68; background: #105f68; }
                QCheckBox::indicator:checked:hover { border-color: #073f47; background: #073f47; }
                QCheckBox::indicator:disabled { border-color: #c8d3ce; background: #edf1ef; }
                QCheckBox::indicator:checked:disabled { background: #8aa9a2; border-color: #8aa9a2; }
                """
                + chevron_rule
                + checkmark_rule
            )
            self.update_selected_file_view()
            QTimer.singleShot(600, self.prompt_default_model_download_if_needed)

        def setting_text(self, key: str, default: str) -> str:
            return str(self.settings.value(key, default))

        def setting_bool(self, key: str, default: bool) -> bool:
            value = self.settings.value(key, default)
            if isinstance(value, bool):
                return value
            return str(value).lower() in {"1", "true", "yes", "on"}

        def setting_int(self, key: str, default: int) -> int:
            try:
                return int(self.settings.value(key, default))
            except (TypeError, ValueError):
                return default

        def setting_float(self, key: str, default: float) -> float:
            try:
                return float(self.settings.value(key, default))
            except (TypeError, ValueError):
                return default

        def make_sliders_icon(self, color: str) -> QIcon:
            pixmap = QPixmap(24, 24)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            pen = QPen(QColor(color), 2.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(QColor(color))
            rows = ((6.0, 15.0), (12.0, 9.0), (18.0, 17.0))
            for y, knob_x in rows:
                painter.drawLine(QPointF(4.5, y), QPointF(19.5, y))
                painter.drawEllipse(QRectF(knob_x - 2.6, y - 2.6, 5.2, 5.2))
            painter.end()
            return QIcon(pixmap)

        def apply_default_migration(self) -> None:
            marker = "ui/defaults_20260620"
            if self.setting_bool(marker, False):
                return
            if not self.settings.value("paths/input_dir", ""):
                self.settings.setValue("paths/input_dir", str(Path.home()))
            self.settings.setValue("options/recursive", False)
            self.settings.setValue("options/speaker_diarization", True)
            self.settings.setValue("formats/txt", True)
            self.settings.setValue("formats/json", False)
            self.settings.setValue("formats/srt", False)
            self.settings.setValue(marker, True)
            self.settings.sync()

        def set_combo_value(self, combo: QComboBox, value: str) -> None:
            index = combo.findText(value)
            if index >= 0:
                combo.setCurrentIndex(index)
            elif combo.isEditable():
                combo.setCurrentText(value)

        def save_settings(self) -> None:
            self.settings.setValue("paths/input_dir", self.input_dir_value)
            self.settings.setValue("paths/output_dir", self.input_dir_value)
            self.settings.setValue("options/engine", self.engine_value)
            self.settings.setValue("options/model", self.model_value)
            self.settings.setValue("options/model_dir", self.model_dir_value)
            self.settings.setValue("options/language", self.language_value)
            self.settings.setValue("options/device", self.device_value)
            self.settings.setValue("options/chunk_seconds", self.chunk_seconds_value)
            self.settings.setValue("options/boundary_search_seconds", self.boundary_search_seconds_value)
            self.settings.setValue("options/overlap_seconds", self.overlap_seconds_value)
            self.settings.setValue("options/silence_threshold_db", self.silence_threshold_db_value)
            self.settings.setValue("options/silence_min_duration", self.silence_min_duration_value)
            self.settings.setValue("options/recursive", self.recursive_value)
            self.settings.setValue("options/overwrite", self.overwrite_value)
            self.settings.setValue("options/speaker_diarization", self.speaker_diarization_value)
            self.settings.setValue("formats/txt", True)
            self.settings.setValue("formats/json", False)
            self.settings.setValue("formats/srt", False)
            self.settings.sync()

        def closeEvent(self, event) -> None:
            self.save_settings()
            super().closeEvent(event)

        def choose_audio_file(self) -> None:
            start = self.input_dir_value.strip() or str(Path.home())
            file_path, _filter = QFileDialog.getOpenFileName(
                self,
                "选择音频文件",
                start,
                self.audio_file_dialog_filter(),
            )
            if not file_path:
                return
            audio_path = Path(file_path).expanduser()
            if not audio_path.is_file() or not self.is_supported_audio_file(audio_path):
                QMessageBox.warning(self, "不是音频文件", f"请选择支持的音频文件：{audio_path.name}")
                return
            self.selected_audio_path = audio_path
            self.input_dir_value = str(audio_path.parent)
            self.output_dir_value = str(audio_path.parent)
            self.result_txt_path = self.planned_txt_path(audio_path)
            self.result_folder_path = audio_path.parent
            self.result_status_text = None
            self.start_button.set_progress(None)
            self.start_button.set_action_text("开始转写")
            self.status_label.setText("已选择音频")
            self.progress_logs = [f"已选择音频：{audio_path.name}"]
            self.save_settings()
            self.update_selected_file_view()

        def audio_file_dialog_filter(self) -> str:
            patterns = " ".join(f"*{extension}" for extension in AppConfig().audio_extensions)
            return f"音频文件 ({patterns});;所有文件 (*)"

        def is_supported_audio_file(self, path: Path) -> bool:
            return path.suffix.lower() in {extension.lower() for extension in AppConfig().audio_extensions}

        def planned_txt_path(self, audio_path: Path) -> Path:
            return output_stem(audio_path, audio_path.parent, audio_path.parent).with_suffix(".txt")

        def update_selected_file_view(self) -> None:
            audio_path = self.selected_audio_path
            has_file = audio_path is not None and audio_path.exists()
            if has_file and audio_path is not None:
                self.file_name_label.setText(audio_path.name)
                self.file_name_label.setToolTip(str(audio_path))
                self.file_meta_label.setText(f"{self.format_file_size(audio_path)}  ·  {audio_path.parent}")
                self.result_txt_path = self.planned_txt_path(audio_path)
                self.result_folder_path = audio_path.parent
            else:
                self.file_name_label.setText("还没有选择文件")
                self.file_name_label.setToolTip("")
                self.file_meta_label.setText("选择一个音频文件开始转写")
                self.result_txt_path = None
                self.result_folder_path = None
            self.start_button.setEnabled(bool(has_file) and self.current_job_id is None)
            txt_ready = self.result_txt_path is not None and self.result_txt_path.exists()
            self.result_file_button.setEnabled(txt_ready)
            self.result_menu_button.setEnabled(txt_ready)
            if txt_ready and self.result_txt_path is not None:
                self.result_status_label.setText(self.result_status_text or "生成文本")
                self.result_file_button.setText(self.result_txt_path.name)
                self.result_file_button.setToolTip(f"打开：{self.result_txt_path}")
            elif not has_file:
                self.result_status_label.setText("生成文本")
                self.result_file_button.setText("文本文件将在转写完成后显示")
                self.result_file_button.setToolTip("转写完成后，点击文件名打开 TXT")
            else:
                self.result_status_label.setText("生成文本")
                self.result_file_button.setText("文本文件将在转写完成后显示")
                self.result_file_button.setToolTip("转写完成后，点击文件名打开 TXT")

        def progress_button_text(self, stage: str, progress_percent: int) -> str:
            if progress_percent >= 100:
                return "正在完成转写 · 100%"
            if stage in {"queued", "starting", "scan"}:
                label = "正在扫描音频"
            elif stage == "audio_tools":
                label = "正在检查音频"
            elif stage == "load_model":
                label = "正在加载模型"
            elif stage in {"file_start", "split"}:
                label = "正在处理音频"
            elif stage in {"write", "file_done", "memory", "done"}:
                label = "正在保存文本"
            else:
                label = "正在识别语音"
            return f"{label} · {progress_percent}%"

        def default_model_for_engine(self, engine: str) -> str:
            return DEFAULT_ONNX_MODEL if engine == "sherpa-onnx" else DEFAULT_MODEL

        def prompt_default_model_download_if_needed(self) -> None:
            if self.model_prompt_shown:
                return
            self.model_prompt_shown = True
            model_dir = Path(self.model_dir_value).expanduser() if self.model_dir_value.strip() else default_model_cache_dir()
            if model_is_cached(self.default_model_for_engine(self.engine_value), model_dir):
                return
            message = QMessageBox(self)
            message.setIcon(QMessageBox.Information)
            message.setWindowTitle("需要下载模型")
            message.setText("还没有找到默认语音模型。")
            message.setInformativeText("首次使用前请先下载默认模型，下载完成后就可以直接开始转写。")
            open_button = message.addButton("打开模型设置", QMessageBox.AcceptRole)
            message.addButton("稍后", QMessageBox.RejectRole)
            message.exec()
            if message.clickedButton() == open_button:
                self.open_settings()

        def open_settings(self) -> None:
            dialog = QDialog(self)
            dialog.setWindowTitle("设置")
            dialog.setMinimumWidth(640)
            dialog_layout = QVBoxLayout(dialog)
            tabs = QTabWidget()
            dialog_layout.addWidget(tabs)

            model_tab = QWidget()
            model_form = QFormLayout(model_tab)
            model_dir_edit = QLineEdit(self.model_dir_value)
            engine_combo = QComboBox()
            engine_combo.addItem("ONNX 离线", "sherpa-onnx")
            engine_combo.addItem("FunASR / PyTorch", "funasr")
            engine_index = engine_combo.findData(self.engine_value)
            engine_combo.setCurrentIndex(engine_index if engine_index >= 0 else 0)
            model_combo = QComboBox()
            model_combo.setEditable(True)
            model_status = QLabel()
            download_button = QPushButton("下载默认模型")
            model_dir_button = QPushButton("选择")
            refresh_button = QPushButton("刷新")
            model_dir_row = QHBoxLayout()
            model_dir_row.addWidget(model_dir_edit, 1)
            model_dir_row.addWidget(model_dir_button)
            model_row = QHBoxLayout()
            model_row.addWidget(model_combo, 1)
            model_row.addWidget(refresh_button)
            download_row = QHBoxLayout()
            download_row.addWidget(download_button)
            download_row.addWidget(model_status, 1)

            def pick_model_dir() -> None:
                directory = QFileDialog.getExistingDirectory(dialog, "选择模型目录", model_dir_edit.text())
                if directory:
                    model_dir_edit.setText(directory)
                    refresh_models()

            model_dir_button.clicked.connect(pick_model_dir)

            def selected_model_dir() -> Path:
                text = model_dir_edit.text().strip()
                return Path(text) if text else default_model_cache_dir()

            def selected_engine() -> str:
                value = engine_combo.currentData()
                return str(value or "funasr")

            def default_selected_model() -> str:
                return self.default_model_for_engine(selected_engine())

            def refresh_models(selected: str | None = None) -> None:
                current = selected or model_combo.currentText().strip() or self.model_value or default_selected_model()
                if selected_engine() == "sherpa-onnx" and current == DEFAULT_MODEL:
                    current = DEFAULT_ONNX_MODEL
                model_combo.clear()
                if selected_engine() == "sherpa-onnx":
                    model_combo.addItem(DEFAULT_ONNX_MODEL)
                else:
                    model_combo.addItems(list_cached_models(selected_model_dir()))
                index = model_combo.findText(current)
                if index >= 0:
                    model_combo.setCurrentIndex(index)
                else:
                    model_combo.setCurrentText(current)
                refresh_model_status()

            def refresh_model_status() -> None:
                model = model_combo.currentText().strip() or default_selected_model()
                cached = model_is_cached(model, selected_model_dir())
                if selected_engine() == "sherpa-onnx":
                    if cached:
                        model_status.setText("已可用")
                    elif bundled_onnx_models_dir() is not None:
                        model_status.setText("已内置，首次使用会安装")
                    else:
                        model_status.setText("未找到内置模型")
                else:
                    model_status.setText("已下载" if cached else "未下载")

            download_state = {"running": False, "done": False, "error": None, "message": ""}

            def start_download() -> None:
                if download_state["running"]:
                    return
                target_dir = selected_model_dir()
                download_state.update({"running": True, "done": False, "error": None, "message": "准备下载"})
                download_button.setEnabled(False)
                model_dir_button.setEnabled(False)
                engine_combo.setEnabled(False)

                def worker() -> None:
                    try:
                        if selected_engine() == "sherpa-onnx":
                            download_state.update({"message": "正在安装内置 ONNX 模型"})
                            install_bundled_onnx_models(target_dir)
                        else:
                            download_default_models(
                                target_dir,
                                progress_callback=lambda model_id, index, total: download_state.update(
                                    {"message": f"下载中 {index}/{total}: {model_id}"}
                                ),
                            )
                    except Exception as exc:
                        download_state.update({"error": str(exc)})
                    finally:
                        download_state.update({"running": False, "done": True})

                threading.Thread(target=worker, name="localasr-model-download", daemon=True).start()

            def poll_download() -> None:
                if download_state["message"]:
                    model_status.setText(str(download_state["message"]))
                if download_state["done"]:
                    download_timer.stop()
                    download_button.setEnabled(True)
                    model_dir_button.setEnabled(True)
                    engine_combo.setEnabled(True)
                    if download_state["error"]:
                        model_status.setText(f"下载失败：{download_state['error']}")
                    else:
                        refresh_models(default_selected_model())
                        model_status.setText("默认模型已下载")
                    download_state["done"] = False

            download_timer = QTimer(dialog)
            download_timer.setInterval(500)
            download_timer.timeout.connect(poll_download)
            download_button.clicked.connect(lambda: (start_download(), download_timer.start()))
            refresh_button.clicked.connect(lambda: refresh_models())
            engine_combo.currentIndexChanged.connect(lambda _index: refresh_models(default_selected_model()))
            model_combo.currentTextChanged.connect(lambda _text: refresh_model_status())
            refresh_models(self.model_value)
            model_form.addRow("默认模型", download_row)
            model_form.addRow("转写引擎", engine_combo)
            model_form.addRow("模型目录", model_dir_row)
            model_form.addRow("当前模型", model_row)
            tabs.addTab(model_tab, "模型设置")

            transcribe_tab = QWidget()
            transcribe_layout = QVBoxLayout(transcribe_tab)
            transcribe_layout.setSpacing(10)
            language_combo = QComboBox()
            language_combo.addItems(["auto", "zh", "en", "yue", "ja", "ko", "nospeech"])
            self.set_combo_value(language_combo, self.language_value)
            device_combo = QComboBox()
            device_combo.setEditable(True)
            device_combo.addItems(["cpu", "cuda:0"])
            self.set_combo_value(device_combo, self.device_value)
            speaker_check = QCheckBox("启用说话人识别")
            speaker_check.setChecked(self.speaker_diarization_value)
            speaker_check.setToolTip("输出中按句标注说话人编号，适合访谈、会议、播客音频。")
            overwrite_check = QCheckBox("覆盖已有结果")
            overwrite_check.setChecked(self.overwrite_value)
            chunk_spin = QSpinBox()
            chunk_spin.setRange(60, 7200)
            chunk_spin.setSingleStep(60)
            chunk_spin.setValue(self.chunk_seconds_value)
            chunk_spin.setSuffix(" 秒")
            chunk_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
            boundary_spin = QSpinBox()
            boundary_spin.setRange(0, 300)
            boundary_spin.setSingleStep(5)
            boundary_spin.setValue(self.boundary_search_seconds_value)
            boundary_spin.setSuffix(" 秒")
            boundary_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
            overlap_spin = QSpinBox()
            overlap_spin.setRange(0, 60)
            overlap_spin.setSingleStep(1)
            overlap_spin.setValue(self.overlap_seconds_value)
            overlap_spin.setSuffix(" 秒")
            overlap_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
            threshold_spin = QSpinBox()
            threshold_spin.setRange(-80, -10)
            threshold_spin.setSingleStep(1)
            threshold_spin.setValue(self.silence_threshold_db_value)
            threshold_spin.setSuffix(" dB")
            threshold_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
            silence_duration_spin = QDoubleSpinBox()
            silence_duration_spin.setRange(0.1, 5.0)
            silence_duration_spin.setSingleStep(0.1)
            silence_duration_spin.setDecimals(1)
            silence_duration_spin.setValue(self.silence_min_duration_value)
            silence_duration_spin.setSuffix(" 秒")
            silence_duration_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)

            basic_group = QGroupBox("基础")
            basic_grid = QGridLayout(basic_group)
            basic_grid.setHorizontalSpacing(12)
            basic_grid.setVerticalSpacing(8)
            basic_grid.addWidget(QLabel("语言"), 0, 0)
            basic_grid.addWidget(language_combo, 0, 1)
            basic_grid.addWidget(QLabel("设备"), 0, 2)
            basic_grid.addWidget(device_combo, 0, 3)
            basic_grid.addWidget(QLabel("说话人"), 1, 0)
            basic_grid.addWidget(speaker_check, 1, 1)
            basic_grid.addWidget(QLabel("已有结果"), 1, 2)
            basic_grid.addWidget(overwrite_check, 1, 3)
            basic_grid.setColumnStretch(1, 1)
            basic_grid.setColumnStretch(3, 1)

            slice_group = QGroupBox("切片与静音")
            slice_grid = QGridLayout(slice_group)
            slice_grid.setHorizontalSpacing(12)
            slice_grid.setVerticalSpacing(8)
            slice_grid.addWidget(QLabel("目标片段"), 0, 0)
            slice_grid.addWidget(chunk_spin, 0, 1)
            slice_grid.addWidget(QLabel("边界搜索"), 0, 2)
            slice_grid.addWidget(boundary_spin, 0, 3)
            slice_grid.addWidget(QLabel("重叠保护"), 1, 0)
            slice_grid.addWidget(overlap_spin, 1, 1)
            slice_grid.addWidget(QLabel("静音阈值"), 1, 2)
            slice_grid.addWidget(threshold_spin, 1, 3)
            slice_grid.addWidget(QLabel("静音时长"), 2, 0)
            slice_grid.addWidget(silence_duration_spin, 2, 1)
            slice_grid.setColumnStretch(1, 1)
            slice_grid.setColumnStretch(3, 1)

            transcribe_layout.addWidget(basic_group)
            transcribe_layout.addWidget(slice_group)
            transcribe_layout.addStretch(1)
            tabs.addTab(transcribe_tab, "转译设置")

            about_tab = QWidget()
            about_form = QFormLayout(about_tab)
            about_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            supported_files = QLabel("m4a、mp3、wav、flac、aac、ogg、opus、wma，以及 mp4、mov、mkv、webm 等常见音视频文件。")
            supported_files.setWordWrap(True)
            about_form.addRow("支持文件", supported_files)
            about_form.addRow("输出格式", QLabel("纯文本 TXT，保存在源音频同一目录"))
            about_form.addRow("版本", QLabel(__version__))
            about_form.addRow("开发者", QLabel("pax"))
            tabs.addTab(about_tab, "关于")

            log_tab = QWidget()
            log_layout = QVBoxLayout(log_tab)
            log_layout.setContentsMargins(12, 12, 12, 12)
            log_view = QPlainTextEdit()
            log_view.setReadOnly(True)
            log_view.setObjectName("SettingsLogView")
            log_view.setPlaceholderText("尚无任务日志。")
            log_layout.addWidget(log_view)
            log_hint = QLabel(f"详细内存诊断日志：{memory_log_path()}")
            log_hint.setObjectName("SettingsLogHint")
            log_hint.setWordWrap(True)
            log_layout.addWidget(log_hint)
            tabs.addTab(log_tab, "进度日志")

            log_state = {"count": -1}

            def refresh_settings_log() -> None:
                if log_state["count"] == len(self.progress_logs):
                    return
                log_view.setPlainText("\n".join(self.progress_logs))
                log_view.moveCursor(QTextCursor.MoveOperation.End)
                log_state["count"] = len(self.progress_logs)

            log_timer = QTimer(dialog)
            log_timer.setInterval(500)
            log_timer.timeout.connect(refresh_settings_log)
            refresh_settings_log()
            log_timer.start()

            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            ok_button = buttons.button(QDialogButtonBox.Ok)
            cancel_button = buttons.button(QDialogButtonBox.Cancel)
            if ok_button is not None:
                ok_button.setText("确认")
            if cancel_button is not None:
                cancel_button.setText("取消")

            def accept_settings() -> None:
                if overlap_spin.value() * 2 >= chunk_spin.value():
                    QMessageBox.warning(dialog, "设置无效", "重叠保护必须小于目标片段时长的一半。")
                    return
                dialog.accept()

            buttons.accepted.connect(accept_settings)
            buttons.rejected.connect(dialog.reject)
            dialog_layout.addWidget(buttons)

            if dialog.exec() == QDialog.Accepted:
                self.overwrite_value = overwrite_check.isChecked()
                self.engine_value = selected_engine()
                self.model_value = model_combo.currentText().strip() or self.default_model_for_engine(self.engine_value)
                self.model_dir_value = model_dir_edit.text().strip() or str(default_model_cache_dir())
                self.language_value = language_combo.currentText()
                self.device_value = device_combo.currentText()
                self.speaker_diarization_value = speaker_check.isChecked()
                self.chunk_seconds_value = chunk_spin.value()
                self.boundary_search_seconds_value = boundary_spin.value()
                self.overlap_seconds_value = overlap_spin.value()
                self.silence_threshold_db_value = threshold_spin.value()
                self.silence_min_duration_value = silence_duration_spin.value()
                self.save_settings()
                self.update_selected_file_view()
            log_timer.stop()

        def format_file_size(self, path: Path) -> str:
            try:
                size = path.stat().st_size
            except OSError:
                return "-"
            units = ["B", "KB", "MB", "GB"]
            value = float(size)
            unit = units[0]
            for unit in units:
                if value < 1024 or unit == units[-1]:
                    break
                value /= 1024
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"

        def selected_formats(self) -> tuple[str, ...]:
            return ("txt",)

        def reveal_in_file_manager(self, path: Path) -> None:
            target = path.expanduser()
            if not target.exists():
                QMessageBox.warning(self, "文件不存在", f"找不到文件：{target}")
                self.update_selected_file_view()
                return
            try:
                system = platform.system()
                if system == "Darwin":
                    subprocess.run(["open", "-R", str(target)], check=False)
                elif system == "Windows":
                    subprocess.run(["explorer", f"/select,{target}"], check=False)
                else:
                    directory = target if target.is_dir() else target.parent
                    subprocess.run(["xdg-open", str(directory)], check=False)
            except Exception as exc:
                QMessageBox.critical(self, "打开失败", str(exc))

        def open_file(self, path: Path) -> None:
            target = path.expanduser()
            if not target.exists():
                QMessageBox.warning(self, "文件不存在", f"找不到文件：{target}")
                self.update_selected_file_view()
                return
            try:
                system = platform.system()
                if system == "Darwin":
                    subprocess.run(["open", str(target)], check=False)
                elif system == "Windows":
                    os.startfile(str(target))  # type: ignore[attr-defined]
                else:
                    subprocess.run(["xdg-open", str(target)], check=False)
            except Exception as exc:
                QMessageBox.critical(self, "打开失败", str(exc))

        def open_text_result(self) -> None:
            if self.result_txt_path is None:
                return
            self.open_file(self.result_txt_path)

        def open_result_folder(self) -> None:
            if self.result_txt_path is not None and self.result_txt_path.exists():
                self.reveal_in_file_manager(self.result_txt_path)
            elif self.selected_audio_path is not None:
                self.reveal_in_file_manager(self.selected_audio_path)

        def delete_text_result(self) -> None:
            path = self.result_txt_path
            if path is None or not path.exists():
                self.update_selected_file_view()
                return
            answer = QMessageBox.question(
                self,
                "删除文本文件",
                f"确定删除“{path.name}”吗？此操作无法撤销。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
            try:
                path.unlink()
            except OSError as exc:
                QMessageBox.critical(self, "删除失败", str(exc))
                return
            self.progress_logs.append(f"已删除文本文件：{path.name}")
            self.status_label.setText("文本文件已删除")
            self.update_selected_file_view()

        def start_job(self) -> None:
            self.save_settings()
            audio_path = self.selected_audio_path
            if audio_path is None or not audio_path.exists():
                QMessageBox.warning(self, "未选择文件", "请先选择一个音频文件。")
                return
            formats = self.selected_formats()
            request = TranscriptionRequest(
                input_dir=audio_path.parent,
                output_dir=audio_path.parent,
                audio_files=(audio_path,),
                recursive=False,
                formats=formats,
                overwrite=self.overwrite_value,
                chunk_seconds=self.chunk_seconds_value,
                boundary_search_seconds=self.boundary_search_seconds_value,
                overlap_seconds=self.overlap_seconds_value,
                silence_threshold_db=self.silence_threshold_db_value,
                silence_min_duration=self.silence_min_duration_value,
                language=self.language_value,
                engine=self.engine_value,
                device=self.device_value,
                model=self.model_value,
                model_dir=Path(self.model_dir_value) if self.model_dir_value.strip() else None,
                speaker_diarization=self.speaker_diarization_value,
            )
            try:
                job = self.manager.submit(request)
            except Exception as exc:
                QMessageBox.critical(self, "任务创建失败", str(exc))
                return
            self.current_job_id = job.id
            self.rendered_log_count = 0
            self.progress_logs = []
            self.result_txt_path = self.planned_txt_path(audio_path)
            self.result_folder_path = audio_path.parent
            self.result_status_text = "正在生成"
            self.choose_input_button.setEnabled(False)
            self.start_button.setEnabled(False)
            self.result_file_button.setEnabled(False)
            self.result_menu_button.setEnabled(False)
            self.result_file_button.setText("正在生成文本文件…")
            self.result_status_label.setText(self.result_status_text)
            self.status_label.setText("转写中")
            self.start_button.set_action_text("正在准备转写 · 0%")
            self.start_button.set_progress(0.0)
            append_memory_log("gui_job_submitted", job.id)
            self.timer.start()

        def refresh_job(self) -> None:
            if self.current_job_id is None:
                return
            try:
                job = self.manager.get(self.current_job_id)
            except KeyError:
                self.timer.stop()
                self.status_label.setText("任务不存在")
                self.start_button.setEnabled(True)
                self.choose_input_button.setEnabled(True)
                return
            progress_percent = round(job.progress * 100)
            self.start_button.set_action_text(self.progress_button_text(job.stage, progress_percent))
            self.start_button.set_progress(job.progress)
            self.status_label.setText(job.message)
            new_logs = job.logs[self.rendered_log_count :]
            if new_logs:
                self.progress_logs.extend(new_logs)
                self.rendered_log_count = len(job.logs)
            if job.status in {"completed", "failed"}:
                finished_job_id = self.current_job_id
                self.timer.stop()
                if job.status == "failed":
                    self.status_label.setText("转写失败")
                    self.start_button.set_progress(None)
                    self.start_button.set_action_text("重新转写")
                    self.progress_logs.append(f"转写失败：{job.error or job.message}")
                    append_memory_log("gui_job_failed", f"{finished_job_id} | {memory_summary()}")
                    QMessageBox.critical(self, "转写失败", job.error or job.message)
                else:
                    if job.results:
                        result = job.results[0]
                        self.result_txt_path = result.txt_path or self.result_txt_path
                        self.result_folder_path = result.source.parent
                    self.status_label.setText("转写完成")
                    self.start_button.set_progress(1.0)
                    self.start_button.set_action_text("转写完成")
                    summary = memory_summary()
                    elapsed_seconds = max(0.0, (job.finished_at or 0.0) - (job.started_at or 0.0))
                    self.result_status_text = f"转写完成，耗时 {format_elapsed(elapsed_seconds)}，文件："
                    self.progress_logs.append("转写完成，文本文件已生成。")
                    append_memory_log("gui_job_completed", f"{finished_job_id} | {summary}")
                if finished_job_id is not None:
                    self.manager.forget(finished_job_id)
                self.current_job_id = None
                gc.collect()
                append_memory_log("gui_job_cleanup", memory_summary())
                self.choose_input_button.setEnabled(True)
                self.update_selected_file_view()
                self.start_button.set_progress(None)
                self.start_button.set_action_text("再次转写" if job.status == "completed" else "重新转写")

        def record_idle_memory(self) -> None:
            append_memory_log("gui_idle", f"job={self.current_job_id or 'none'}")

    app = QApplication(sys.argv if argv is None else ["localasr-gui", *argv])
    icon_path = resolve_asset("localasr-icon.png")
    if icon_path is not None:
        app.setWindowIcon(QIcon(str(icon_path)))
    window = MainWindow()
    window.show()
    return app.exec()


def smoke_test() -> int:
    import importlib.util

    from .audio import resolve_audio_tool

    try:
        from PySide6 import QtCore, QtGui, QtWidgets
    except Exception as exc:
        print(f"smoke-test failed: 运行时依赖导入失败：{exc}", file=sys.stderr)
        return 1
    checks = {
        "ffmpeg": resolve_audio_tool("ffmpeg"),
        "ffprobe": resolve_audio_tool("ffprobe"),
        "PySide6.QtCore": Path(QtCore.__file__),
        "PySide6.QtGui": Path(QtGui.__file__),
        "PySide6.QtWidgets": Path(QtWidgets.__file__),
    }
    if os.environ.get("LOCALASR_ENGINE") == "sherpa-onnx":
        checks["sherpa_onnx"] = bool(importlib.util.find_spec("sherpa_onnx"))
        checks["onnx-runtime-models"] = bundled_onnx_models_dir()
    else:
        funasr_spec = importlib.util.find_spec("funasr")
        funasr_version = None
        if funasr_spec is not None and funasr_spec.origin:
            funasr_version = Path(funasr_spec.origin).with_name("version.txt")
        checks["funasr/version.txt"] = funasr_version
    missing = []
    for name, value in checks.items():
        if value is None or value is False:
            missing.append(name)
        elif value is not True and not Path(str(value)).exists():
            missing.append(name)
    if missing:
        print(f"smoke-test failed: 缺少 {', '.join(missing)}", file=sys.stderr)
        return 1
    print("smoke-test ok")
    return 0
