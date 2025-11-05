# src/exception.py
import sys
from datetime import datetime
import traceback

class CustomException(Exception):
    """
    Custom exception class with contextual information (file, line, and traceback).
    """

    def __init__(self, error_message: str, error_detail: sys = None):
        super().__init__(error_message)
        self.error_message = self.format_error_message(error_message, error_detail)

    @staticmethod
    def format_error_message(error_message: str, error_detail: sys = None) -> str:
        _, _, exc_tb = sys.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "N/A"
        line_no = exc_tb.tb_lineno if exc_tb else "N/A"
        tb_info = "".join(traceback.format_exception(*sys.exc_info()))
        formatted_message = (
            f"\n[ERROR] Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            f"\nFile: {file_name}"
            f"\nLine: {line_no}"
            f"\nMessage: {error_message}"
            f"\nTraceback: {tb_info}"
        )
        return formatted_message

    def __str__(self):
        return self.error_message
