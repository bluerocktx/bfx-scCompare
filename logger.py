from __future__ import annotations

from collections import defaultdict
from typing import Any


class InternalLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.log = defaultdict(dict)
            cls._instance.warnings = []
        return cls._instance

    def write_log(self, nested_keys: list[str], value: Any):
        current_dict = self.log
        for key in nested_keys[:-1]:
            current_dict = current_dict[key]
        current_dict[nested_keys[-1]] = value

    def write_warning(self, warning: str):
        self.warnings.append(warning)

    def output_log(self):
        return dict(self.log)
