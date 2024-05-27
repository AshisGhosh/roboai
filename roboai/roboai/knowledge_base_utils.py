import os
import json
import pydantic
from datetime import datetime


class KnowledgeBaseItem(pydantic.BaseModel):
    key: str
    value: pydantic.Json
    tags: list[str] = []
    timestamp: str = pydantic.Field(default_factory=lambda: datetime.now().isoformat())


class KnowledgeBase:
    def __init__(self, file_path: str = "/app/roboai/knowledge_base.json"):
        self.file_path = file_path
        self.data = self.load_data()

    def load_data(self):
        print("Current working directory:", os.getcwd())
        print("Files in the directory:", os.listdir())
        with open(self.file_path, "r") as f:
            data = json.load(f)
        return data

    @property
    def all_data(self):
        return self.data

    @property
    def knowledge(self):
        return [
            KnowledgeBaseItem(
                key=key,
                value=value["value"],
                tags=value["tags"],
                timestamp=value["timestamp"],
            )
            for key, value in self.data.items()
        ]

    def get_knowledge_as_string(self):
        return "\n".join([f"{value['value']}" for value in self.data.values()])

    def get_data(self, key: str):
        return self.data.get(key, None)

    def add_data(self, key: str, value, tags: list[str] = []):
        self.data[key] = {
            "value": value,
            "tags": tags,
            "timestamp": datetime.now().isoformat(),
        }
        self.save_data()

    def save_data(self):
        with open(self.file_path, "w") as f:
            json.dump(self.data, f, indent=4)
