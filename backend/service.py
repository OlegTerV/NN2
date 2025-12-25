from __future__ import annotations

import json
import os
import typing as t
from pathlib import Path

import ultralytics
import bentoml
from bentoml.validators import ContentType

Image = t.Annotated[Path, ContentType("image/*")]
os.environ["YOLO_MODEL"] = "best.pt"

@bentoml.service(resources={"gpu": 1})
class YoloService:
    def __init__(self):
        from ultralytics import YOLO
        yolo_model = os.getenv("YOLO_MODEL", "yolo11n-seg.pt")
        self.model = YOLO(yolo_model)

    @bentoml.api(batchable=True)
    def predict(self, images: list[Image]) -> list[list[dict]]:
        results = self.model.predict(source=images)
        return [json.loads(result.tojson()) for result in results]

    @bentoml.api
    def render(self, image: Image) -> Image:
        result = self.model.predict(image)[0]
        output = image.parent.joinpath(f"{image.stem}_result{image.suffix}")
        result.save(str(output))
        return output