from __future__ import annotations

import json
import os
import typing as t
from pathlib import Path
import cv2
import ultralytics
import bentoml
from bentoml.validators import ContentType
import numpy as np
import enum
from embedding import SentenceTransformers, BentoMLEmbeddings

Image = t.Annotated[Path, ContentType("image/*")]
os.environ["YOLO_MODEL"] = "yolov11n-face.pt"

class UserRequestStatus(enum.Enum):
    something_went_wrong = 5
    ok = 4
    already_in_the_database = 3
    no_face_in_the_photo = 2
    is_not_registered = 1

@bentoml.service(resources={"gpu": 1})
class YoloService:
    def __init__(self):
        from ultralytics import YOLO
        yolo_model = os.getenv("YOLO_MODEL", "yolov11n-face.pt")
        self.model = YOLO(yolo_model)
        embedding_service = SentenceTransformers()
        self.embed_model = BentoMLEmbeddings(embedding_service)
    
    def render(self, image: Image) -> Image:
        result = self.model.predict(image)[0]
        output = image.parent.joinpath(f"{image.stem}_result{image.suffix}")
        result.save(str(output))
        return output
    
    @bentoml.api
    def add_user(self, image: Image) -> UserRequestStatus:
        result = self.model.predict(image)[0]
        faces = self.face_det(result.boxes.xyxy.tolist(), image)
        if len(faces) == 0:
            return UserRequestStatus.no_face_in_the_photo
        
        face_emb = self.face_vec(faces[0])
        #nearest_face, nearest_prob = dbSearch()
        #if nearest_prob > 0.9:
            #return UserRequestStatus.already_in_the_database
        #try
            #dbAddUser()
        #catch
            #return UserRequestStatus.something_went_wrong
        return UserRequestStatus.ok
    
    @bentoml.api
    def find_user(self, image: Image) -> UserRequestStatus:
        result = self.model.predict(image)[0]
        faces = self.face_det(result.boxes.xyxy.tolist(), image)
        if len(faces) == 0:
            return UserRequestStatus.no_face_in_the_photo
        
        face_emb = self.face_vec(faces[0])
        #nearest_face, nearest_prob = dbSearch()
        #if nearest_prob > 0.9:
            #return UserRequestStatus.ok
        #try
            #dbAddUser()
        #catch
            #return UserRequestStatus.something_went_wrong

    def predict(self, images: list[Image]) -> list[list[dict]]: #нужно ли это???
        results = self.model.predict(source=images)
        return [json.loads(result.tojson()) for result in results]
    
    def face_det(self, boxes: list[list[float]], srcImage: Image) -> list[np.ndarray]:
        img = cv2.imread(srcImage)
        crop_faces_list = []

        # crop method. Перебрать ограничивающие рамки
        for i, box in enumerate(boxes):
            print(i, box)
            x1, y1, x2, y2 = box
            crop_object = img[int(y1):int(y2), int(x1):int(x2)]
            crop_faces_list.append(crop_object)

        return crop_faces_list

    def face_vec(face: np.ndarray):
        oneLineArr = np.concatenate(np.concatenate(face)) #из многомерного numpy В одномерный numpy

        strings_list = []
        for color in oneLineArr:
            strings_list.append(str(color))

        #face_vector = self.embed_model.


        return
