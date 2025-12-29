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
import base64
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams, PointStruct
from PIL import Image

CustomImage = t.Annotated[Path, ContentType("image/*")]
os.environ["YOLO_MODEL"] = "yolov11n-face.pt"
db_host = os.getenv("DB_SERVICE", "http://database:6333")

class UserRequestStatus(enum.Enum):
    something_went_wrong = 5
    ok = 4
    already_in_the_database = 3
    no_face_in_the_photo = 2
    is_not_registered = 1

@bentoml.service( traffic={"timeout": 600}, resources={"gpu": 1})
class YoloService:
    #client = QdrantClient(url=db_host)
    client = QdrantClient(":memory:")
    db_collection_name = "photos"

    def __init__(self):
        from ultralytics import YOLO
        yolo_model = os.getenv("YOLO_MODEL", "yolov11n-face.pt")
        self.model = YOLO(yolo_model)
        embedding_service = SentenceTransformers()
        self.embed_model = BentoMLEmbeddings(embedding_service)
        self.client.create_collection(
            collection_name = self.db_collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
    
    def render(self, image: CustomImage) -> CustomImage:
        result = self.model.predict(image)[0]
        output = image.parent.joinpath(f"{image.stem}_result{image.suffix}")
        result.save(str(output))
        return output
    
    @bentoml.api
    def add_user(self, image: CustomImage) -> UserRequestStatus:
        result = self.model.predict(image)[0]
        faces = self.face_det(result.boxes.xyxy.tolist(), image)
        if len(faces) == 0:
            return UserRequestStatus.no_face_in_the_photo
        
        face_emb = self.face_vec(faces[0]) #emb для первого лица из списка "вырезанных" лиц

        hits = self.client.query_points( #пытаемся получить самое похожее лицо
            collection_name=self.db_collection_name,
            query= face_emb,
            limit=1
        )
        
        if (len(hits.points)) == 0: #в бд нет лиц вообще
            self.addUserToDB(face_emb, self.img2string(faces[0]))
            return UserRequestStatus.ok

        if (hits.points[0].score>=0.85): #лицо с фото сильно похоже на найденное в бд лицо
            return UserRequestStatus.already_in_the_database
        else: 
            self.addUserToDB(face_emb, self.img2string(faces[0]))
            return UserRequestStatus.ok
    
    @bentoml.api
    def find_user(self, image: CustomImage) -> UserRequestStatus:
        result = self.model.predict(image)[0]
        faces = self.face_det(result.boxes.xyxy.tolist(), image)
        if len(faces) == 0:
            return UserRequestStatus.no_face_in_the_photo
        
        face_emb = self.face_vec(faces[0])

        hits = self.client.query_points( #пытаемся получить самое похожее лицо
            collection_name=self.db_collection_name,
            query= face_emb,
            limit=1
        )

        if (len(hits.points)) == 0: #в бд нет лиц вообще
            return UserRequestStatus.is_not_registered
        
        if (hits.points[0].score>=0.85): #лицо с фото сильно похоже на найденное в бд лицо
            return UserRequestStatus.ok
        else:
            return UserRequestStatus.is_not_registered

    def predict(self, images: list[CustomImage]) -> list[list[dict]]: #нужно ли это???
        results = self.model.predict(source=images)
        return [json.loads(result.tojson()) for result in results]
    
    def face_det(self, boxes: list[list[float]], srcImage: CustomImage) -> list[np.ndarray]:
        img = cv2.imread(str(srcImage))
        crop_faces_list = []

        # crop method. Перебрать ограничивающие рамки
        for i, box in enumerate(boxes):
            print(i, box)
            x1, y1, x2, y2 = box
            crop_object = img[int(y1):int(y2), int(x1):int(x2)]
            crop_faces_list.append(crop_object)

        return crop_faces_list

    def face_vec(self, face: np.ndarray) -> np.ndarray: 
        oneLineArr = np.concatenate(np.concatenate(face)) #из многомерного numpy в одномерный numpy

        strings_list = []
        for color in oneLineArr:
            strings_list.append(str(color))

        face_vector = self.embed_model._get_query_embedding(strings_list)

        return face_vector

    def addUserToDB(self, vec: np.ndarray, image2String: str):
        self.client.upsert(
            collection_name=self.db_collection_name,
            points=[
                PointStruct(
                    id=uuid.uuid4(),
                    vector=vec,
                    payload={"bin_data": "Пока что нет"}
                )
            ]
        )

    def img2string(self, crop_photo: np.ndarray) -> str:
        #img = Image.fromarray(crop_photo)
        b64string = ""
        #with open(img, "rb") as image:
            #b64string = base64.b64encode(image.read()).decode("UTF-8")
        return b64string