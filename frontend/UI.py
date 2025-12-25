import bentoml, streamlit
from pathlib import Path
import typing as t
from bentoml.validators import ContentType
import tempfile
import os

src_image, dst_image = streamlit.columns(2)
result = None
bentoml_host = os.getenv("BENTOML_SERVICE", "http://backend:3000")

Image = t.Annotated[Path, ContentType("image/*")]
upload_file = streamlit.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "bmp", "webp"])

with src_image:
    if upload_file is not None:
        streamlit.image(upload_file, channels="BGR", width=900)
        suffix = Path(upload_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(upload_file.getvalue())
            tmp_path = Path(tmp_file.name)

        with bentoml.SyncHTTPClient(bentoml_host) as client:
            result = client.render(tmp_path)

with dst_image:
    if upload_file is not None:
        streamlit.image(result, channels="BGR", width=900)