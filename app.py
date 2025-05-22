from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Carrega o modelo treinado
model = YOLO('best.pt')

@app.post("/processar/detectar_cartao")
async def detectar_cartao(imagem: UploadFile = File(...)):
    conteudo = await imagem.read()
    img = Image.open(io.BytesIO(conteudo))

    results = model(img)
    boxes = results[0].boxes

    if len(boxes) == 0:
        return JSONResponse(content={"conf": 0.0, "bbox": []})

    # Seleciona a caixa com maior confiança
    confs = boxes.conf.tolist()
    idx = confs.index(max(confs))

    conf = confs[idx]
    bbox = boxes.xyxy[idx].tolist()  # [x1, y1, x2, y2]

    return JSONResponse(content={"conf": conf, "bbox": bbox})
