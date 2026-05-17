from fastapi import FastAPI, File, UploadFile, Form
from sentence_transformers import SentenceTransformer
from PIL import Image
import grpc
import io
import vectordb_pb2
import vectordb_pb2_grpc

app = FastAPI(title="VectraDB AI Gateway")

print("Loading AI Vision Model (CLIP)...")
model = SentenceTransformer('clip-ViT-B-32')
print("Model loaded successfully.")

channel = grpc.insecure_channel('vectradb:50051')
stub = vectordb_pb2_grpc.VectorDatabaseStub(channel)

@app.post("/upload/")
async def upload_image(category: str = Form(...), file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    vector_embedding = model.encode(image).tolist()

    request = vectordb_pb2.InsertRequest(
        embedding=vectordb_pb2.Vector(elements=vector_embedding),
        category=category
    )
    
    try:
        response = stub.Insert(request)
        return {"status": "success", "db_id": response.id, "category": category}
    except Exception as e:
        return {"status": "error", "message": str(e)}