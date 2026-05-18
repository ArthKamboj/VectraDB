from fastapi import FastAPI, File, UploadFile, Form
from sentence_transformers import SentenceTransformer
from PIL import Image
import grpc
import io
import os
import vectordb_pb2
import vectordb_pb2_grpc

app = FastAPI(title="VectraDB AI Gateway")

os.makedirs("uploads", exist_ok=True)


print("Loading AI Vision Model (CLIP)...")
model = SentenceTransformer('clip-ViT-B-32')
print("Model loaded successfully.")


channel = grpc.insecure_channel('vectradb:50051')
stub = vectordb_pb2_grpc.VectorDatabaseStub(channel)

@app.post("/upload/")
async def upload_image(category: str = Form(...),
                       user_id : str = Form(...),                       
                       file: UploadFile = File(...),
):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    vector_embedding = model.encode(image).tolist()

    request = vectordb_pb2.InsertRequest(
        embedding=vectordb_pb2.Vector(elements=vector_embedding),
        category=category,
        user_id=user_id
    )
    
    try:
        response = stub.Insert(request)
        
        image_path = f"uploads/{user_id}_{response.id}.jpg"
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(image_path, format="JPEG")
        
        return {"status": "success", "db_id": response.id, "category": category}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/search/")
async def search_image(k: int = Form(5), 
                       user_id : str = Form(...),
                       file: UploadFile = File(...)
):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    vector_embedding = model.encode(image).tolist()

    request = vectordb_pb2.SearchRequest(
        query=vectordb_pb2.Vector(elements=vector_embedding),
        k=k,
        filter_category="",
        user_id=user_id
    )
    
    try:
        response = stub.Search(request)
        results = [{"id": res.id, "distance": res.distance, "category": res.category} for res in response.results]
        return {"status": "success", "matches": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}