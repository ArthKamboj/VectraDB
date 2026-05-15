import grpc
import vectordb_pb2
import vectordb_pb2_grpc
import random
import time

def run():
    
    print("Connecting to C++ Vector Database on localhost:50051...")
    channel = grpc.insecure_channel('localhost:50051')
    stub = vectordb_pb2_grpc.VectorDatabaseStub(channel)

    
    print("\n--- Blasting 100 Vectors into the DB ---")
    categories = ["tech", "art", "music", "science"]
    
    for i in range(100):
        
        vec = [random.uniform(0.0, 10.0), random.uniform(0.0, 10.0), random.uniform(0.0, 10.0)]
        cat = random.choice(categories)
        
        request = vectordb_pb2.InsertRequest(
            embedding=vectordb_pb2.Vector(elements=vec),
            category=cat
        )
        response = stub.Insert(request)
        
        if i % 25 == 0:
            print(f"Inserted Document ID {response.id} | Category: {cat}")

    
    print("\n--- Searching for closest 'art' vectors to [5.0, 5.0, 5.0] ---")
    query_vec = [5.0, 5.0, 5.0]
    
    search_request = vectordb_pb2.SearchRequest(
        query=vectordb_pb2.Vector(elements=query_vec),
        k=5,
        nprobe=1,
        filter_category="art"
    )
    
    
    start_time = time.time()
    search_response = stub.Search(search_request)
    end_time = time.time()

    print(f"Search completed in {(end_time - start_time) * 1000:.2f} ms!")
    for res in search_response.results:
        print(f"ID: {res.id} | Distance: {res.distance:.4f} | Category: {res.category}")

if __name__ == '__main__':
    run()