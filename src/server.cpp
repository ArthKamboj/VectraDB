#include <iostream>
#include <memory>
#include <string>
#include <grpcpp/grpcpp.h>
#include <random>

#include "vectordb.grpc.pb.h" 
#include "core_math.cpp" 

using namespace std;

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using vectordb::VectorDatabase;
using vectordb::InsertRequest;
using vectordb::InsertResponse;
using vectordb::SearchRequest;
using vectordb::SearchResponse;

class VectorDatabaseImpl final : public VectorDatabase::Service {
private:
    PersistentDocumentStore& store;
    HNSWIndex& index;

public:
    VectorDatabaseImpl(PersistentDocumentStore& s, HNSWIndex& i) : store(s), index(i) {}

    Status Insert(ServerContext* context, const InsertRequest* request, InsertResponse* reply) override {
        try {
            Vector vec(request->embedding().elements().begin(), request->embedding().elements().end());
            string category = request->category();

            size_t new_id = store.add(vec, category);
            
            index.add(new_id, vec);

            reply->set_id(new_id);
            reply->set_success(true);
            return Status::OK;
        } catch (const exception& e) {
            return Status(grpc::StatusCode::INTERNAL, e.what());
        }
    }

    Status Search(ServerContext* context, const SearchRequest* request, SearchResponse* reply) override {
        try {
            Vector query(request->query().elements().begin(), request->query().elements().end());
            int k = request->k();
            string filter = request->filter_category();

            auto result_ids = index.search(query, k * 5);

            int matched = 0;
            for (size_t id : result_ids) {
                if (matched >= k) break;

                const auto& doc = store.documents[id];

                if (!filter.empty() && doc.category != filter) {
                    continue; 
                }

                float dist = VectorMath::euclidean_distance(query, doc.embedding);

                vectordb::SearchResult* grpc_res = reply->add_results();
                grpc_res->set_id(id);
                grpc_res->set_distance(dist);
                grpc_res->set_category(doc.category);
                
                matched++;
            }

            return Status::OK;
        } catch (const exception& e) {
            return Status(grpc::StatusCode::INTERNAL, e.what());
        }
    }
    
    grpc::Status TriggerSnapshot(grpc::ServerContext* context, const vectordb::SnapshotRequest* request, vectordb::SnapshotResponse* reply) override {
        try {
            store.create_snapshot("snapshot.bin");
            reply->set_success(true);
            reply->set_message("Snapshot triggered successfully. WAL compacted to 0 bytes.");
            return grpc::Status::OK;
        } catch (const exception& e) {
            reply->set_success(false);
            reply->set_message(string("Snapshot failed: ") + e.what());
            return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
        }
    }
};

void RunServer() {
    string server_address("0.0.0.0:50051");
    
    PersistentDocumentStore my_store("production.wal", "snapshot.bin");
    HNSWIndex my_index;

    VectorDatabaseImpl service(my_store, my_index);

    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    
    unique_ptr<Server> server(builder.BuildAndStart());
    cout << "Vector Database listening on " << server_address << " with HNSW Graph initialized." << endl;

    server->Wait();
}

int main() {
    RunServer();
    return 0;
}