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
            int nprobe = request->nprobe();
            string filter = request->filter_category();

            auto results = index.search(query, k, nprobe, store, filter);

            for (const auto& res : results) {
                vectordb::SearchResult* grpc_res = reply->add_results();
                grpc_res->set_id(res.id);
                grpc_res->set_distance(res.distance);
                grpc_res->set_category(res.category);
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
        } catch (const std::exception& e) {
            reply->set_success(false);
            reply->set_message(std::string("Snapshot failed: ") + e.what());
            return grpc::Status(grpc::StatusCode::INTERNAL, e.what());
        }
    }
};

void RunServer() {
    string server_address("0.0.0.0:50051");
    
    PersistentDocumentStore my_store("production.wal", "snapshot.bin");
    HNSWIndex my_index;
    
    cout << "Training the IVF Index with initial data...\n";
    vector<Vector> training_data;
    mt19937 rng(1337);
    uniform_real_distribution<float> dist(0.0f, 10.0f);
    
    for(int i = 0; i < 100; i++) {
        training_data.push_back({dist(rng), dist(rng), dist(rng)});
    }
    
    my_index.train(training_data, 5, 10);
    VectorDatabaseImpl service(my_store, my_index);

    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    
    unique_ptr<Server> server(builder.BuildAndStart());
    cout << "Vector Database listening on " << server_address << endl;

    server->Wait();
}

int main() {
    RunServer();
    return 0;
}