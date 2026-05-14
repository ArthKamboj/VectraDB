#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <queue>
#include <algorithm>
#include <random>
#include <mutex>
#include <shared_mutex>
#include <fstream>

using namespace std;

using Vector = vector<float>;

class VectorMath {

    public:
        static float dot_product(const Vector& a, const Vector& b) {
            if(a.size() != b.size()) {
                throw runtime_error("Vectors must be of same dimensionality");
            }
            float result = 0.0f;
            for(size_t i=0; i<a.size(); i++) {
                result += a[i]*b[i];
            }
            return result;
        }

        static float euclidean_distance(const Vector& a, const Vector& b) {
            if(a.size() != b.size()) {
                throw runtime_error("Vectors must be of same dimensionality");
            }
            float sum_sq_diff = 0.0f;
            for(size_t i=0; i<a.size(); i++) {
                float diff = a[i]-b[i];
                sum_sq_diff += diff*diff;
            }
            return sqrt(sum_sq_diff);
        }

        static float cosine_similarity(const Vector& a, const Vector& b) {
            float dot = dot_product(a,b);
            float norm_a = sqrt(dot_product(a,a));
            float norm_b = sqrt(dot_product(b,b));
            if(norm_a == 0.0f || norm_b == 0.0f) {
                return 0.0f;
            }
            return (dot/(norm_a*norm_b));
        }
};

struct Document {
    size_t id;
    Vector embedding;
    string category;
    bool is_deleted;
};

enum class Opcode : char {
    ADD = 'A',
    REMOVE = 'R'
};

struct SearchResult {
    size_t id;
    float distance;
    string category;

    bool operator<(const SearchResult& other) const {
        return distance < other.distance; 
    }
};

class PersistentDocumentStore {
private:
    vector<Document> documents;
    mutable shared_mutex store_mutex;
    
    ofstream wal_file;
    string wal_path;

    void log_add(const Vector& vec, const string& category) {
        if (!wal_file.is_open()) return;
        
        char op = static_cast<char>(Opcode::ADD);
        wal_file.write(&op, sizeof(op));

        size_t vec_size = vec.size();
        wal_file.write(reinterpret_cast<const char*>(&vec_size), sizeof(vec_size));
        wal_file.write(reinterpret_cast<const char*>(vec.data()), vec_size * sizeof(float));

        size_t str_len = category.length();
        wal_file.write(reinterpret_cast<const char*>(&str_len), sizeof(str_len));
        wal_file.write(category.c_str(), str_len);
        
        wal_file.flush();
    }

    void log_remove(size_t id) {
        if (!wal_file.is_open()) return;

        char op = static_cast<char>(Opcode::REMOVE);
        wal_file.write(&op, sizeof(op));
        wal_file.write(reinterpret_cast<const char*>(&id), sizeof(id));
        
        wal_file.flush();
    }

public:
    PersistentDocumentStore(const string& path) : wal_path(path) {
        recover_from_wal();
        wal_file.open(wal_path, ios::binary | ios::app);
        if (!wal_file) {
            throw runtime_error("Failed to open WAL file!");
        }
    }

    ~PersistentDocumentStore() {
        if (wal_file.is_open()) wal_file.close();
    }

    const Document get_document(size_t id) const {
        shared_lock<shared_mutex> lock(store_mutex);
        if (id >= documents.size()) throw out_of_range("Invalid ID");
        return documents[id];
    }

    void recover_from_wal() {
        ifstream in_file(wal_path, ios::binary);
        if (!in_file) {
            cout << "No existing WAL found. Starting fresh.\n";
            return;
        }

        cout << "Replaying Write-Ahead Log...\n";
        while (in_file.peek() != EOF) {
            char op;
            in_file.read(&op, sizeof(op));

            if (op == static_cast<char>(Opcode::ADD)) {
                size_t vec_size;
                in_file.read(reinterpret_cast<char*>(&vec_size), sizeof(vec_size));
                
                Vector vec(vec_size);
                in_file.read(reinterpret_cast<char*>(vec.data()), vec_size * sizeof(float));

                size_t str_len;
                in_file.read(reinterpret_cast<char*>(&str_len), sizeof(str_len));
                
                string category(str_len, '\0');
                in_file.read(&category[0], str_len);

                size_t new_id = documents.size();
                documents.push_back({new_id, vec, category, false});
            } 
            else if (op == static_cast<char>(Opcode::REMOVE)) {
                size_t id;
                in_file.read(reinterpret_cast<char*>(&id), sizeof(id));
                if (id < documents.size()) {
                    documents[id].is_deleted = true;
                }
            }
        }
        cout << "Recovered " << documents.size() << " documents.\n";
    }

    size_t add(const Vector& vec, const string& category) {
        unique_lock<shared_mutex> lock(store_mutex);
        
        size_t new_id = documents.size();
        documents.push_back({new_id, vec, category, false});
        
        log_add(vec, category);
        return new_id;
    }

    void remove(size_t id) {
        unique_lock<shared_mutex> lock(store_mutex);
        
        if (id < documents.size() && !documents[id].is_deleted) {
            documents[id].is_deleted = true;
            log_remove(id);
        }
    }

    size_t size() const {
        shared_lock<shared_mutex> lock(store_mutex);
        return documents.size();
    }
};

    struct Cluster {
        Vector centroid;
        vector<size_t> document_ids;
};

class IVFIndex {
    private:
        vector<Cluster> clusters;

    public:
        void train(const vector<Vector>& training_data, int num_clusters, int iterations=10) {
            if(training_data.size() < num_clusters) {
                throw runtime_error("Not enough data to form clusters");
            }

            mt19937 rng(42);
            uniform_int_distribution<size_t> dist(0, training_data.size()-1);

            for (int i=0; i<num_clusters; i++) {
                Cluster c;
                c.centroid = training_data[dist(rng)];
                clusters.push_back(c);
            }

            for (int iter = 0; iter < iterations; ++iter) {
                vector<vector<Vector>> new_buckets(num_clusters);

                for(const auto& vec : training_data) {
                    int best_cluster = find_closest_centroid(vec);
                    new_buckets[best_cluster].push_back(vec);
                }

                for(int i=0; i<num_clusters; ++i) {
                    if(new_buckets[i].empty()) continue;
                    Vector new_centroid(training_data[0].size(), 0.0f);
                    for (const auto& vec : new_buckets[i]) {
                        for (size_t d = 0; d < vec.size(); ++d) {
                            new_centroid[d] += vec[d];
                        }
                    }
                    for (size_t d = 0; d < new_centroid.size(); ++d) {
                        new_centroid[d] /= new_buckets[i].size();
                    }
                    clusters[i].centroid = new_centroid;
                }
            }
            cout << "IVF Index trained with " << num_clusters << " clusters.\n";
        }

        int find_closest_centroid (const Vector& vec) {
            int best_idx = -1;
            float min_dist = numeric_limits<float>::max();
            for (size_t i=0; i<clusters.size(); i++) {
                float dist = VectorMath::euclidean_distance(vec, clusters[i].centroid);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_idx = i;
                }
            }
            return best_idx;
        }

        vector<SearchResult> search (const Vector& query, int k, int nprobe, PersistentDocumentStore& store, const string& filter="") {
            priority_queue<pair<float, int>> closest_clusters;
            
            for(size_t i=0; i<closest_clusters.size(); i++) {
                float dist = VectorMath::euclidean_distance(query, clusters[i].centroid);
                closest_clusters.push({-dist, i});
            }

            vector<size_t> candidate_ids;
            for (int p = 0; p < nprobe && !closest_clusters.empty(); ++p) {
                int cluster_idx = closest_clusters.top().second;
                closest_clusters.pop();
                
                for (size_t doc_id : clusters[cluster_idx].document_ids) {
                    candidate_ids.push_back(doc_id);
                }
            }

            priority_queue<SearchResult> max_heap;
            for(size_t id : candidate_ids) {
                const Document& doc = store.get_document(id);

                if(doc.is_deleted) continue;
                if(!filter.empty() && doc.category!=filter) continue;

                float dist = VectorMath::euclidean_distance(query, doc.embedding);
                if (max_heap.size() < k) {
                    max_heap.push({doc.id, dist, doc.category});
                }
                else if (dist < max_heap.top().distance) {
                    max_heap.pop();
                    max_heap.push({doc.id, dist, doc.category});
                }
            }

            vector<SearchResult> results;
            while(!max_heap.empty()) {
                results.push_back(max_heap.top());
                max_heap.pop();
            }
            reverse(results.begin(), results.end());
            return results;
        }   
};

int main() {

    
    return 0;
}