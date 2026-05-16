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
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <cmath>
#include <immintrin.h>

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

            size_t n=a.size();
            size_t i=0;

           __m256 sum_vec = _mm256_setzero_ps();
           for (; i+7<n; i+=8) {
                __m256 v_a = _mm256_loadu_ps(&a[i]);
                __m256 v_b = _mm256_loadu_ps(&b[i]);
                __m256 diff = _mm256_sub_ps(v_a, v_b);
                __m256 sq_diff = _mm256_mul_ps(diff, diff);
                sum_vec = _mm256_add_ps(sum_vec, sq_diff);
            }

            alignas(32) float temp[8];
            _mm256_storeu_ps(temp, sum_vec);
            float sum = temp[0] + temp[1] + temp[2] + temp[3] + 
                        temp[4] + temp[5] + temp[6] + temp[7];

            for (; i<n; i++) {
                float diff = a[i] - b[i];
                sum += diff*diff;
            }
            return sum;
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

    bool operator>(const SearchResult& other) const {
        return distance > other.distance; 
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
    PersistentDocumentStore(const string& w_path, const string& s_path) : wal_path(w_path) {
        load_snapshot(s_path);
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

    bool load_snapshot(const string& snapshot_path) {
        ifstream snap_file(snapshot_path, ios::binary);
        if (!snap_file) return false;

        size_t num_docs;
        snap_file.read(reinterpret_cast<char*>(&num_docs), sizeof(num_docs));

        documents.clear();
        documents.reserve(num_docs);

        for (size_t i = 0; i < num_docs; ++i) {
            Document doc;
            
            snap_file.read(reinterpret_cast<char*>(&doc.id), sizeof(doc.id));
            snap_file.read(reinterpret_cast<char*>(&doc.is_deleted), sizeof(doc.is_deleted));
            
            size_t vec_size;
            snap_file.read(reinterpret_cast<char*>(&vec_size), sizeof(vec_size));
            doc.embedding.resize(vec_size);
            snap_file.read(reinterpret_cast<char*>(doc.embedding.data()), vec_size * sizeof(float));

            size_t str_len;
            snap_file.read(reinterpret_cast<char*>(&str_len), sizeof(str_len));
            doc.category.resize(str_len);
            snap_file.read(&doc.category[0], str_len);

            documents.push_back(doc);
        }
        cout << "Loaded snapshot containing " << documents.size() << " documents.\n";
        return true;
    }

    void create_snapshot(const string& snapshot_path) {
        unique_lock<shared_mutex> lock(store_mutex); 

        ofstream snap_file(snapshot_path, ios::binary | ios::trunc);
        if (!snap_file) throw runtime_error("Failed to create snapshot file!");

        size_t num_docs = documents.size();
        snap_file.write(reinterpret_cast<const char*>(&num_docs), sizeof(num_docs));

        for (const auto& doc : documents) {
            snap_file.write(reinterpret_cast<const char*>(&doc.id), sizeof(doc.id));
            snap_file.write(reinterpret_cast<const char*>(&doc.is_deleted), sizeof(doc.is_deleted));
            
            size_t vec_size = doc.embedding.size();
            snap_file.write(reinterpret_cast<const char*>(&vec_size), sizeof(vec_size));
            snap_file.write(reinterpret_cast<const char*>(doc.embedding.data()), vec_size * sizeof(float));
            
            size_t str_len = doc.category.length();
            snap_file.write(reinterpret_cast<const char*>(&str_len), sizeof(str_len));
            snap_file.write(doc.category.c_str(), str_len);
        }

        snap_file.flush();
        snap_file.close();

        if (wal_file.is_open()) wal_file.close();
        wal_file.open(wal_path, ios::binary | ios::trunc);
        wal_file.close(); 
        wal_file.open(wal_path, ios::binary | ios::app); 

        cout << "Snapshot saved and WAL compacted successfully.\n";
    }
};

struct GraphNode {
    size_t id;
    vector<size_t> friends;
};

class NSWGraph {
    private:
        vector<GraphNode> nodes;
        size_t entry_point_id;
        bool has_entry_point = false;
        int max_friends = 16; 

    public:
        vector<SearchResult> search(const Vector& query, int k, const PersistentDocumentStore& store) {
            if (!has_entry_point) return {};

            unordered_set<size_t> visited;
            
            priority_queue<SearchResult, vector<SearchResult>, greater<SearchResult>> candidates;
            
            priority_queue<SearchResult> top_results;

            const Document& entry_doc = store.get_document(entry_point_id);
            float entry_dist = VectorMath::euclidean_distance(query, entry_doc.embedding);
            
            candidates.push({entry_point_id, entry_dist, entry_doc.category});
            top_results.push({entry_point_id, entry_dist, entry_doc.category});
            visited.insert(entry_point_id);

            while (!candidates.empty()) {
                SearchResult current = candidates.top();
                candidates.pop();
                if (current.distance > top_results.top().distance && top_results.size() == k) {
                    break; 
                }

                for (size_t friend_id : nodes[current.id].friends) {
                    if (visited.find(friend_id) != visited.end()) continue;
                    visited.insert(friend_id);

                    const Document& friend_doc = store.get_document(friend_id);
                    if (friend_doc.is_deleted) continue;

                    float dist = VectorMath::euclidean_distance(query, friend_doc.embedding);

                    if (top_results.size() < k || dist < top_results.top().distance) {
                        candidates.push({friend_id, dist, friend_doc.category});
                        top_results.push({friend_id, dist, friend_doc.category});
                        
                        if (top_results.size() > k) {
                            top_results.pop();
                        }
                    }
                }
            }

            vector<SearchResult> results;
            while (!top_results.empty()) {
                results.push_back(top_results.top());
                top_results.pop();
            }
            reverse(results.begin(), results.end());
            return results;
        }

        void add_node(size_t new_id, const PersistentDocumentStore& store) {
            if (new_id >= nodes.size()) {
                nodes.resize(new_id + 1);
            }
            nodes[new_id].id = new_id;

            if (!has_entry_point) {
                entry_point_id = new_id;
                has_entry_point = true;
                return;
            }

            const Document& new_doc = store.get_document(new_id);
            vector<SearchResult> closest_neighbors = search(new_doc.embedding, max_friends, store);

            for (const auto& neighbor : closest_neighbors) {
                nodes[new_id].friends.push_back(neighbor.id);
                nodes[neighbor.id].friends.push_back(new_id);
            }
        }
};

struct HNSWNode {
    size_t doc_id;
    Vector embedding;
    int max_layer;
    vector<vector<size_t>> neighbors; 
};

class HNSWIndex {
    private:
        unordered_map<size_t, HNSWNode> nodes;
        int max_graph_layer = 0;

        const int M = 16;
        const int M0 = 32;
        const int ef_search = 50;
        const double mult = 1/log(1.0*M);

        int generate_random_layer() {
            double r = ((double) rand() / (RAND_MAX));
            if (r == 0.0) r = 0.00001;
            return (int)(-log(r)*mult);
        }

    public:
        HNSWIndex() {
            srand(1337);
        }
};