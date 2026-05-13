#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <queue>
#include <algorithm>

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

struct SearchResult {
    size_t id;
    float distance;
    string category;

    bool operator<(const SearchResult& other) const {
        return distance < other.distance; 
    }
};

class DocumentStore {
    private:
        vector<Document> documents;

    public:
        void add(const Vector& vec, const string& category) {
            size_t new_id = documents.size();
            documents.push_back({new_id, vec, category, false});
        }
        void remove(size_t id) {
            if(id<documents.size()) documents[id].is_deleted = true;
        }

        vector<SearchResult> search_with_filter(const Vector& query, int k, const string& filter_category) {
            priority_queue<SearchResult> max_heap;

            for (const auto& doc : documents) {
                
                if(doc.is_deleted) continue;
                if(!filter_category.empty() && doc.category!=filter_category) continue;

                float dist = VectorMath::euclidean_distance(query, doc.embedding);

                if(max_heap.size()<k) {
                    max_heap.push({doc.id, dist, doc.category});
                }
                else if(dist < max_heap.top().distance) {
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

    DocumentStore db;
    

    db.add({1.0f, 2.0f, 3.0f}, "electronics");
    db.add({1.5f, 2.5f, 3.5f}, "clothing");
    db.add({8.0f, 8.0f, 8.0f}, "electronics");
    db.add({0.9f, 2.1f, 3.1f}, "clothing");
    
    Vector query = {1.0f, 2.0f, 3.0f};


    cout << "--- All Active Documents ---\n";
    auto results1 = db.search_with_filter(query, 2, "");
    for (const auto& res : results1) {
        cout << "ID: " << res.id << " | Dist: " << res.distance << " | Cat: " << res.category << "\n";
    }


    cout << "\n--- Deleting ID 0 & Filtering by 'clothing' ---\n";
    db.remove(0);
    
    auto results2 = db.search_with_filter(query, 2, "clothing");
    for (const auto& res : results2) {
        cout << "ID: " << res.id << " | Dist: " << res.distance << " | Cat: " << res.category << "\n";
    }

    return 0;
}