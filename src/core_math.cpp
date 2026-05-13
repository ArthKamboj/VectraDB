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

struct SearchResult {
    size_t id;
    float distance;

    bool operator<(const SearchResult& other) const {
        return distance < other.distance; 
    }
};

class FlatIndex {
    private:
        vector<Vector> data;

    public:
        void add(const Vector& vec) {
            data.push_back(vec);
        }
        vector<SearchResult> search(const Vector& query, int k) {
            priority_queue<SearchResult> max_heap;

            for(size_t i=0; i<data.size(); i++) {
                float dist = VectorMath::euclidean_distance(query, data[i]);
                if(max_heap.size()<k) {
                    max_heap.push({i, dist});
                }

                else if(dist < max_heap.top().distance) {
                    max_heap.pop();
                    max_heap.push({i, dist});
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
        size_t size() const {
            return data.size();
        }
};

int main() {

    FlatIndex db;
    
    db.add({1.0f, 2.0f, 3.0f});
    db.add({1.5f, 2.5f, 3.5f});
    db.add({8.0f, 8.0f, 8.0f});
    db.add({0.9f, 2.1f, 3.1f});
    db.add({-1.0f, -2.0f, -3.0f});

    Vector query = {1.0f, 2.0f, 3.0f};
    int k=2;

    cout << "Searching for top " << k << " matches...\n";
    vector<SearchResult> results = db.search(query, k);

    for (const auto& res : results) {
        std::cout << "Vector ID: " << res.id << " | Distance: " << res.distance << "\n";
    }

    return 0;
}