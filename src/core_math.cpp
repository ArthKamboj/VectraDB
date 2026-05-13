#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

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

int main() {

    Vector v1 = {1.5f, 2.0f, 3.1f};
    Vector v2 = {1.0f, 2.5f, 3.0f};

    cout << "Euclidean Distance: " << VectorMath::euclidean_distance(v1, v2) << "\n";
    cout << "Cosine Similarity:  " << VectorMath::cosine_similarity(v1, v2) << "\n";
}