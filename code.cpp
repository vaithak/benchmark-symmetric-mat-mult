#include <Eigen/Eigen>
#include <iostream>
#include <chrono>
#include <utility>

using namespace std;
using namespace Eigen;


double xAx_symmetric(const Eigen::MatrixXd& A, const Eigen::VectorXd& x, bool custom_loop)
{
    const auto dim = A.rows();
    if (custom_loop) {
        double sum = 0;
        for (Eigen::Index i = 0; i < dim; ++i) {
            const auto x_i = x[i];
            sum += A(i, i) * x_i * x_i;
            for (Eigen::Index j = 0; j < i; ++j) {
                sum += 2 * A(j, i) * x_i * x[j];
            }
        }
        return sum;
    }
    else {
        return x.transpose() * A.selfadjointView<Eigen::Upper>() * x;
    }
}


// ---------- Code for getting execution time of a function -----------------------------
// Reference: https://stackoverflow.com/a/22387757/7317290
typedef std::chrono::high_resolution_clock::time_point TimeVar;

#define duration(a) std::chrono::duration_cast<std::chrono::nanoseconds>(a).count()
#define timeNow() std::chrono::high_resolution_clock::now()

template<typename F, typename... Args>
double funcTime(F func, Args&&... args){
    TimeVar t1=timeNow();
    func(std::forward<Args>(args)...);
    return duration(timeNow()-t1);
}
//---------------------------------------------------------------------------------------


int main(int argc, char const *argv[])
{
    std::cout << std::scientific;

    int max_dim = 100;
    int hop_dim = 2;
    int instances_per_dim = 100;

    double tol = 0.001;
    cout << "Dimension | Time taken by Custom Loop (in ns) | Time taken using Eigen's selfAdjointView (in ns)" << endl;
    cout << "--- | --- | ---" << endl;

    for(int d=1; d<=max_dim; d+=hop_dim) {
        cout << d << " | ";
        double time_1 = 0.0;
        double time_2 = 0.0;

        for (int i=0; i<instances_per_dim; ++i) {
            MatrixXd C = MatrixXd::Random(d, d);
            MatrixXd A = C * C.transpose();
            VectorXd b = VectorXd::Random(d, 1);

            // run with custom loop
            time_1 += funcTime(xAx_symmetric, A, b, true);

            // run with Eigen's selfAdjointView
            time_2 += funcTime(xAx_symmetric, A, b, false);
        }

        time_1 /= double(instances_per_dim);
        time_2 /= double(instances_per_dim);
        std::cout << time_1 << " | " << time_2 << endl;
    }

    return 0;
}
