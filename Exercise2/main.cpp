#include <iostream>
#include "Eigen/Eigen"


using namespace std;
using namespace Eigen;

// MatrixXd represents a matrix of doubles with dynamic size
// The same for VectorXd

bool SolveSystems(const Matrix2d& A,
                  double& detA,
                  double& condA,
                  const Vector2d& b,
                  double& errRel_PA,
                  double& errRel_QR)
{
    JacobiSVD<Matrix2d> svd(A);
    Vector2d singularValuesA = svd.singularValues();
    condA = singularValuesA.maxCoeff()/singularValuesA.minCoeff();
    detA = A.determinant();
    if (singularValuesA.minCoeff() < 1e-16)
    {
        errRel_PA = -1;
        errRel_QR = -1;
        return false;
    }
    Vector2d exactSolution(2);
    exactSolution << -1.0e+0, -1e+0;

    Vector2d x_PA = A.fullPivLu().solve(b);
    Vector2d x_QR = A.colPivHouseholderQr().solve(b);

    errRel_PA = (exactSolution - x_PA).norm()/exactSolution.norm();
    errRel_QR = (exactSolution - x_QR).norm()/exactSolution.norm();
    return true;
}



int main()
{
    Matrix2d A1, A2, A3;
    Vector2d b1, b2, b3;

    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
        8.320502943378437e-01, -9.992887623566787e-01;
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;

    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01, -8.324762492991313e-01;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;

    A3 << 5.547001962252291e-01, -5.547001955851905e-01,
        8.320502943378437e-01, -8.320502947645361e-01;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;

    // first case
    double detA1, condA1, errRel1_PA, errRel1_QR;
    if (SolveSystems(A1, detA1, condA1, b1, errRel1_PA, errRel1_QR)) {
        cout << scientific << "Sistema 1 (PALU). Errore relativo: " << errRel1_PA << endl;
        cout << scientific << "Sistema 1 (QR). Errore relativo: " << errRel1_QR << endl;
        cout << "-------------------------------------" << endl;
    }
    else {
        cout << "La matrice è singolare";
    }


    // second case
    double detA2, condA2, errRel2_PA, errRel2_QR;
    if (SolveSystems(A2, detA2, condA2, b2, errRel2_PA, errRel2_QR)) {
        cout << scientific << "Sistema 2 (PALU). Errore relativo: " << errRel2_PA << endl;
        cout << scientific << "Sistema 2 (QR). Errore relativo: " << errRel2_QR << endl;
        cout << "-------------------------------------" << endl;
    }
    else {
        cout << "La matrice è singolare";
    }


    // third case
    double detA3, condA3, errRel3_PA, errRel3_QR;
    if (SolveSystems(A3, detA3, condA3, b3, errRel3_PA, errRel3_QR)) {
        cout << scientific << "Sistema 3 (PALU). Errore relativo: " << errRel3_PA << endl;
        cout << scientific << "Sistema 3 (QR). Errore relativo: " << errRel3_QR << endl;
        cout << "-------------------------------------" << endl;
    }
    else {
        cout << "La matrice è singolare";
    }



    return 0;

}



