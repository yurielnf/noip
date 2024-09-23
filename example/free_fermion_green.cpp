#include <iostream>
#include <iomanip>
#include <fstream>
#include <armadillo>

#include <irlm.h>


using namespace std;
using namespace arma;


int main(int argc, char **argv)
{
    cout<<"Hola mundo "<<setprecision(15)<<endl;

    int len=20;
    double dt=0.1;
    bool is_greater=true;
    if (argc>=2) len=atoi(argv[1]);
    if (argc>=3) dt=atof(argv[2]);
    if (argc>=4) is_greater=atoi(argv[3]);

    auto model1=IRLM {.L=len, .t=0.5, .V=0.1, .U=0.0, .ed=0.0};

    arma::mat H1=model1.matrices().first;

    arma::vec eval1;
    arma::mat evec1;

    eig_sym(eval1,evec1,H1);

    auto green=[&](int i, int j, double t, bool is_greater) {
        cx_mat phase=diagmat(arma::exp(eval1*cx_double(0,-t)));
        double Ef=arma::sum(eval1(arma::find(eval1<0)));
        rowvec jket=evec1.row(j);
        cx_rowvec iket=evec1.row(i)*phase;
        cx_double sum=0;
        for(auto a=0; a<eval1.size(); a++)
           if ((is_greater && eval1[a]>=0) ||
               (!is_greater && eval1[a]<0)) sum += std::conj(iket[a])*jket[a];
        return sum;
    };

    ofstream out("free_green_L"s+to_string(len)+"_g"+to_string(is_greater)+".txt");
    out<<"time M m energy n0\n"<<setprecision(14);
    for(auto i=0; i*dt<=len; i++) {
        double t=dt*i;
        cx_double g=green(0,0,t,is_greater);
        out<<t<<" 0 0 0 "<< g.real() <<" "<< g.imag()<< " 0 0"<<endl;
    }

    return 0;
}
