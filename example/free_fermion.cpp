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
    if (argc>=2) len=atoi(argv[1]);
    if (argc>=3) dt=atof(argv[2]);

    auto model0=IRLM {.L=len, .t=0.5, .V=0.0, .U=-1.0, .ed=-10, .connected=false};
    auto model1=IRLM {.L=len, .t=0.5, .V=0.1, .U=0.0, .ed=0.0};

    arma::mat H0=model0.matrices().first;
    arma::mat H1=model1.matrices().first;

    arma::vec eval0, eval1;
    arma::mat evec0, evec1;

    eig_sym(eval0,evec0,H0);
    eig_sym(eval1,evec1,H1);

    cout<<"number of particles: "<<arma::find(eval0<0).eval().size()<<endl;

    auto corr=[&](int i, int j, double t) {
        cx_mat phase=diagmat(arma::exp(eval1*cx_double(0,-t)));
        cx_rowvec jket=evec1.row(j)*phase*evec1.t()*evec0;
        cx_rowvec iket=evec1.row(i)*phase*evec1.t()*evec0;
        cx_double sum=0;
        for(auto a=0; a<eval0.size(); a++)
           if (eval0[a]<0) sum += std::conj(iket[a])*jket[a];
        return sum;
    };

    auto corr_all=[&](double t) {
        cx_mat phase=diagmat(arma::exp(eval1*cx_double(0,-t)));
        cx_mat jket=evec1*phase*evec1.t()*evec0;
        cx_mat iket=evec1*phase*evec1.t()*evec0;
        cx_mat sum(arma::size(evec0), fill::zeros);
        for(auto a=0; a<eval0.size(); a++)
            // if (eval0[a]<0) sum += std::conj(iket[a])*jket[a];
            if (eval0[a]<0) sum += arma::conj(iket.col(a))*jket.col(a).st();
        return sum;
    };

    ofstream out("free_L"s+to_string(len)+".txt");
    out<<"time M m energy n0\n"<<setprecision(14);
    out<<"0 0 0 0 "<< corr(0,0,0).real() <<" "<< (corr(0,1,0)+corr(1,0,0)).real() <<endl;

    for(auto i=0; i*dt<=len; i++) {
        double t=dt*(i+1);
        out<<t<<" 0 0 0 "<< corr(0,0,t).real() <<" "<< (corr(0,1,t)+corr(1,0,t)).real() <<endl;
    }

    // for(int ti:{2000,6000,10000}) {
    //     arma::cx_mat cc=corr_all((ti+1)*dt);
    //     arma::real(cc.diag()).eval().save("ni_L"s+to_string(len)+"_t"+to_string(ti)+".txt",arma::raw_ascii);
    //     arma::real(arma::square(arma::abs(cc.col(0)))).eval().save("ci0_abs2_L"s+to_string(len)+"_t"+to_string(ti)+".txt",arma::raw_ascii);
    // }

    return 0;
}
