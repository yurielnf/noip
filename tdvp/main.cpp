#include "irlm.h"
#include "it_tdvp.h"

#include<iostream>

using namespace std;

int main()
{
    IRLM model {.L=10, .t=0.5, .V=0.15, .U=-0.5};
    HamSys sys=model.Ham();

    cout<<"bond dimensions of H:\n";
    for(int i=1; i<sys.sites.length(); i++)
        cout<<rightLinkIndex(sys.ham,i).dim()<<" ";
    cout<<endl;


    return 0;
}
