// $CXX -std=c++17 -I/${HOME}/opt prepare_input.cpp

#include<iostream>
#include<fstream>
#include<string>
#include<nlohmann/json.hpp>
#include <filesystem>
namespace fs = std::filesystem;

using namespace std;
using namespace std::string_literals;
using nlohmann::json;


int main()
{
    ifstream in("param.json");
    json j;
    in>>j;

    string folderBase="ip"+j["ip"]["type"].get<string>().substr(0,1)+
            "_U"+j["irlm"]["U"].dump()+
            "_dt"+j["tdvp"]["dt"].dump()+
            "_cdt"+j["circuit"]["dt"].dump()+
            "_cnImp"+j["circuit"]["nImp"].dump();
    fs::create_directory(folderBase);
    for(int L:{24,52,100,200}) {

        string folder=folderBase+"/L"+to_string(L);
        fs::create_directory(folder);
        j["irlm"]["L"]=L;
        ofstream out(folder+"/param.json");
        out<<setw(4)<<j<<endl;
        fs::current_path(folder);
        string cmd="sbatch -J "s+folder+" --output=out_"+folder+" ../../submit.sh";
        system(cmd.c_str());
        fs::current_path("../..");
    }
    
    return 0;
}
