#include <glob.h>
#include <vector>
#include <string>
#include <iostream>
#include <tuple>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <random>
#include <iterator>


using namespace std;

const int NUM_FOLDERS = 1680;
const string DATA_DIR = "./lfw/lfw/";

template <typename Iter, typename RandomGenerator>
Iter select_randomly(Iter start, Iter end, RandomGenerator &g)
{
    std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
    std::advance(start, dis(g));
    return start;
}

template <typename Iter>
Iter select_randomly(Iter start, Iter end)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return select_randomly(start, end, gen);
}

vector<string> globVector(const string &pattern)
{
    glob_t glob_result;
    glob(pattern.c_str(), 0, NULL, &glob_result);
    vector<string> files;
    for (unsigned int i = 0; i < glob_result.gl_pathc; ++i)
    {
        files.push_back(string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    return files;
}

vector<string> files = globVector("./lfw/lfw/*");

string get_triplet_data()
{

    string anchor = *select_randomly(files.begin(), files.end());
    string negative = *select_randomly(files.begin(), files.end());
    vector<string> anchor_files = globVector(anchor + "/*");
    vector<string> negative_files = globVector(negative + "/*");

    string anchor_file = *select_randomly(anchor_files.begin(), anchor_files.end());
    string positive_file = *select_randomly(anchor_files.begin(), anchor_files.end());
    string negative_file = *select_randomly(negative_files.begin(), negative_files.end());
    while (anchor_file == positive_file){
        positive_file = *select_randomly(anchor_files.begin(), anchor_files.end());
    }
    string ret_string = anchor_file + " " + positive_file + " " + negative_file;
    return ret_string;
}


int main(){
    srand(time(NULL));
    vector<string> files = globVector("./lfw/lfw/*");
    for (int i=0; i < 10; i++){
        string anchor, positive, negative;
        cout << get_triplet_data() << endl;
    }
    return 0;
}