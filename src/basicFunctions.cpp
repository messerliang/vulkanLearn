#include "basicFunctions.h"

#include <fstream>
#include <sstream>

std::string readFile(const char* filePath)
{
    std::ifstream inF(filePath, std::ios::in);
    if (!inF.is_open())
    {
        std::string msg = std::string("open ") + filePath + std::string(" fail\n");
        throw std::runtime_error(msg.c_str());
    }

    std::ostringstream oss;
    oss << inF.rdbuf();

    return oss.str();
}
