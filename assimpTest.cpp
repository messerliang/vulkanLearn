#if 1

#include <string>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <iostream>


const std::string vampire_file = "vulkanTesting/asset/dancing_vampire.dae";

int main()
{
    
    std::cout << "begin...\n";
    Assimp::Importer importer;
    
    const aiScene* scene = importer.ReadFile(vampire_file.c_str(), aiProcess_Triangulate|aiProcess_FlipUVs | aiProcess_CalcTangentSpace);
    
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cerr << "Assimp Error: " << importer.GetErrorString() << std::endl;
        return -1;
    }

    std::cout << "Model has " << scene->mNumMeshes << " meshes." << std::endl;

    
}

#endif
