#ifndef CUSTOM_H
#define CUSTOM_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace Data_Structure {

    // Struct for detected objects
    class Detection {
    public:
        int ClassId;
        float Confidence;
        cv::Rect box;
        Detection(int id, float conf, cv::Rect b);
    };

    // Manually implemented simple hash map (chained hashing)
    class Hash_Node {
    public:
        std::string key;
        int value;
        Hash_Node* next;
        Hash_Node(std::string k, int v);
    };

    class Hash_Map {
    private:
        static const int SIZE = 100;
        Hash_Node* table[SIZE];
        int Hash_Function(std::string key);
    public:
        Hash_Map();
        void Insert(std::string key);
        int Get(std::string key);
        void Print_All();
        ~Hash_Map();
    };

    // Helper function
    std::vector<std::string> Get_Output_Layer_Names(const cv::dnn::Net& net);

}

#endif
