#include "custom.h"
#include <iostream>

namespace Data_Structure {

    Detection::Detection(int id, float conf, cv::Rect b)
        : ClassId(id), Confidence(conf), box(b) {
    }

    Hash_Node::Hash_Node(std::string k, int v) : key(k), value(v), next(nullptr) {}

    Hash_Map::Hash_Map() {
        for (int i = 0; i < SIZE; ++i) table[i] = nullptr;
    }

    int Hash_Map::Hash_Function(std::string key) {
        int hash = 0;
        for (char ch : key)
            hash = (hash * 31 + ch) % SIZE;
        return hash;
    }

    void Hash_Map::Insert(std::string key) {
        int index = Hash_Function(key);
        Hash_Node* node = table[index];
        while (node != nullptr) {
            if (node->key == key) {
                node->value++;
                return;
            }
            node = node->next;
        }
        Hash_Node* newNode = new Hash_Node(key, 1);
        newNode->next = table[index];
        table[index] = newNode;
    }

    int Hash_Map::Get(std::string key) {
        int index = Hash_Function(key);
        Hash_Node* node = table[index];
        while (node) {
            if (node->key == key)
                return node->value;
            node = node->next;
        }
        return 0;
    }

    void Hash_Map::Print_All() {
        for (int i = 0; i < SIZE; ++i) {
            Hash_Node* node = table[i];
            while (node) {
                std::cout << node->key << ": " << node->value << "\n";
                node = node->next;
            }
        }
    }

    Hash_Map::~Hash_Map() {
        for (int i = 0; i < SIZE; ++i) {
            Hash_Node* node = table[i];
            while (node) {
                Hash_Node* temp = node;
                node = node->next;
                delete temp;
            }
        }
    }

    std::vector<std::string> Get_Output_Layer_Names(const cv::dnn::Net& net) {
        static std::vector<std::string> names;
        if (names.empty()) {
            std::vector<int> outLayers = net.getUnconnectedOutLayers();
            std::vector<std::string> layersNames = net.getLayerNames();
            names.resize(outLayers.size());
            for (size_t i = 0; i < outLayers.size(); ++i)
                names[i] = layersNames[outLayers[i] - 1];
        }
        return names;
    }
}
