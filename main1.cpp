#include "custom.h"
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>


int main(void) {
    cv::dnn::Net net = cv::dnn::readNetFromDarknet("yolov3.cfg", "yolov3.weights");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    std::vector<std::string> classes;
    std::ifstream ifs("coco.names");
    std::string line;
    while (getline(ifs, line)) classes.push_back(line);

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Cannot open camera." << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    float confThreshold = 0.5;
    float nmsThreshold = 0.4;
    int distanceThresholdArea = 70000;

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(416, 416), cv::Scalar(), true, false);
        net.setInput(blob);

        std::vector<cv::Mat> outputs;
        net.forward(outputs, Data_Structure::Get_Output_Layer_Names(net));

        std::vector< Data_Structure::Detection> detections;

        for (auto& output : outputs) {
            for (int i = 0; i < output.rows; ++i) {
                cv::Mat scores = output.row(i).colRange(5, output.cols);
                cv::Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

                if (confidence > confThreshold) {
                    int centerX = (int)(output.at<float>(i, 0) * frame.cols);
                    int centerY = (int)(output.at<float>(i, 1) * frame.rows);
                    int width = (int)(output.at<float>(i, 2) * frame.cols);
                    int height = (int)(output.at<float>(i, 3) * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;
                    int area = width * height;

                    if (area >= distanceThresholdArea) {
                        detections.emplace_back(classIdPoint.x, (float)confidence, cv::Rect(left, top, width, height));
                    }
                }
            }
        }

        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        for (auto& d : detections) {
            boxes.push_back(d.box);
            confidences.push_back(d.Confidence);
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

        Data_Structure::Hash_Map map;

        for (int idx : indices) {
            Data_Structure::Detection d = detections[idx];
            std::string label = cv::format("%s: %.2f", classes[d.ClassId].c_str(), d.Confidence);
            rectangle(frame, d.box, cv::Scalar(0, 255, 0), 2);
            putText(frame, label, cv::Point(d.box.x, d.box.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
            std::cout << "[NEAR DETECTION] " << label << std::endl;
            map.Insert(classes[d.ClassId]);
        }

        imshow("Detection System", frame);
        map.Print_All();

        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    
    return 0;
}
