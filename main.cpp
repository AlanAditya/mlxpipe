#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include "src/vision.h"

int main(int argc, char** argv) {
    std::cout << "Starting application..." << std::endl;
    // 1. Initialize MLX Vision
    std::string palm_model = "tools/models/palm_detection_full.safetensors";
    std::string hand_model = "tools/models/hand_landmarks.safetensors";
    
    if (!mlx_vision::initialize_mlx(palm_model, hand_model)) {
        std::cerr << "Failed to initialize MLX Vision" << std::endl;
        return -1;
    }
    
    // 2. Open Webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open webcam." << std::endl;
        return -1;
    }
    
    std::cout << "Webcam opened. Press 'q' to quit." << std::endl;
    
    cv::Mat frame;    
    auto start_time = std::chrono::high_resolution_clock::now();
    int frame_count = 0;
    float fps = 0.0f;
    float last_inference_time_ms = 0.0f;
    
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Blank frame grabbed." << std::endl;
            break;
        }
        // 3. Run Pipeline
        auto inf_start = std::chrono::high_resolution_clock::now();
        auto detections = mlx_vision::detect_hands(frame);
        auto inf_end = std::chrono::high_resolution_clock::now();
        last_inference_time_ms = std::chrono::duration<float, std::milli>(inf_end - inf_start).count();
        
        // 4. Draw Detections and Landmarks
        for (const auto& hand : detections) {
            const auto& det = hand.box;
            
            int x = det.x * frame.cols;
            int y = det.y * frame.rows;
            int width = det.width * frame.cols;
            int height = det.height * frame.rows;
            
            cv::Rect rect(x, y, width, height);
            cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);
            
            // Draw score
            char score_text[32];
            snprintf(score_text, sizeof(score_text), "Hand: %.2f", hand.score);
            cv::putText(frame, score_text, cv::Point(x, y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            
            // Draw Landmarks
            for (const auto& lm : hand.landmarks) {
                int lmx = lm.x * frame.cols;
                int lmy = lm.y * frame.rows;
                cv::circle(frame, cv::Point(lmx, lmy), 4, cv::Scalar(0, 0, 255), -1);
            }
            
            // Draw Connections
            std::vector<std::pair<int, int>> connections = {
                {0, 1}, {1, 2}, {2, 3}, {3, 4}, // Thumb
                {0, 5}, {5, 6}, {6, 7}, {7, 8}, // Index finger
                {5, 9}, {9, 10}, {10, 11}, {11, 12}, // Middle finger
                {9, 13}, {13, 14}, {14, 15}, {15, 16}, // Ring finger
                {13, 17}, {17, 18}, {18, 19}, {19, 20}, // Pinky
                {0, 17} // Palm base
            };
            
            for (const auto& conn : connections) {
                int p1x = hand.landmarks[conn.first].x * frame.cols;
                int p1y = hand.landmarks[conn.first].y * frame.rows;
                int p2x = hand.landmarks[conn.second].x * frame.cols;
                int p2y = hand.landmarks[conn.second].y * frame.rows;
                cv::line(frame, cv::Point(p1x, p1y), cv::Point(p2x, p2y), cv::Scalar(255, 0, 0), 2);
            }
        }
        
        frame_count++;
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> elapsed = current_time - start_time;
        if (elapsed.count() >= 1.0f) {
            fps = frame_count / elapsed.count();
            start_time = current_time;
            frame_count = 0;
        }
        
        char fps_text[64];
        snprintf(fps_text, sizeof(fps_text), "FPS: %.1f | Inference: %.1f ms", fps, last_inference_time_ms);
        cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
        
        // 5. Show the output
        cv::imshow("MediaPipe MLX Port", frame);
        
        // Exit on 'q'
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }
    
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
