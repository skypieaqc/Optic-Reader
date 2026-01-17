#pragma once
#include <opencv2/opencv.hpp>
#include <array>
#include "core/CornerFinder.hpp"

namespace core {

struct WarpResult {
    bool ok = false;
    cv::Mat warped;             
    cv::Mat debug;              
    std::array<cv::Point2f,4> corners{}; 
};

class PerspectiveCorrector {
public:
    PerspectiveCorrector(int outW, int outH);
    
    WarpResult findAndWarp(const cv::Mat& bgr, bool wantDebug) const;

private:
    int outW_, outH_;
    CornerFinder finder_;
};

}