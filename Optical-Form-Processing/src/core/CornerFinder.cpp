#include "core/CornerFinder.hpp"
#include <algorithm>
#include <cmath>

using namespace cv;

namespace core {

std::vector<Point2f> CornerFinder::orderTLTRBRBL(const std::vector<Point2f>& pts, Point2f C) const {
    if (pts.size() != 4) {
        return pts;
    }
    
    std::vector<Point2f> sortedPts = pts;
    
    std::sort(sortedPts.begin(), sortedPts.end(), [](const Point2f& a, const Point2f& b) {
        return (a.x + a.y) < (b.x + b.y);
    });
    
    Point2f topLeft = sortedPts[0];
    Point2f bottomRight = sortedPts[3];
    
    std::vector<Point2f> remaining = {sortedPts[1], sortedPts[2]};
    std::sort(remaining.begin(), remaining.end(), [](const Point2f& a, const Point2f& b) {
        return (a.y - a.x) < (b.y - b.x);
    });
    
    Point2f topRight = remaining[0];
    Point2f bottomLeft = remaining[1];
    
    return {topLeft, topRight, bottomRight, bottomLeft};
}

bool CornerFinder::findCornerSquares(const Mat& gray, std::vector<Point2f>& corners, Mat* dbg) const {
    Mat th;
    
    GaussianBlur(gray, th, Size(5, 5), 0);
    adaptiveThreshold(th, th, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 15, 3);
    
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(th, th, MORPH_OPEN, kernel, Point(-1,-1), 1);
    morphologyEx(th, th, MORPH_CLOSE, kernel, Point(-1,-1), 1);
    
    if (dbg) cvtColor(gray, *dbg, COLOR_GRAY2BGR);
    
    std::vector<std::vector<Point>> contours;
    findContours(th, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    std::vector<std::vector<Point>> candidateContours;
    
    for (auto& c : contours) {
        double area = contourArea(c);
        if (area < 50 || area > 15000) continue;
        
        Rect br = boundingRect(c);
        float ar = (float)br.width / br.height;
        if (ar < 0.7f || ar > 1.4f) continue;
        
        double extent = area / br.area();
        if (extent < 0.65) continue;
        
        std::vector<Point> hull;
        convexHull(c, hull);
        double hullArea = contourArea(hull);
        double solidity = area / hullArea;
        if (solidity < 0.8) continue;
        
        candidateContours.push_back(c);
    }
    
    if (dbg) drawContours(*dbg, candidateContours, -1, Scalar(0,255,255), 2);
    
    if (candidateContours.size() < 4) {
        if (dbg) {
            std::string msg = "Yeterli aday yok: " + std::to_string(candidateContours.size()) + "/4";
            putText(*dbg, msg, {20,40}, FONT_HERSHEY_SIMPLEX, 0.8, {0,0,255}, 2);
        }
        return false;
    }
    
    std::sort(candidateContours.begin(), candidateContours.end(), [](auto& a, auto& b) {
        return contourArea(a) > contourArea(b);
    });
    
    std::vector<std::vector<Point>> finalFour(candidateContours.begin(), 
                                               candidateContours.begin() + std::min((size_t)4, candidateContours.size()));
    
    if (dbg) drawContours(*dbg, finalFour, -1, Scalar(0,255,0), 3);
    
    std::vector<Point2f> centers;
    for (auto& c : finalFour) {
        Moments m = moments(c);
        if (m.m00 != 0) {
            centers.push_back(Point2f(m.m10 / m.m00, m.m01 / m.m00));
        }
    }
    
    if (centers.size() != 4) {
        if (dbg) putText(*dbg, "Merkez hesaplama hatasi", {20,70}, FONT_HERSHEY_SIMPLEX, 0.8, {0,0,255}, 2);
        return false;
    }
    
    corners = orderTLTRBRBL(centers, Point2f(gray.cols / 2.f, gray.rows / 2.f));
    
    if (dbg) {
        std::vector<std::string> labels = {"TL", "TR", "BR", "BL"};
        std::vector<Scalar> colors = {
            Scalar(255,0,0),
            Scalar(0,255,0),
            Scalar(0,0,255),
            Scalar(255,255,0)
        };
        
        for (int i=0; i<4; ++i) {
            circle(*dbg, corners[i], 10, colors[i], -1);
            putText(*dbg, labels[i], corners[i] + Point2f(15, 0), 
                    FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2);
        }
        
        for (int i = 0; i < 4; ++i) {
            line(*dbg, corners[i], corners[(i+1)%4], Scalar(0,255,255), 2);
        }
    }
    
    return true;
}

CornerResult CornerFinder::processFrame(const Mat& bgr, bool debug_on) const {
    CornerResult R;
    if (bgr.empty()) return R;
    
    Mat gray;
    cvtColor(bgr, gray, COLOR_BGR2GRAY);
    
    std::vector<Point2f> srcPoints;
    Mat dbgImg;
    R.paper_ok = findCornerSquares(gray, srcPoints, debug_on ? &dbgImg : nullptr);
    
    if (debug_on) R.debug_bgr = dbgImg;
    
    if (!R.paper_ok) return R;
    
    std::vector<Point2f> dstPoints = {
        {0, 0},
        {(float)outW_ - 1, 0},
        {(float)outW_ - 1, (float)outH_ - 1},
        {0, (float)outH_ - 1}
    };
    
    Mat H = getPerspectiveTransform(srcPoints, dstPoints);
    
    warpPerspective(gray, R.warped_gray, H, 
                    Size(outW_, outH_), 
                    INTER_LINEAR,
                    BORDER_REPLICATE);
    
    for(int i=0; i<4; ++i) R.markers_orig[i] = srcPoints[i];
    
    return R;
}

}
