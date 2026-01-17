#include "core/PerspectiveCorrector.hpp"
using namespace cv;

namespace core {

PerspectiveCorrector::PerspectiveCorrector(int outW, int outH)
    : outW_(outW), outH_(outH), finder_(outW, outH) {}

WarpResult PerspectiveCorrector::findAndWarp(const cv::Mat& bgr, bool wantDebug) const {
    WarpResult R;
    if (bgr.empty()) return R;

    CornerResult C = finder_.processFrame(bgr, wantDebug);
    if (wantDebug) R.debug = C.debug_bgr.empty() ? bgr.clone() : C.debug_bgr;

    if (!C.paper_ok) {
        R.ok = false;
        return R;
    }

    cv::Mat warpedGray = C.warped_gray.clone();
    cv::Mat denoised;
    cv::bilateralFilter(warpedGray, denoised, 9, 100, 100);

    cv::Ptr<CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(2.0);
    clahe->setTilesGridSize(cv::Size(8, 8));
    cv::Mat enhanced;
    clahe->apply(denoised, enhanced);

    cv::Mat blurred;
    cv::GaussianBlur(enhanced, blurred, cv::Size(5, 5), 1.0);
    cv::Mat sharpened;
    cv::addWeighted(enhanced, 1.2, blurred, -0.2, 0, sharpened);

    cv::Mat warpedBgr;
    cv::cvtColor(sharpened, warpedBgr, cv::COLOR_GRAY2BGR);

    R.warped = warpedBgr;
    R.corners = C.markers_orig;
    R.ok = !R.warped.empty();

    return R;
}

}
