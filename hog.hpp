#ifndef _SCOTTGS_HOG_HPP
#define _SCOTTGS_HOG_HPP

#include <vector>
#include <opencv2/core/core.hpp>

namespace scottgs {

	std::vector<double> generateCellWeightedHOG(const cv::Mat& cell, const cv::Mat& weightCell);

}; // END scottgs namespace

#endif
