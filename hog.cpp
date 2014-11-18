#include "HOG.hpp"
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cgi/logger/logger.hpp>
#include <cgi/core/exceptions/exceptions.hpp>
#include <cgi/core/process_utilities/Timing.hpp>

#include <boost/math/special_functions/fpclassify.hpp>

#define GJS_HARDCODE_HOG_BUCKETS 16
#define RAD2DEG(x) (180 *(x) / M_PI)
#define DEG2RAD(x) (M_PI *(x) / 180)

namespace scottgs
{

/**
 * Creates a HOG from an image and a binary mask with respect to magnitude and orientation
 * @param the original image
 * @param the binary mask for the image (1 = part of blob, 0 = not part of blob->don't consider it)
 */
std::vector<double> generateCellWeightedHOG(const cv::Mat& cell, const cv::Mat& mask)
{
    cgi::log::MetaLogStream& log(cgi::log::MetaLogStream::instance());

    // make sure the cell and mask are the same size
    if (cell.size() != mask.size())
      CGI_THROW(std::runtime_error, "The original image and mask sizes do not match up"); 

    const int hogBuckets = GJS_HARDCODE_HOG_BUCKETS;
    std::vector<double> features(hogBuckets, 0.0);
    
    double sumWeights = cv::sum(mask)[0];
    if (sumWeights == 0.0) // early bail
      return features;

    // ====================================================
    // Generate Weighted-HOG
    // Bins = 16
    // ====================================================
    // Compute the derivatives in the X and Y directions
    cv::Mat grad_x, grad_y;

    cv::Sobel(cell, grad_x, CV_64F, 1, 0, 3); 
    cv::Sobel(cell, grad_y, CV_64F, 0, 1, 3);
    
    // Build bucketting information
    const double bucketSize = 360.0 / hogBuckets;                     // with 16 bins, bucketSize = 22.5
    const double halfBucket = bucketSize / 2.0;                           // with 16 bins, halfBucket = 11.25
    
    std::vector<double> bucketCenters;
    bucketCenters.reserve(hogBuckets);                                       // reserve 16 spots in our vector
    for (unsigned int b = 0; b < hogBuckets; ++b) {
      bucketCenters.push_back((b * bucketSize) + halfBucket);
    }
    
    log << cgi::log::Priority::DEBUG << "scottgs::generateCellWeightedHOG" 
      << "Buckets=" << hogBuckets << ",BucketSize=" << bucketSize << cgi::log::flush;
    
    // Compute the orientation and magnitude of each gradient 
    for (int y = 0; y < dx.rows; ++y) 
      for (int x = 0; x < grad_x.cols; ++x) {
      // if it's a 1 (since binary mask) -- apply the operations
      if (mask.at<int>(x,y) > 0) {
        // these Sobel derivatives will always be a number in our case, but not in the general case
        const double dxVal = grad_x.at<double>(x,y);
        const double dyVal = grad_y.at<double>(x,y);

        // dxVal and dyVal come together to make a right angle triangle--apply Pythaogras Theorem and you get magnitude
        const double magnitude = sqrt(dxVal * dxVal + dyVal * dyVal); 
        
        // get the orientation of the gradient
        // tan theta = dyVal / dxVal   => theta = arctan(dyVal / dxVal) 
        // Arctan domain = [-180,180], + 180 to put orientation domain into [0,360]
        const double orientation = RAD2DEG(atan2(dyVal , dxVal)) + 180;    

        // Bin the histograms
        // 1) Compute the primary bucket
        // 2) Compute the secondary bucket to give some weight towards that unless perfect hit
        int primaryBucket = static_cast<int>(floor(orientation / bucketSize)); // with 16 bins, bucketSize = 22.5 degrees

        // error handling if orientation = 360 degrees exactly because vectors are 0-based
        // i.e. our buckets are numbered 0...hogBuckets - 1
        if (primaryBucket >= hogBuckets) 
          primaryBucket = hogBuckets - 1;
        
        // distance from primary bucket center
        double distPrimaryBucketCenter = orientation - bucketCenters.at(primaryBucket);
        
        // this will be [-1, 1]
        double alpha = fabs(distPrimaryBucketCenter / bucketSize); 
        
        // init secondary to the primary and then make comparisons to change the secondary if needed
        int secondaryBucket = primaryBucket; // valid if dead center in the bucket
        if (distPrimaryBucketCenter > 0) // need to look at the next bucket
          // use % to wrap from last bucket to first
          secondaryBucket = (primaryBucket + 1) % hogBuckets;
        else if (distPrimaryBucketCenter < 0) // need to look at the previous bucket
          // similarly, need to use % to wrap from first bucket to the last
          secondaryBucket = (primaryBucket - 1) % hogBuckets;
        // otherwise, we are dead on the bucket center and hence secondaryBucket = primaryBucket and alpha = 0

        // add a portion of this to the primary bucket and the rest to the secondary
        const double primaryAdded = magnitude * (1 - alpha);
        const double secondaryAdded = magnitude * alpha;
        features.at(primaryBucket) += primaryAdded;
        features.at(secondaryBucket) += secondaryAdded;
      }
    }
    
    return features;
  }
}
