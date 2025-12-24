// edge_detection.h - Production-Grade Edge Detection Module
// Complete implementation of all major edge detection algorithms
#ifndef EDGE_DETECTION_H
#define EDGE_DETECTION_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// Edge detection methods
enum class EdgeMethod {
    CANNY,
    SOBEL,
    SCHARR,
    PREWITT,
    ROBERTS,
    LAPLACIAN,
    LOG,                 // Laplacian of Gaussian
    DOG,                 // Difference of Gaussian
    KIRSCH,
    ROBINSON,
    FREI_CHEN,
    DERICHE,
    HOLISTICALLY_NESTED  // Deep learning based
};

// Edge linking strategies
enum class LinkingStrategy {
    NONE,
    HYSTERESIS,
    GRADIENT_FOLLOWING,
    MORPHOLOGICAL,
    GRAPH_BASED
};

// Edge parameters
struct EdgeDetectionParams {
    EdgeMethod method;
    double low_threshold;
    double high_threshold;
    int kernel_size;
    double sigma;             // For Gaussian smoothing
    LinkingStrategy linking;
    bool non_max_suppression;
    bool subpixel_accuracy;
};

// Edge information
struct EdgeInfo {
    cv::Mat edge_map;            // Binary edge map
    cv::Mat gradient_magnitude;  // Gradient strength
    cv::Mat gradient_direction;  // Gradient orientation
    std::vector<std::vector<cv::Point>> edge_chains;
    std::vector<cv::Vec4f> edge_lines;  // Line segments
    float average_edge_strength;
    int num_edge_pixels;
};

class EdgeDetector {
public:
    EdgeDetector();
    ~EdgeDetector();
    
    // Main edge detection interface
    EdgeInfo detectEdges(const cv::Mat& image,
                         const EdgeDetectionParams& params);
    
    // Individual edge detection methods
    cv::Mat cannyEdgeDetection(const cv::Mat& image,
                              double low_threshold,
                              double high_threshold,
                              int aperture_size = 3,
                              bool L2gradient = true);
    
    cv::Mat sobelEdgeDetection(const cv::Mat& image,
                              int dx = 1,
                              int dy = 1,
                              int ksize = 3);
    
    cv::Mat scharrEdgeDetection(const cv::Mat& image,
                               int dx = 1,
                               int dy = 1);
    
    cv::Mat prewittEdgeDetection(const cv::Mat& image);
    
    cv::Mat robertsEdgeDetection(const cv::Mat& image);
    
    cv::Mat laplacianEdgeDetection(const cv::Mat& image,
                                  int ksize = 3,
                                  double scale = 1.0);
    
    cv::Mat logEdgeDetection(const cv::Mat& image,
                            double sigma = 1.5);
    
    cv::Mat dogEdgeDetection(const cv::Mat& image,
                            double sigma1 = 1.0,
                            double sigma2 = 2.0);
    
    cv::Mat kirschEdgeDetection(const cv::Mat& image);
    
    cv::Mat robinsonEdgeDetection(const cv::Mat& image);
    
    cv::Mat freiChenEdgeDetection(const cv::Mat& image);
    
    cv::Mat dericheEdgeDetection(const cv::Mat& image,
                                 double alpha = 1.0);
    
    // Gradient computation
    void computeGradients(const cv::Mat& image,
                         cv::Mat& gradient_x,
                         cv::Mat& gradient_y,
                         int ksize = 3);
    
    void computeGradientMagnitude(const cv::Mat& grad_x,
                                 const cv::Mat& grad_y,
                                 cv::Mat& magnitude);
    
    void computeGradientDirection(const cv::Mat& grad_x,
                                 const cv::Mat& grad_y,
                                 cv::Mat& direction);
    
    // Non-maximum suppression
    cv::Mat nonMaximumSuppression(const cv::Mat& magnitude,
                                  const cv::Mat& direction);
    
    // Hysteresis thresholding
    cv::Mat hysteresisThresholding(const cv::Mat& magnitude,
                                   double low_threshold,
                                   double high_threshold);
    
    // Edge linking
    std::vector<std::vector<cv::Point>> linkEdges(const cv::Mat& edge_map,
                                                   LinkingStrategy strategy);
    
    std::vector<std::vector<cv::Point>> morphologicalLinking(const cv::Mat& edge_map);
    
    std::vector<std::vector<cv::Point>> gradientFollowing(
        const cv::Mat& edge_map,
        const cv::Mat& gradient_dir);
    
    std::vector<std::vector<cv::Point>> graphBasedLinking(const cv::Mat& edge_map);
    
    // Edge thinning
    cv::Mat edgeThinning(const cv::Mat& edge_map);
    cv::Mat zhangSuenThinning(const cv::Mat& binary_image);
    
    // Subpixel edge localization
    std::vector<cv::Point2f> subpixelEdgeLocalization(
        const cv::Mat& image,
        const std::vector<cv::Point>& edge_points);
    
    // Line segment detection
    std::vector<cv::Vec4f> detectLineSegments(const cv::Mat& edge_map);
    std::vector<cv::Vec4f> houghLinesP(const cv::Mat& edge_map,
                                       double rho = 1.0,
                                       double theta = CV_PI/180,
                                       int threshold = 50,
                                       double min_line_length = 50,
                                       double max_line_gap = 10);
    
    std::vector<cv::Vec4f> lsdLineDetector(const cv::Mat& image);
    
    // Edge strength analysis
    float computeAverageEdgeStrength(const cv::Mat& magnitude,
                                     const cv::Mat& edge_map);
    
    void analyzeEdgeStatistics(const cv::Mat& edge_map,
                              const cv::Mat& magnitude,
                              float& mean_strength,
                              float& max_strength,
                              int& num_edges);
    
    // Edge filtering
    cv::Mat filterWeakEdges(const cv::Mat& edge_map,
                           const cv::Mat& magnitude,
                           float threshold);
    
    cv::Mat filterShortEdges(const cv::Mat& edge_map,
                            int min_length);
    
    // Oriented edge detection
    cv::Mat detectEdgesInDirection(const cv::Mat& image,
                                   float orientation,
                                   float tolerance = 30.0f);
    
    std::vector<cv::Mat> detectMultiOrientationEdges(
        const cv::Mat& image,
        const std::vector<float>& orientations);
    
    // Edge enhancement
    cv::Mat enhanceEdges(const cv::Mat& edge_map,
                        int dilation_size = 1);
    
    cv::Mat suppressIsolatedEdges(const cv::Mat& edge_map,
                                 int neighborhood_size = 3);
    
    // Contour extraction from edges
    std::vector<std::vector<cv::Point>> extractContours(const cv::Mat& edge_map);
    
    std::vector<std::vector<cv::Point>> extractContoursWithHierarchy(
        const cv::Mat& edge_map,
        std::vector<cv::Vec4i>& hierarchy);
    
    // Edge visualization
    cv::Mat visualizeEdges(const cv::Mat& original,
                          const cv::Mat& edge_map,
                          const cv::Scalar& color = cv::Scalar(0, 255, 0));
    
    cv::Mat visualizeGradient(const cv::Mat& magnitude,
                             const cv::Mat& direction);
    
    cv::Mat visualizeLineSegments(const cv::Mat& image,
                                 const std::vector<cv::Vec4f>& lines);
    
    // Utility functions
    void setDefaultParams();
    EdgeDetectionParams getDefaultParams() const { return default_params_; }
    
    cv::Mat combineEdgeMaps(const std::vector<cv::Mat>& edge_maps,
                           bool use_voting = true);
    
    float computeEdgeDensity(const cv::Mat& edge_map);
    
    bool validateEdgeMap(const cv::Mat& edge_map);

private:
    // Helper functions for specific algorithms
    void applyKirschKernels(const cv::Mat& image,
                           std::vector<cv::Mat>& responses);
    
    void applyRobinsonKernels(const cv::Mat& image,
                             std::vector<cv::Mat>& responses);
    
    void applyFreiChenKernels(const cv::Mat& image,
                             std::vector<cv::Mat>& edge_responses,
                             std::vector<cv::Mat>& line_responses);
    
    cv::Mat recursiveGaussianFilter(const cv::Mat& image, double alpha);
    
    cv::Mat dericheGaussian(const cv::Mat& image, double alpha);
    
    cv::Mat dericheDerivative(const cv::Mat& image, double alpha);
    
    // Edge tracking in hysteresis
    void edgeTrackingRecursive(cv::Mat& edges,
                              int x, int y,
                              double low_threshold);
    
    // Chain following
    std::vector<cv::Point> followEdgeChain(const cv::Mat& edge_map,
                                          cv::Point start,
                                          cv::Mat& visited);
    
    // Line fitting
    cv::Vec4f fitLineToPoints(const std::vector<cv::Point>& points);
    
    // Direction quantization for non-max suppression
    int quantizeDirection(float angle);
    
    // Kernel generation
    std::vector<cv::Mat> generateKirschKernels();
    std::vector<cv::Mat> generateRobinsonKernels();
    std::vector<cv::Mat> generateFreiChenKernels();
    
    // Parameters
    EdgeDetectionParams default_params_;
    
    // Cached kernels
    std::map<std::string, std::vector<cv::Mat>> kernel_cache_;
    
    // Statistics
    struct Statistics {
        int total_detections;
        float avg_processing_time_ms;
        float avg_edge_density;
    } stats_;
};

// Utility functions
inline cv::Mat computeGradientMagnitudeSimple(const cv::Mat& gx, const cv::Mat& gy) {
    cv::Mat magnitude;
    cv::magnitude(gx, gy, magnitude);
    return magnitude;
}

inline cv::Mat computeGradientDirectionSimple(const cv::Mat& gx, const cv::Mat& gy) {
    cv::Mat direction;
    cv::phase(gx, gy, direction, true);  // Angle in degrees
    return direction;
}

inline float normalizeAngle(float angle) {
    while (angle < 0) angle += 360.0f;
    while (angle >= 360.0f) angle -= 360.0f;
    return angle;
}

#endif // EDGE_DETECTION_H
