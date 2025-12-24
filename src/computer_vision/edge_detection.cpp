// edge_detection.cpp - COMPLETE PRODUCTION IMPLEMENTATION
// Full industrial-grade edge detection for biometric systems
#include "edge_detection.h"
#include "../utils/logger.h"
#include <cmath>
#include <algorithm>
#include <queue>
#include <set>

EdgeDetector::EdgeDetector() {
    setDefaultParams();
    stats_ = {0, 0.0f, 0.0f};
}

EdgeDetector::~EdgeDetector() {
    kernel_cache_.clear();
}

void EdgeDetector::setDefaultParams() {
    default_params_.method = EdgeMethod::CANNY;
    default_params_.low_threshold = 50.0;
    default_params_.high_threshold = 150.0;
    default_params_.kernel_size = 3;
    default_params_.sigma = 1.4;
    default_params_.linking = LinkingStrategy::HYSTERESIS;
    default_params_.non_max_suppression = true;
    default_params_.subpixel_accuracy = false;
}

EdgeInfo EdgeDetector::detectEdges(const cv::Mat& image,
                                   const EdgeDetectionParams& params) {
    auto start = std::chrono::steady_clock::now();
    
    EdgeInfo info;
    
    // Convert to grayscale if needed
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // Select edge detection method
    switch (params.method) {
        case EdgeMethod::CANNY:
            info.edge_map = cannyEdgeDetection(gray, params.low_threshold,
                                               params.high_threshold, params.kernel_size);
            break;
        
        case EdgeMethod::SOBEL:
            info.edge_map = sobelEdgeDetection(gray, 1, 1, params.kernel_size);
            break;
        
        case EdgeMethod::SCHARR:
            info.edge_map = scharrEdgeDetection(gray, 1, 1);
            break;
        
        case EdgeMethod::PREWITT:
            info.edge_map = prewittEdgeDetection(gray);
            break;
        
        case EdgeMethod::ROBERTS:
            info.edge_map = robertsEdgeDetection(gray);
            break;
        
        case EdgeMethod::LAPLACIAN:
            info.edge_map = laplacianEdgeDetection(gray, params.kernel_size);
            break;
        
        case EdgeMethod::LOG:
            info.edge_map = logEdgeDetection(gray, params.sigma);
            break;
        
        case EdgeMethod::DOG:
            info.edge_map = dogEdgeDetection(gray, params.sigma, params.sigma * 1.6);
            break;
        
        case EdgeMethod::KIRSCH:
            info.edge_map = kirschEdgeDetection(gray);
            break;
        
        case EdgeMethod::ROBINSON:
            info.edge_map = robinsonEdgeDetection(gray);
            break;
        
        case EdgeMethod::FREI_CHEN:
            info.edge_map = freiChenEdgeDetection(gray);
            break;
        
        case EdgeMethod::DERICHE:
            info.edge_map = dericheEdgeDetection(gray, 1.0);
            break;
        
        default:
            info.edge_map = cannyEdgeDetection(gray, params.low_threshold,
                                               params.high_threshold);
    }
    
    // Compute gradients
    cv::Mat grad_x, grad_y;
    computeGradients(gray, grad_x, grad_y, params.kernel_size);
    computeGradientMagnitude(grad_x, grad_y, info.gradient_magnitude);
    computeGradientDirection(grad_x, grad_y, info.gradient_direction);
    
    // Edge linking if requested
    if (params.linking != LinkingStrategy::NONE) {
        info.edge_chains = linkEdges(info.edge_map, params.linking);
    }
    
    // Detect line segments
    info.edge_lines = detectLineSegments(info.edge_map);
    
    // Compute statistics
    analyzeEdgeStatistics(info.edge_map, info.gradient_magnitude,
                         info.average_edge_strength,
                         info.average_edge_strength,
                         info.num_edge_pixels);
    
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    stats_.total_detections++;
    stats_.avg_processing_time_ms = 
        (stats_.avg_processing_time_ms * (stats_.total_detections - 1) + duration) / 
        stats_.total_detections;
    
    return info;
}

cv::Mat EdgeDetector::cannyEdgeDetection(const cv::Mat& image,
                                         double low_threshold,
                                         double high_threshold,
                                         int aperture_size,
                                         bool L2gradient) {
    cv::Mat edges;
    cv::Canny(image, edges, low_threshold, high_threshold, aperture_size, L2gradient);
    return edges;
}

cv::Mat EdgeDetector::sobelEdgeDetection(const cv::Mat& image,
                                         int dx, int dy, int ksize) {
    cv::Mat grad_x, grad_y;
    cv::Sobel(image, grad_x, CV_16S, dx, 0, ksize);
    cv::Sobel(image, grad_y, CV_16S, 0, dy, ksize);
    
    // Convert to absolute values
    cv::Mat abs_grad_x, abs_grad_y;
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);
    
    // Combine gradients
    cv::Mat gradient;
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, gradient);
    
    // Threshold
    cv::Mat edges;
    cv::threshold(gradient, edges, 50, 255, cv::THRESH_BINARY);
    
    return edges;
}

cv::Mat EdgeDetector::scharrEdgeDetection(const cv::Mat& image, int dx, int dy) {
    cv::Mat grad_x, grad_y;
    cv::Scharr(image, grad_x, CV_16S, dx, 0);
    cv::Scharr(image, grad_y, CV_16S, 0, dy);
    
    cv::Mat abs_grad_x, abs_grad_y;
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);
    
    cv::Mat gradient;
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, gradient);
    
    cv::Mat edges;
    cv::threshold(gradient, edges, 50, 255, cv::THRESH_BINARY);
    
    return edges;
}

cv::Mat EdgeDetector::prewittEdgeDetection(const cv::Mat& image) {
    // Prewitt kernels
    cv::Mat kernel_x = (cv::Mat_<float>(3, 3) <<
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 1
    );
    
    cv::Mat kernel_y = (cv::Mat_<float>(3, 3) <<
        -1, -1, -1,
         0,  0,  0,
         1,  1,  1
    );
    
    cv::Mat grad_x, grad_y;
    cv::filter2D(image, grad_x, CV_32F, kernel_x);
    cv::filter2D(image, grad_y, CV_32F, kernel_y);
    
    // Compute magnitude
    cv::Mat magnitude;
    cv::magnitude(grad_x, grad_y, magnitude);
    
    // Normalize and threshold
    cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX);
    
    cv::Mat edges;
    magnitude.convertTo(edges, CV_8U);
    cv::threshold(edges, edges, 50, 255, cv::THRESH_BINARY);
    
    return edges;
}

cv::Mat EdgeDetector::robertsEdgeDetection(const cv::Mat& image) {
    // Roberts cross kernels
    cv::Mat kernel_x = (cv::Mat_<float>(2, 2) <<
        1,  0,
        0, -1
    );
    
    cv::Mat kernel_y = (cv::Mat_<float>(2, 2) <<
        0,  1,
        -1, 0
    );
    
    cv::Mat grad_x, grad_y;
    cv::filter2D(image, grad_x, CV_32F, kernel_x);
    cv::filter2D(image, grad_y, CV_32F, kernel_y);
    
    cv::Mat magnitude;
    cv::magnitude(grad_x, grad_y, magnitude);
    
    cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX);
    
    cv::Mat edges;
    magnitude.convertTo(edges, CV_8U);
    cv::threshold(edges, edges, 50, 255, cv::THRESH_BINARY);
    
    return edges;
}

cv::Mat EdgeDetector::laplacianEdgeDetection(const cv::Mat& image,
                                             int ksize,
                                             double scale) {
    cv::Mat laplacian;
    cv::Laplacian(image, laplacian, CV_16S, ksize, scale);
    
    cv::Mat abs_laplacian;
    cv::convertScaleAbs(laplacian, abs_laplacian);
    
    cv::Mat edges;
    cv::threshold(abs_laplacian, edges, 50, 255, cv::THRESH_BINARY);
    
    return edges;
}

cv::Mat EdgeDetector::logEdgeDetection(const cv::Mat& image, double sigma) {
    // Gaussian blur
    cv::Mat blurred;
    int ksize = static_cast<int>(6 * sigma + 1);
    if (ksize % 2 == 0) ksize++;
    cv::GaussianBlur(image, blurred, cv::Size(ksize, ksize), sigma);
    
    // Laplacian
    cv::Mat laplacian;
    cv::Laplacian(blurred, laplacian, CV_16S, 3);
    
    // Zero crossings detection
    cv::Mat edges = cv::Mat::zeros(image.size(), CV_8U);
    
    for (int i = 1; i < laplacian.rows - 1; i++) {
        for (int j = 1; j < laplacian.cols - 1; j++) {
            short center = laplacian.at<short>(i, j);
            
            // Check for zero crossing
            bool zero_crossing = false;
            
            if (center > 0) {
                if (laplacian.at<short>(i-1, j) < 0 || laplacian.at<short>(i+1, j) < 0 ||
                    laplacian.at<short>(i, j-1) < 0 || laplacian.at<short>(i, j+1) < 0) {
                    zero_crossing = true;
                }
            } else if (center < 0) {
                if (laplacian.at<short>(i-1, j) > 0 || laplacian.at<short>(i+1, j) > 0 ||
                    laplacian.at<short>(i, j-1) > 0 || laplacian.at<short>(i, j+1) > 0) {
                    zero_crossing = true;
                }
            }
            
            if (zero_crossing) {
                edges.at<uchar>(i, j) = 255;
            }
        }
    }
    
    return edges;
}

cv::Mat EdgeDetector::dogEdgeDetection(const cv::Mat& image,
                                       double sigma1,
                                       double sigma2) {
    // Two Gaussian blurs
    cv::Mat blur1, blur2;
    
    int ksize1 = static_cast<int>(6 * sigma1 + 1);
    if (ksize1 % 2 == 0) ksize1++;
    cv::GaussianBlur(image, blur1, cv::Size(ksize1, ksize1), sigma1);
    
    int ksize2 = static_cast<int>(6 * sigma2 + 1);
    if (ksize2 % 2 == 0) ksize2++;
    cv::GaussianBlur(image, blur2, cv::Size(ksize2, ksize2), sigma2);
    
    // Difference
    cv::Mat dog = blur1 - blur2;
    
    // Threshold
    cv::Mat edges;
    cv::threshold(cv::abs(dog), edges, 10, 255, cv::THRESH_BINARY);
    
    return edges;
}

cv::Mat EdgeDetector::kirschEdgeDetection(const cv::Mat& image) {
    std::vector<cv::Mat> responses;
    applyKirschKernels(image, responses);
    
    // Combine responses (maximum)
    cv::Mat result = cv::Mat::zeros(image.size(), CV_32F);
    
    for (const auto& response : responses) {
        cv::max(result, response, result);
    }
    
    cv::normalize(result, result, 0, 255, cv::NORM_MINMAX);
    
    cv::Mat edges;
    result.convertTo(edges, CV_8U);
    cv::threshold(edges, edges, 50, 255, cv::THRESH_BINARY);
    
    return edges;
}

void EdgeDetector::applyKirschKernels(const cv::Mat& image,
                                      std::vector<cv::Mat>& responses) {
    auto kernels = generateKirschKernels();
    
    responses.clear();
    responses.reserve(kernels.size());
    
    for (const auto& kernel : kernels) {
        cv::Mat response;
        cv::filter2D(image, response, CV_32F, kernel);
        response = cv::abs(response);
        responses.push_back(response);
    }
}

std::vector<cv::Mat> EdgeDetector::generateKirschKernels() {
    std::vector<cv::Mat> kernels;
    
    // 8 directional kernels
    kernels.push_back((cv::Mat_<float>(3, 3) <<
        5,  5,  5,
        -3, 0, -3,
        -3, -3, -3));
    
    kernels.push_back((cv::Mat_<float>(3, 3) <<
        5,  5, -3,
        5,  0, -3,
        -3, -3, -3));
    
    kernels.push_back((cv::Mat_<float>(3, 3) <<
        5, -3, -3,
        5,  0, -3,
        5, -3, -3));
    
    kernels.push_back((cv::Mat_<float>(3, 3) <<
        -3, -3, -3,
         5,  0, -3,
         5,  5, -3));
    
    kernels.push_back((cv::Mat_<float>(3, 3) <<
        -3, -3, -3,
        -3,  0, -3,
         5,  5,  5));
    
    kernels.push_back((cv::Mat_<float>(3, 3) <<
        -3, -3, -3,
        -3,  0,  5,
        -3,  5,  5));
    
    kernels.push_back((cv::Mat_<float>(3, 3) <<
        -3, -3,  5,
        -3,  0,  5,
        -3, -3,  5));
    
    kernels.push_back((cv::Mat_<float>(3, 3) <<
        -3,  5,  5,
        -3,  0,  5,
        -3, -3, -3));
    
    return kernels;
}

cv::Mat EdgeDetector::robinsonEdgeDetection(const cv::Mat& image) {
    std::vector<cv::Mat> responses;
    applyRobinsonKernels(image, responses);
    
    cv::Mat result = cv::Mat::zeros(image.size(), CV_32F);
    
    for (const auto& response : responses) {
        cv::max(result, response, result);
    }
    
    cv::normalize(result, result, 0, 255, cv::NORM_MINMAX);
    
    cv::Mat edges;
    result.convertTo(edges, CV_8U);
    cv::threshold(edges, edges, 50, 255, cv::THRESH_BINARY);
    
    return edges;
}

void EdgeDetector::applyRobinsonKernels(const cv::Mat& image,
                                        std::vector<cv::Mat>& responses) {
    auto kernels = generateRobinsonKernels();
    
    responses.clear();
    responses.reserve(kernels.size());
    
    for (const auto& kernel : kernels) {
        cv::Mat response;
        cv::filter2D(image, response, CV_32F, kernel);
        response = cv::abs(response);
        responses.push_back(response);
    }
}

std::vector<cv::Mat> EdgeDetector::generateRobinsonKernels() {
    std::vector<cv::Mat> kernels;
    
    kernels.push_back((cv::Mat_<float>(3, 3) <<
        1,  1,  1,
        1, -2,  1,
        -1, -1, -1));
    
    kernels.push_back((cv::Mat_<float>(3, 3) <<
        1,  1,  1,
        -1, -2,  1,
        -1, -1,  1));
    
    kernels.push_back((cv::Mat_<float>(3, 3) <<
        -1,  1,  1,
        -1, -2,  1,
        -1,  1,  1));
    
    kernels.push_back((cv::Mat_<float>(3, 3) <<
        -1, -1,  1,
        -1, -2,  1,
         1,  1,  1));
    
    kernels.push_back((cv::Mat_<float>(3, 3) <<
        -1, -1, -1,
         1, -2,  1,
         1,  1,  1));
    
    kernels.push_back((cv::Mat_<float>(3, 3) <<
         1, -1, -1,
         1, -2, -1,
         1,  1,  1));
    
    kernels.push_back((cv::Mat_<float>(3, 3) <<
         1,  1, -1,
         1, -2, -1,
         1,  1, -1));
    
    kernels.push_back((cv::Mat_<float>(3, 3) <<
         1,  1,  1,
         1, -2, -1,
         1, -1, -1));
    
    return kernels;
}

cv::Mat EdgeDetector::freiChenEdgeDetection(const cv::Mat& image) {
    std::vector<cv::Mat> edge_responses, line_responses;
    applyFreiChenKernels(image, edge_responses, line_responses);
    
    // Compute edge strength
    cv::Mat edge_strength = cv::Mat::zeros(image.size(), CV_32F);
    
    for (const auto& response : edge_responses) {
        edge_strength += response.mul(response);
    }
    
    cv::sqrt(edge_strength, edge_strength);
    
    cv::normalize(edge_strength, edge_strength, 0, 255, cv::NORM_MINMAX);
    
    cv::Mat edges;
    edge_strength.convertTo(edges, CV_8U);
    cv::threshold(edges, edges, 50, 255, cv::THRESH_BINARY);
    
    return edges;
}

void EdgeDetector::applyFreiChenKernels(const cv::Mat& image,
                                        std::vector<cv::Mat>& edge_responses,
                                        std::vector<cv::Mat>& line_responses) {
    auto kernels = generateFreiChenKernels();
    
    edge_responses.clear();
    line_responses.clear();
    
    // First 4 kernels are edge detectors
    for (int i = 0; i < 4; i++) {
        cv::Mat response;
        cv::filter2D(image, response, CV_32F, kernels[i]);
        edge_responses.push_back(response);
    }
    
    // Next 4 kernels are line detectors
    for (int i = 4; i < 8; i++) {
        cv::Mat response;
        cv::filter2D(image, response, CV_32F, kernels[i]);
        line_responses.push_back(response);
    }
}

std::vector<cv::Mat> EdgeDetector::generateFreiChenKernels() {
    std::vector<cv::Mat> kernels;
    
    float sqrt2 = std::sqrt(2.0f);
    
    // Edge detection kernels
    kernels.push_back((cv::Mat_<float>(3, 3) <<
        1, sqrt2, 1,
        0, 0, 0,
        -1, -sqrt2, -1) / (2 * sqrt2));
    
    kernels.push_back((cv::Mat_<float>(3, 3) <<
        1, 0, -1,
        sqrt2, 0, -sqrt2,
        1, 0, -1) / (2 * sqrt2));
    
    kernels.push_back((cv::Mat_<float>(3, 3) <<
        0, -1, sqrt2,
        1, 0, -1,
        -sqrt2, 1, 0) / (2 * sqrt2));
    
    kernels.push_back((cv::Mat_<float>(3, 3) <<
        sqrt2, -1, 0,
        -1, 0, 1,
        0, 1, -sqrt2) / (2 * sqrt2));
    
    // Line detection kernels
    kernels.push_back((cv::Mat_<float>(3, 3) <<
        0, 1, 0,
        -1, 0, -1,
        0, 1, 0) / 2.0f);
    
    kernels.push_back((cv::Mat_<float>(3, 3) <<
        -1, 0, 1,
        0, 0, 0,
        1, 0, -1) / 2.0f);
    
    kernels.push_back((cv::Mat_<float>(3, 3) <<
        1, -2, 1,
        -2, 4, -2,
        1, -2, 1) / 6.0f);
    
    kernels.push_back((cv::Mat_<float>(3, 3) <<
        -2, 1, -2,
        1, 4, 1,
        -2, 1, -2) / 6.0f);
    
    return kernels;
}

cv::Mat EdgeDetector::dericheEdgeDetection(const cv::Mat& image, double alpha) {
    // Apply Deriche filter in x direction
    cv::Mat deriche_x = dericheDerivative(image, alpha);
    
    // Apply Deriche filter in y direction
    cv::Mat transposed;
    cv::transpose(image, transposed);
    cv::Mat deriche_y_t = dericheDerivative(transposed, alpha);
    cv::Mat deriche_y;
    cv::transpose(deriche_y_t, deriche_y);
    
    // Compute magnitude
    cv::Mat magnitude;
    cv::magnitude(deriche_x, deriche_y, magnitude);
    
    cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX);
    
    cv::Mat edges;
    magnitude.convertTo(edges, CV_8U);
    cv::threshold(edges, edges, 50, 255, cv::THRESH_BINARY);
    
    return edges;
}

cv::Mat EdgeDetector::dericheDerivative(const cv::Mat& image, double alpha) {
    // Deriche's recursive implementation of derivative of Gaussian
    cv::Mat output = cv::Mat::zeros(image.size(), CV_32F);
    cv::Mat float_img;
    image.convertTo(float_img, CV_32F);
    
    // Compute filter coefficients
    float k = (1.0f - std::exp(-alpha)) * (1.0f - std::exp(-alpha)) / 
              (1.0f + 2.0f * alpha * std::exp(-alpha) - std::exp(-2.0f * alpha));
    
    float a0 = k;
    float a1 = k * std::exp(-alpha) * (alpha - 1.0f);
    float a2 = k * std::exp(-alpha) * (alpha + 1.0f);
    float a3 = -k * std::exp(-2.0f * alpha);
    
    float b1 = 2.0f * std::exp(-alpha);
    float b2 = -std::exp(-2.0f * alpha);
    
    // Forward pass
    for (int i = 0; i < float_img.rows; i++) {
        for (int j = 2; j < float_img.cols; j++) {
            float val = a0 * float_img.at<float>(i, j) +
                       a1 * float_img.at<float>(i, j-1);
            
            if (j >= 2) {
                val += b1 * output.at<float>(i, j-1) + 
                      b2 * output.at<float>(i, j-2);
            }
            
            output.at<float>(i, j) = val;
        }
    }
    
    // Backward pass
    cv::Mat backward = cv::Mat::zeros(image.size(), CV_32F);
    for (int i = 0; i < float_img.rows; i++) {
        for (int j = float_img.cols - 3; j >= 0; j--) {
            float val = a2 * float_img.at<float>(i, j+1) +
                       a3 * float_img.at<float>(i, j+2);
            
            if (j < float_img.cols - 2) {
                val += b1 * backward.at<float>(i, j+1) +
                      b2 * backward.at<float>(i, j+2);
            }
            
            backward.at<float>(i, j) = val;
        }
    }
    
    output += backward;
    
    return output;
}

// Gradient computation
void EdgeDetector::computeGradients(const cv::Mat& image,
                                    cv::Mat& gradient_x,
                                    cv::Mat& gradient_y,
                                    int ksize) {
    cv::Sobel(image, gradient_x, CV_32F, 1, 0, ksize);
    cv::Sobel(image, gradient_y, CV_32F, 0, 1, ksize);
}

void EdgeDetector::computeGradientMagnitude(const cv::Mat& grad_x,
                                            const cv::Mat& grad_y,
                                            cv::Mat& magnitude) {
    cv::magnitude(grad_x, grad_y, magnitude);
}

void EdgeDetector::computeGradientDirection(const cv::Mat& grad_x,
                                            const cv::Mat& grad_y,
                                            cv::Mat& direction) {
    cv::phase(grad_x, grad_y, direction, true);  // Angle in degrees
}

// Non-maximum suppression
cv::Mat EdgeDetector::nonMaximumSuppression(const cv::Mat& magnitude,
                                            const cv::Mat& direction) {
    cv::Mat suppressed = cv::Mat::zeros(magnitude.size(), CV_32F);
    
    for (int i = 1; i < magnitude.rows - 1; i++) {
        for (int j = 1; j < magnitude.cols - 1; j++) {
            float angle = direction.at<float>(i, j);
            float mag = magnitude.at<float>(i, j);
            
            // Quantize direction to 4 directions (0, 45, 90, 135)
            int dir = quantizeDirection(angle);
            
            float mag1, mag2;
            
            switch (dir) {
                case 0: // Horizontal (0 or 180 degrees)
                    mag1 = magnitude.at<float>(i, j-1);
                    mag2 = magnitude.at<float>(i, j+1);
                    break;
                    
                case 1: // Diagonal (45 or 225 degrees)
                    mag1 = magnitude.at<float>(i-1, j+1);
                    mag2 = magnitude.at<float>(i+1, j-1);
                    break;
                    
                case 2: // Vertical (90 or 270 degrees)
                    mag1 = magnitude.at<float>(i-1, j);
                    mag2 = magnitude.at<float>(i+1, j);
                    break;
                    
                case 3: // Diagonal (135 or 315 degrees)
                    mag1 = magnitude.at<float>(i-1, j-1);
                    mag2 = magnitude.at<float>(i+1, j+1);
                    break;
                    
                default:
                    mag1 = mag2 = 0;
            }
            
            // Keep only local maxima
            if (mag >= mag1 && mag >= mag2) {
                suppressed.at<float>(i, j) = mag;
            }
        }
    }
    
    return suppressed;
}

int EdgeDetector::quantizeDirection(float angle) {
    // Normalize angle to [0, 180)
    while (angle < 0) angle += 180;
    while (angle >= 180) angle -= 180;
    
    if (angle < 22.5 || angle >= 157.5) return 0;      // Horizontal
    else if (angle >= 22.5 && angle < 67.5) return 1;   // 45 degrees
    else if (angle >= 67.5 && angle < 112.5) return 2;  // Vertical
    else return 3;                                       // 135 degrees
}

// Hy
