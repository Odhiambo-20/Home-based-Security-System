// fingerprint_scanner.cpp - Complete Hand Scanner Implementation
#include "fingerprint_scanner.h"
#include "../ai_ml/neural_network.h"
#include "../computer_vision/image_preprocessor.h"
#include "../utils/logger.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <algorithm>
#include <numeric>
#include <fstream>

FingerprintScanner::FingerprintScanner()
    : initialized_(false)
    , scanning_(false)
    , sensor_fd_(-1)
    , sensor_handle_(nullptr)
    , matching_threshold_(0.85f)
    , quality_threshold_(40.0f)
    , min_minutiae_count_(30)
    , image_width_(800)
    , image_height_(600)
    , image_dpi_(500)
{
    // Initialize Gabor filter parameters
    gabor_params_.wavelengths = {4.0f, 8.0f, 16.0f};
    gabor_params_.orientations = {0.0f, M_PI/4, M_PI/2, 3*M_PI/4};
    gabor_params_.sigma = 5.0f;
    gabor_params_.gamma = 0.5f;
    
    metrics_ = {0.0f, 0.0f, 0, 0, 0};
}

FingerprintScanner::~FingerprintScanner() {
    shutdown();
}

bool FingerprintScanner::initialize(const std::string& config_path) {
    Logger::info("Initializing Fingerprint Scanner for whole hand");
    
    try {
        // Initialize hardware sensor
        if (!initializeSensor()) {
            Logger::error("Failed to initialize fingerprint sensor hardware");
            return false;
        }
        
        // Load CNN model for feature extraction
        fingerprint_cnn_ = std::make_unique<NeuralNetwork>();
        if (!fingerprint_cnn_->loadModel("/usr/local/share/biometric_security/models/fingerprint_cnn.tflite")) {
            Logger::error("Failed to load fingerprint CNN model");
            return false;
        }
        
        // Initialize image preprocessor
        preprocessor_ = std::make_unique<ImagePreprocessor>();
        
        initialized_ = true;
        status_message_ = "Fingerprint scanner ready for whole hand capture";
        Logger::info("Fingerprint scanner initialized successfully");
        
        return true;
        
    } catch (const std::exception& e) {
        Logger::error(std::string("Fingerprint scanner initialization failed: ") + e.what());
        return false;
    }
}

void FingerprintScanner::shutdown() {
    if (initialized_) {
        Logger::info("Shutting down fingerprint scanner");
        releaseSensor();
        fingerprint_cnn_.reset();
        preprocessor_.reset();
        initialized_ = false;
    }
}

HandScanData FingerprintScanner::captureHandScan(int timeout_ms) {
    std::lock_guard<std::mutex> lock(scan_mutex_);
    
    auto start_time = std::chrono::steady_clock::now();
    HandScanData hand_data;
    hand_data.scan_complete = false;
    hand_data.scan_id = generateUniqueId();
    hand_data.timestamp = std::chrono::system_clock::now();
    
    Logger::info("Starting whole hand capture");
    
    // Wait for hand presence
    int wait_count = 0;
    while (!detectHandPresence() && wait_count < timeout_ms / 100) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        wait_count++;
    }
    
    if (wait_count >= timeout_ms / 100) {
        Logger::warning("Hand presence detection timeout");
        status_message_ = "No hand detected - place hand on scanner";
        return hand_data;
    }
    
    // Capture full hand image
    cv::Mat full_hand = captureRawImage();
    if (full_hand.empty()) {
        Logger::error("Failed to capture hand image");
        status_message_ = "Image capture failed";
        return hand_data;
    }
    
    hand_data.full_hand_image = full_hand.clone();
    
    // Detect hand side (left or right)
    hand_data.hand_side = detectHandSide(full_hand);
    Logger::info("Detected " + handSideToString(hand_data.hand_side) + " hand");
    
    // Segment individual fingers from whole hand
    std::vector<cv::Rect> finger_rois = segmentFingers(full_hand);
    
    if (finger_rois.size() < 4) {
        Logger::warning("Could not segment all fingers - only found " + 
                       std::to_string(finger_rois.size()));
        status_message_ = "Please position hand properly - all fingers must be visible";
        return hand_data;
    }
    
    // Process each finger
    hand_data.fingers.reserve(5);
    float total_quality = 0.0f;
    int valid_fingers = 0;
    
    for (size_t i = 0; i < finger_rois.size() && i < 5; i++) {
        FingerData finger_data;
        
        // Extract finger ROI
        cv::Mat finger_roi = full_hand(finger_rois[i]).clone();
        finger_data.image = finger_roi;
        finger_data.capture_time = std::chrono::system_clock::now();
        
        // Identify which finger this is
        finger_data.finger_type = identifyFinger(finger_roi, hand_data.hand_side);
        Logger::info("Processing " + fingerTypeToString(finger_data.finger_type));
        
        // Preprocess fingerprint
        finger_data.processed_image = preprocessFingerprint(finger_roi);
        
        // Extract minutiae
        cv::Mat thinned = thinRidges(finger_data.processed_image);
        finger_data.minutiae = extractMinutiae(thinned);
        
        // Extract deep learning features
        finger_data.feature_vector = extractDeepFeatures(finger_data.processed_image);
        
        // Assess quality
        finger_data.quality = assessQuality(finger_data);
        finger_data.is_valid = finger_data.quality.is_acceptable;
        
        if (finger_data.is_valid) {
            total_quality += finger_data.quality.nfiq_score;
            valid_fingers++;
        }
        
        hand_data.fingers.push_back(std::move(finger_data));
    }
    
    // Calculate overall quality
    hand_data.overall_quality = valid_fingers > 0 ? 
                                 total_quality / valid_fingers : 0.0f;
    
    // Check if scan is complete and acceptable
    hand_data.scan_complete = (valid_fingers >= 4) && 
                             (hand_data.overall_quality >= quality_threshold_);
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    Logger::info("Hand scan complete in " + std::to_string(duration) + "ms");
    Logger::info("Valid fingers: " + std::to_string(valid_fingers) + "/5");
    Logger::info("Overall quality: " + std::to_string(hand_data.overall_quality));
    
    status_message_ = hand_data.scan_complete ? 
                     "Hand scan successful" : 
                     "Hand scan quality insufficient - please retry";
    
    // Update metrics
    metrics_.total_scans++;
    metrics_.avg_capture_time_ms = 
        (metrics_.avg_capture_time_ms * (metrics_.total_scans - 1) + duration) / 
        metrics_.total_scans;
    
    if (hand_data.scan_complete) {
        metrics_.successful_scans++;
    } else {
        metrics_.failed_scans++;
    }
    
    return hand_data;
}

bool FingerprintScanner::detectHandPresence() {
    // Capture quick preview image
    cv::Mat preview = captureRawImage();
    if (preview.empty()) return false;
    
    // Convert to grayscale
    cv::Mat gray;
    if (preview.channels() == 3) {
        cv::cvtColor(preview, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = preview;
    }
    
    // Calculate mean intensity in center region
    int cx = gray.cols / 2;
    int cy = gray.rows / 2;
    int w = gray.cols / 3;
    int h = gray.rows / 3;
    
    cv::Rect center_roi(cx - w/2, cy - h/2, w, h);
    cv::Mat center_region = gray(center_roi);
    
    double mean_intensity = cv::mean(center_region)[0];
    
    // Hand present if significant intensity detected
    // (adjust threshold based on sensor characteristics)
    return mean_intensity > 30.0 && mean_intensity < 220.0;
}

HandSide FingerprintScanner::detectHandSide(const cv::Mat& hand_image) {
    // Convert to grayscale
    cv::Mat gray;
    if (hand_image.channels() == 3) {
        cv::cvtColor(hand_image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = hand_image.clone();
    }
    
    // Threshold to get hand mask
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    
    // Find largest contour (hand)
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) return HandSide::UNKNOWN;
    
    // Get largest contour
    auto largest = std::max_element(contours.begin(), contours.end(),
        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
            return cv::contourArea(a) < cv::contourArea(b);
        });
    
    // Find convex hull
    std::vector<cv::Point> hull;
    cv::convexHull(*largest, hull);
    
    // Find convexity defects
    std::vector<int> hull_indices;
    cv::convexHull(*largest, hull_indices);
    
    std::vector<cv::Vec4i> defects;
    if (hull_indices.size() > 3) {
        cv::convexityDefects(*largest, hull_indices, defects);
    }
    
    // Detect thumb position by finding largest defect (between thumb and index)
    if (!defects.empty()) {
        int max_depth_idx = 0;
        float max_depth = 0;
        
        for (size_t i = 0; i < defects.size(); i++) {
            float depth = defects[i][3] / 256.0f;
            if (depth > max_depth) {
                max_depth = depth;
                max_depth_idx = i;
            }
        }
        
        // Get defect point (valley between thumb and index)
        cv::Point defect_point = (*largest)[defects[max_depth_idx][2]];
        
        // Calculate hand centroid
        cv::Moments m = cv::moments(*largest);
        cv::Point2f centroid(m.m10 / m.m00, m.m01 / m.m00);
        
        // Thumb on left side = right hand, thumb on right = left hand
        if (defect_point.x < centroid.x) {
            return HandSide::RIGHT;
        } else {
            return HandSide::LEFT;
        }
    }
    
    return HandSide::UNKNOWN;
}

std::vector<cv::Rect> FingerprintScanner::segmentFingers(const cv::Mat& hand_image) {
    std::vector<cv::Rect> finger_rois;
    
    // Convert to grayscale
    cv::Mat gray;
    if (hand_image.channels() == 3) {
        cv::cvtColor(hand_image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = hand_image.clone();
    }
    
    // Threshold
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    
    // Morphological operations to clean up
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
    
    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) return finger_rois;
    
    // Get hand contour
    auto hand_contour = *std::max_element(contours.begin(), contours.end(),
        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
            return cv::contourArea(a) < cv::contourArea(b);
        });
    
    // Detect fingertips
    std::vector<cv::Point> fingertips = detectFingerTips(hand_contour);
    
    // For each fingertip, create a bounding box for the finger
    for (const auto& tip : fingertips) {
        // Define finger region (from tip downward)
        int finger_width = 80;   // Approximate finger width in pixels
        int finger_length = 200;  // Approximate finger length
        
        cv::Point top_left(tip.x - finger_width/2, tip.y);
        cv::Point bottom_right(tip.x + finger_width/2, tip.y + finger_length);
        
        // Clamp to image boundaries
        top_left.x = std::max(0, top_left.x);
        top_left.y = std::max(0, top_left.y);
        bottom_right.x = std::min(hand_image.cols - 1, bottom_right.x);
        bottom_right.y = std::min(hand_image.rows - 1, bottom_right.y);
        
        cv::Rect finger_roi(top_left, bottom_right);
        
        // Validate ROI has minimum size
        if (finger_roi.width > 40 && finger_roi.height > 100) {
            finger_rois.push_back(finger_roi);
        }
    }
    
    // Sort finger ROIs from left to right
    std::sort(finger_rois.begin(), finger_rois.end(),
        [](const cv::Rect& a, const cv::Rect& b) {
            return a.x < b.x;
        });
    
    Logger::info("Segmented " + std::to_string(finger_rois.size()) + " fingers");
    
    return finger_rois;
}

std::vector<cv::Point> FingerprintScanner::detectFingerTips(
    const std::vector<cv::Point>& hand_contour) {
    
    std::vector<cv::Point> fingertips;
    
    // Find convex hull
    std::vector<cv::Point> hull;
    cv::convexHull(hand_contour, hull);
    
    // Find convexity defects
    std::vector<int> hull_indices;
    cv::convexHull(hand_contour, hull_indices);
    
    std::vector<cv::Vec4i> defects;
    if (hull_indices.size() > 3) {
        cv::convexityDefects(hand_contour, hull_indices, defects);
    }
    
    // Fingertips are points on convex hull that are far from defects
    cv::Moments m = cv::moments(hand_contour);
    cv::Point2f centroid(m.m10 / m.m00, m.m01 / m.m00);
    
    for (const auto& point : hull) {
        float dist_to_centroid = cv::norm(point - centroid);
        
        // Fingertip criteria:
        // 1. Far from centroid (finger extended)
        // 2. In upper region of hand
        // 3. Not too close to other fingertips
        
        if (dist_to_centroid > 100 && point.y < centroid.y) {
            // Check not too close to existing tips
            bool too_close = false;
            for (const auto& tip : fingertips) {
                if (cv::norm(point - tip) < 50) {
                    too_close = true;
                    break;
                }
            }
            
            if (!too_close) {
                fingertips.push_back(point);
            }
        }
    }
    
    // Limit to 5 fingertips
    if (fingertips.size() > 5) {
        // Keep the 5 highest (lowest y-coordinate)
        std::sort(fingertips.begin(), fingertips.end(),
            [](const cv::Point& a, const cv::Point& b) {
                return a.y < b.y;
            });
        fingertips.resize(5);
    }
    
    return fingertips;
}

FingerType FingerprintScanner::identifyFinger(const cv::Mat& finger_roi, 
                                              const HandSide& hand_side) {
    // This is a simplified version - in production, use ML-based classification
    // Based on position and size characteristics
    
    int width = finger_roi.cols;
    int height = finger_roi.rows;
    float aspect_ratio = static_cast<float>(height) / width;
    
    // Thumb is typically wider and shorter
    if (aspect_ratio < 2.0f && width > 70) {
        return FingerType::THUMB;
    }
    // Other fingers based on typical proportions
    else if (aspect_ratio > 2.5f) {
        return FingerType::MIDDLE;  // Middle finger typically longest
    }
    else if (width < 60) {
        return FingerType::PINKY;   // Pinky typically thinnest
    }
    else {
        // Could be index or ring - would need position information
        return FingerType::INDEX;
    }
}

cv::Mat FingerprintScanner::preprocessFingerprint(const cv::Mat& raw_image) {
    cv::Mat processed = raw_image.clone();
    
    // Convert to grayscale if needed
    if (processed.channels() == 3) {
        cv::cvtColor(processed, processed, cv::COLOR_BGR2GRAY);
    }
    
    // 1. Normalization
    processed = normalizeImage(processed);
    
    // 2. Ridge enhancement using oriented Gabor filters
    processed = enhanceRidges(processed);
    
    // 3. Binarization
    processed = binarizeFingerprint(processed);
    
    return processed;
}

cv::Mat FingerprintScanner::normalizeImage(const cv::Mat& image) {
    cv::Mat normalized;
    cv::Mat float_img;
    image.convertTo(float_img, CV_32F);
    
    // Compute mean and std
    cv::Scalar mean, stddev;
    cv::meanStdDev(float_img, mean, stddev);
    
    // Normalize to mean=100, std=20
    float desired_mean = 100.0f;
    float desired_std = 20.0f;
    
    normalized = (float_img - mean[0]) * (desired_std / stddev[0]) + desired_mean;
    normalized.convertTo(normalized, CV_8U);
    
    return normalized;
}

cv::Mat FingerprintScanner::enhanceRidges(const cv::Mat& image) {
    // Compute orientation field
    cv::Mat orientation = computeOrientationField(image);
    
    // Apply oriented Gabor filtering
    cv::Mat enhanced = cv::Mat::zeros(image.size(), CV_32F);
    
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            float orient = orientation.at<float>(i, j);
            
            // Apply Gabor filter at local orientation
            cv::Mat gabor = applyGaborFilter(image, 8.0f, orient);
            enhanced.at<float>(i, j) = gabor.at<float>(i, j);
        }
    }
    
    cv::Mat result;
    cv::normalize(enhanced, result, 0, 255, cv::NORM_MINMAX, CV_8U);
    
    return result;
}

// Continued in next part due to length...
std::string generateUniqueId() {
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    return "SCAN_" + std::to_string(timestamp);
}

bool FingerprintScanner::initializeSensor() {
    // Open SPI device for fingerprint sensor
    sensor_fd_ = open("/dev/spidev0.0", O_RDWR);
    if (sensor_fd_ < 0) {
        Logger::error("Failed to open fingerprint sensor device");
        return false;
    }
    
    // Initialize sensor (sensor-specific commands would go here)
    Logger::info("Fingerprint sensor hardware initialized");
    return true;
}

void FingerprintScanner::releaseSensor() {
    if (sensor_fd_ >= 0) {
        close(sensor_fd_);
        sensor_fd_ = -1;
    }
}

cv::Mat FingerprintScanner::captureRawImage() {
    // Placeholder - actual implementation would read from sensor
    // For now, capture from camera/sensor device
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        return cv::Mat();
    }
    
    cv::Mat frame;
    cap >> frame;
    cap.release();
    
    return frame;
}

// Additional helper implementations would continue...
