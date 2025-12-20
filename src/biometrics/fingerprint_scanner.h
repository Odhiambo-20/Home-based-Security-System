// fingerprint_scanner.h - Comprehensive Hand Fingerprint Scanner
// Scans all 5 fingers simultaneously or sequentially
#ifndef FINGERPRINT_SCANNER_H
#define FINGERPRINT_SCANNER_H

#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <atomic>
#include <mutex>
#include <chrono>

// Forward declarations
class NeuralNetwork;
class ImagePreprocessor;

// Finger identification enum
enum class FingerType {
    THUMB = 0,
    INDEX = 1,
    MIDDLE = 2,
    RING = 3,
    PINKY = 4,
    UNKNOWN = 5
};

// Hand side identification
enum class HandSide {
    LEFT,
    RIGHT,
    UNKNOWN
};

// Fingerprint quality metrics
struct FingerprintQuality {
    float nfiq_score;          // NFIQ 2.0 quality score (0-100)
    float clarity;             // Image clarity (0-1)
    float contrast;            // Contrast level (0-1)
    int minutiae_count;        // Number of detected minutiae
    float ridge_frequency;     // Average ridge frequency
    bool is_acceptable;        // Overall quality acceptable
};

// Individual minutiae point
struct Minutiae {
    cv::Point2f position;      // (x, y) coordinates
    float orientation;         // Ridge orientation in radians
    enum Type { ENDING, BIFURCATION } type;
    float quality;             // Minutiae quality score
};

// Single finger data structure
struct FingerData {
    FingerType finger_type;
    cv::Mat image;                           // Raw fingerprint image
    cv::Mat processed_image;                 // Preprocessed image
    std::vector<Minutiae> minutiae;          // Detected minutiae points
    std::vector<float> feature_vector;       // Deep learning features (128D)
    FingerprintQuality quality;
    bool is_valid;
    std::chrono::system_clock::time_point capture_time;
};

// Complete hand scan data
struct HandScanData {
    HandSide hand_side;
    std::vector<FingerData> fingers;         // All 5 fingers
    cv::Mat full_hand_image;                 // Complete hand image
    float overall_quality;                   // Average quality across fingers
    bool scan_complete;
    std::string scan_id;                     // Unique identifier
    std::chrono::system_clock::time_point timestamp;
};

// Fingerprint template for storage
struct FingerprintTemplate {
    std::string user_id;
    HandSide hand_side;
    std::vector<std::vector<Minutiae>> finger_minutiae;  // 5 fingers
    std::vector<std::vector<float>> finger_embeddings;   // 5 x 128D vectors
    std::vector<uint8_t> encrypted_data;                 // Encrypted template
    std::chrono::system_clock::time_point enrollment_time;
};

// Matching result structure
struct MatchResult {
    bool is_match;
    float confidence_score;       // 0-1, higher is better
    std::vector<float> finger_scores;  // Individual finger match scores
    int matched_minutiae_count;
    std::string matched_user_id;
    float false_accept_rate;      // Estimated FAR at this threshold
};

class FingerprintScanner {
public:
    FingerprintScanner();
    ~FingerprintScanner();
    
    // Initialization and configuration
    bool initialize(const std::string& config_path);
    void shutdown();
    bool isInitialized() const { return initialized_; }
    
    // Hand scanning operations
    HandScanData captureHandScan(int timeout_ms = 30000);
    bool captureSequentialFingers(HandScanData& hand_data);
    bool detectHandPresence();
    HandSide detectHandSide(const cv::Mat& hand_image);
    
    // Finger segmentation from whole hand
    std::vector<cv::Rect> segmentFingers(const cv::Mat& hand_image);
    FingerType identifyFinger(const cv::Mat& finger_roi, const HandSide& hand_side);
    
    // Image preprocessing
    cv::Mat preprocessFingerprint(const cv::Mat& raw_image);
    cv::Mat enhanceRidges(const cv::Mat& image);
    cv::Mat normalizeImage(const cv::Mat& image);
    cv::Mat binarizeFingerprint(const cv::Mat& image);
    cv::Mat thinRidges(const cv::Mat& binary_image);
    
    // Feature extraction
    std::vector<Minutiae> extractMinutiae(const cv::Mat& thinned_image);
    std::vector<float> extractDeepFeatures(const cv::Mat& fingerprint);
    FingerprintQuality assessQuality(const FingerData& finger_data);
    
    // Template operations
    FingerprintTemplate createTemplate(const HandScanData& hand_scan, 
                                       const std::string& user_id);
    bool saveTemplate(const FingerprintTemplate& fp_template, 
                     const std::string& filepath);
    FingerprintTemplate loadTemplate(const std::string& filepath);
    
    // Matching operations
    MatchResult matchFingerprints(const HandScanData& probe, 
                                 const FingerprintTemplate& gallery);
    float computeSimilarity(const std::vector<float>& vec1, 
                          const std::vector<float>& vec2);
    float matchMinutiae(const std::vector<Minutiae>& probe_minutiae,
                       const std::vector<Minutiae>& gallery_minutiae);
    
    // Multi-finger fusion
    float fuseFingerScores(const std::vector<float>& individual_scores);
    
    // Liveness detection helpers
    bool checkFingerLiveness(const FingerData& finger_data);
    float detectSpoofing(const cv::Mat& fingerprint_image);
    
    // Utility functions
    void visualizeMinutiae(cv::Mat& image, 
                          const std::vector<Minutiae>& minutiae);
    void visualizeHandSegmentation(cv::Mat& image, 
                                   const std::vector<cv::Rect>& finger_rois);
    std::string getStatusMessage() const { return status_message_; }
    
    // Configuration
    void setMatchingThreshold(float threshold) { matching_threshold_ = threshold; }
    void setQualityThreshold(float threshold) { quality_threshold_ = threshold; }
    void setMinMinutiaeCount(int count) { min_minutiae_count_ = count; }

private:
    // Hardware interface
    bool initializeSensor();
    cv::Mat captureRawImage();
    void releaseSensor();
    
    // Image processing helpers
    cv::Mat applyGaborFilter(const cv::Mat& image, float wavelength, 
                            float orientation);
    cv::Mat computeOrientationField(const cv::Mat& image);
    cv::Mat computeFrequencyField(const cv::Mat& image);
    
    // Minutiae processing
    void filterFalseMinutiae(std::vector<Minutiae>& minutiae);
    int computeCrossingNumber(const cv::Mat& image, int x, int y);
    float computeMinutiaeQuality(const Minutiae& minutia, 
                                const cv::Mat& image);
    
    // Deep learning inference
    bool loadNeuralNetwork(const std::string& model_path);
    std::vector<float> runInference(const cv::Mat& input_image);
    
    // Hand analysis
    std::vector<cv::Point> detectFingerTips(const cv::Mat& hand_contour);
    std::vector<cv::Point> detectFingerValleys(const cv::Mat& hand_contour);
    cv::RotatedRect fitHandBoundingBox(const cv::Mat& hand_mask);
    
    // Security
    std::vector<uint8_t> encryptTemplate(const FingerprintTemplate& fp_template);
    FingerprintTemplate decryptTemplate(const std::vector<uint8_t>& encrypted_data);
    
    // Member variables
    std::atomic<bool> initialized_;
    std::atomic<bool> scanning_;
    std::mutex scan_mutex_;
    
    // Hardware handles
    int sensor_fd_;                    // File descriptor for sensor device
    void* sensor_handle_;              // Sensor hardware handle
    
    // Configuration parameters
    float matching_threshold_;         // Default: 0.85
    float quality_threshold_;          // Default: 40.0 (NFIQ)
    int min_minutiae_count_;          // Default: 30
    int image_width_;                 // Sensor image width
    int image_height_;                // Sensor image height
    int image_dpi_;                   // 500 or 1000 DPI
    
    // AI/ML components
    std::unique_ptr<NeuralNetwork> fingerprint_cnn_;
    std::unique_ptr<ImagePreprocessor> preprocessor_;
    
    // Processing parameters
    struct GaborParams {
        std::vector<float> wavelengths;
        std::vector<float> orientations;
        float sigma;
        float gamma;
    } gabor_params_;
    
    // Status
    std::string status_message_;
    std::chrono::system_clock::time_point last_scan_time_;
    
    // Calibration data
    cv::Mat sensor_calibration_matrix_;
    cv::Mat sensor_distortion_coeffs_;
    
    // Performance metrics
    struct PerformanceMetrics {
        float avg_capture_time_ms;
        float avg_processing_time_ms;
        int total_scans;
        int successful_scans;
        int failed_scans;
    } metrics_;
};

// Helper functions (inline for performance)
inline float euclideanDistance(const cv::Point2f& p1, const cv::Point2f& p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

inline float angleDifference(float angle1, float angle2) {
    float diff = std::abs(angle1 - angle2);
    if (diff > M_PI) diff = 2 * M_PI - diff;
    return diff;
}

inline std::string fingerTypeToString(FingerType type) {
    switch(type) {
        case FingerType::THUMB: return "Thumb";
        case FingerType::INDEX: return "Index";
        case FingerType::MIDDLE: return "Middle";
        case FingerType::RING: return "Ring";
        case FingerType::PINKY: return "Pinky";
        default: return "Unknown";
    }
}

inline std::string handSideToString(HandSide side) {
    switch(side) {
        case HandSide::LEFT: return "Left";
        case HandSide::RIGHT: return "Right";
        default: return "Unknown";
    }
}

#endif // FINGERPRINT_SCANNER_H
