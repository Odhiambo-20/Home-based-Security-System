// fingerprint_scanner.h - Industrial Production Fingerprint Scanner
// Multi-modal biometric authentication system with advanced security
// Copyright (c) 2025 - All Rights Reserved
// Compliant with: ISO/IEC 19794-2, ANSI/NIST-ITL, FBI EBTS, NFIQ 2.0

#ifndef FINGERPRINT_SCANNER_H
#define FINGERPRINT_SCANNER_H

#include <vector>
#include <string>
#include <memory>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <chrono>
#include <queue>
#include <condition_variable>
#include <functional>
#include <unordered_map>
#include <array>
#include <optional>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <openssl/evp.h>

// Constants
constexpr int MAX_FINGERS = 5;
constexpr int MINUTIAE_DESCRIPTOR_SIZE = 128;
constexpr int DEEP_FEATURE_DIMENSION = 512;
constexpr float DEFAULT_MATCHING_THRESHOLD = 0.85f;

// Enumerations
enum class FingerType : uint8_t { THUMB, INDEX, MIDDLE, RING, PINKY, UNKNOWN = 255 };
enum class HandSide : uint8_t { LEFT, RIGHT, UNKNOWN = 255 };
enum class SensorStatus : uint8_t { UNINITIALIZED, READY, CAPTURING, PROCESSING, ERROR };
enum class MinutiaeType : uint8_t { RIDGE_ENDING, BIFURCATION, DOT, SPUR };
enum class LivenessStatus : uint8_t { LIVE_FINGER, SPOOF_DETECTED, UNCERTAIN };
enum class QualityLevel : uint8_t { EXCELLENT, GOOD, FAIR, POOR, REJECTED };
enum class ScannerError : int32_t {
    SUCCESS = 0, HARDWARE_FAILURE = -1, SENSOR_NOT_RESPONDING = -2,
    NO_FINGER_DETECTED = -4, POOR_IMAGE_QUALITY = -5, TIMEOUT = -8
};

// Core structures
struct MinutiaePoint {
    cv::Point2f position;
    float orientation;
    MinutiaeType type;
    float quality;
    std::array<float, 8> descriptor;
};

struct RidgeFeatures {
    cv::Mat orientation_field;
    cv::Mat frequency_field;
    cv::Mat quality_map;
    float mean_frequency;
};

struct FingerprintQualityMetrics {
    float nfiq2_score;
    float clarity_score;
    int minutiae_count;
    QualityLevel quality_level;
    bool is_acceptable;
};

struct LivenessFeatures {
    float blood_flow_score;
    float perspiration_score;
    float multispectral_response[4];
    LivenessStatus status;
    float confidence;
};

struct FingerBiometricData {
    FingerType finger_type;
    cv::Mat raw_image;
    cv::Mat enhanced_image;
    std::vector<MinutiaePoint> minutiae;
    RidgeFeatures ridge_features;
    Eigen::VectorXf deep_embedding;
    FingerprintQualityMetrics quality;
    LivenessFeatures liveness;
    bool is_valid;
    std::chrono::system_clock::time_point capture_timestamp;
};

struct HandGeometry {
    float palm_width_mm;
    std::array<float, 5> finger_lengths_mm;
    cv::Point2f palm_center;
};

struct HandBiometricData {
    HandSide hand_side;
    std::array<FingerBiometricData, MAX_FINGERS> fingers;
    cv::Mat full_hand_image;
    HandGeometry geometry;
    float overall_quality;
    bool scan_complete;
    int valid_finger_count;
    std::string scan_id;
    std::chrono::system_clock::time_point timestamp;
};

struct BiometricTemplate {
    std::string user_id;
    std::string template_id;
    HandSide hand_side;
    std::array<std::vector<MinutiaePoint>, MAX_FINGERS> finger_minutiae;
    std::array<Eigen::VectorXf, MAX_FINGERS> finger_embeddings;
    HandGeometry hand_geometry;
    std::vector<uint8_t> encrypted_blob;
    std::array<uint8_t, 32> template_hash;
    std::chrono::system_clock::time_point enrollment_time;
};

struct MatchingScore {
    float minutiae_score;
    float embedding_score;
    float geometric_score;
    float combined_score;
};

struct FingerMatchResult {
    FingerType finger_type;
    bool matched;
    MatchingScore scores;
    int matched_minutiae_count;
    float confidence;
};

struct VerificationResult {
    bool is_match;
    float overall_confidence;
    std::array<FingerMatchResult, MAX_FINGERS> finger_results;
    int total_matched_fingers;
    LivenessStatus liveness_status;
    std::string matched_user_id;
    uint32_t matching_time_ms;
};

struct IdentificationResult {
    bool identification_successful;
    std::vector<std::pair<std::string, float>> candidates; // user_id, score
    uint32_t search_time_ms;
};

// Configuration
struct SystemConfig {
    struct {
        uint32_t capture_width = 800;
        uint32_t capture_height = 600;
        std::string sensor_device_path = "/dev/spidev0.0";
    } sensor;
    
    struct {
        std::vector<float> gabor_wavelengths = {4.0f, 8.0f, 16.0f};
        int max_minutiae_per_finger = 150;
        bool use_gpu = true;
    } processing;
    
    struct {
        float verification_threshold = 0.85f;
        float identification_threshold = 0.90f;
    } matching;
    
    struct {
        bool require_liveness = true;
        bool encrypt_templates = true;
        std::string encryption_key_path;
    } security;
};

// Main class
class FingerprintScanner {
public:
    FingerprintScanner();
    ~FingerprintScanner();
    
    // Core operations
    ScannerError initialize(const SystemConfig& config);
    ScannerError shutdown();
    
    ScannerError captureFullHand(HandBiometricData& output, int timeout_ms = 30000);
    ScannerError captureSingleFinger(FingerBiometricData& output, FingerType expected);
    
    bool detectHandPresence(float& confidence);
    HandSide detectHandSide(const cv::Mat& hand_image, float& confidence);
    std::vector<cv::Rect> segmentFingers(const cv::Mat& hand_image);
    
    // Image processing
    cv::Mat normalizeImage(const cv::Mat& image);
    cv::Mat enhanceRidges(const cv::Mat& image, const cv::Mat& orientation);
    cv::Mat binarizeFingerprint(const cv::Mat& enhanced);
    cv::Mat thinRidges(const cv::Mat& binary);
    
    // Feature extraction
    RidgeFeatures extractRidgeFeatures(const cv::Mat& image, const cv::Mat& mask);
    cv::Mat computeOrientationField(const cv::Mat& image, int block_size = 16);
    std::vector<MinutiaePoint> extractMinutiae(const cv::Mat& thinned,
                                               const cv::Mat& orientation);
    Eigen::VectorXf extractDeepFeatures(const cv::Mat& fingerprint);
    
    // Quality and liveness
    FingerprintQualityMetrics assessQuality(const FingerBiometricData& data);
    LivenessFeatures detectLiveness(const FingerBiometricData& data);
    
    // Template management
    BiometricTemplate createTemplate(const HandBiometricData& hand_data,
                                    const std::string& user_id);
    ScannerError saveTemplate(const BiometricTemplate& temp, const std::string& path);
    ScannerError loadTemplate(const std::string& path, BiometricTemplate& temp);
    
    // Matching
    VerificationResult verifyFingerprint(const HandBiometricData& probe,
                                        const BiometricTemplate& gallery);
    IdentificationResult identifyFingerprint(const HandBiometricData& probe,
                                            const std::vector<BiometricTemplate>& gallery);
    
    // Utilities
    SensorStatus getStatus() const { return status_; }
    std::string getStatusMessage() const;

private:
    // Hardware
    ScannerError initializeHardware();
    cv::Mat captureRawImageFromSensor();
    
    // Processing helpers
    cv::Mat applyGaborFilter(const cv::Mat& image, float wavelength, float orientation);
    int computeCrossingNumber(const cv::Mat& image, int x, int y);
    void filterFalseMinutiae(std::vector<MinutiaePoint>& minutiae);
    
    // Matching helpers
    float computeMinutiaeScore(const std::vector<MinutiaePoint>& probe,
                              const std::vector<MinutiaePoint>& gallery);
    float computeEmbeddingSimilarity(const Eigen::VectorXf& e1, const Eigen::VectorXf& e2);
    
    // Deep learning
    ScannerError loadNeuralNetworks();
    Eigen::VectorXf runCNNInference(const cv::Mat& input);
    
    // Security
    std::vector<uint8_t> encryptData(const std::vector<uint8_t>& data);
    std::vector<uint8_t> decryptData(const std::vector<uint8_t>& data);
    
    // Member variables
    std::atomic<SensorStatus> status_{SensorStatus::UNINITIALIZED};
    mutable std::shared_mutex mutex_;
    SystemConfig config_;
    int sensor_fd_{-1};
    
    // ML models
    std::unique_ptr<tflite::FlatBufferModel> cnn_model_;
    std::unique_ptr<tflite::Interpreter> cnn_interpreter_;
    std::unique_ptr<tflite::FlatBufferModel> liveness_model_;
    std::unique_ptr<tflite::Interpreter> liveness_interpreter_;
    
    // Crypto
    EVP_CIPHER_CTX* cipher_ctx_{nullptr};
    std::vector<uint8_t> encryption_key_;
    
    std::string status_message_;
};

#endif // FINGERPRINT_SCANNER_H
