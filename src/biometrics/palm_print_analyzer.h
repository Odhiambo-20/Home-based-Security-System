// palm_print_analyzer.h - Industrial Production Grade Palm Print Analysis
// Complete implementation for whole hand biometric authentication
// Copyright (c) 2025 Biometric Security Systems
// Version: 2.0.0 - Production Release

#ifndef PALM_PRINT_ANALYZER_H
#define PALM_PRINT_ANALYZER_H

#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <atomic>
#include <chrono>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Dense>
#include <thread>
#include <condition_variable>
#include <queue>

// Forward declarations
class NeuralNetwork;
class EncryptionEngine;
class Logger;

// Configuration constants
namespace PalmConfig {
    constexpr int TARGET_PALM_WIDTH = 224;
    constexpr int TARGET_PALM_HEIGHT = 224;
    constexpr int DEEP_FEATURE_DIM = 512;
    constexpr int GABOR_FEATURE_DIM = 128;
    constexpr int LBP_FEATURE_DIM = 256;
    constexpr int GEOMETRIC_FEATURE_DIM = 64;
    constexpr float DEFAULT_MATCH_THRESHOLD = 0.78f;
    constexpr float DEFAULT_QUALITY_THRESHOLD = 65.0f;
    constexpr int MAX_PALM_LINES = 50;
    constexpr int MIN_PALM_AREA = 8000;
    constexpr int GABOR_SCALES = 5;
    constexpr int GABOR_ORIENTATIONS = 8;
    constexpr float MATCH_TIMEOUT_MS = 500.0f;
}

// Hand side enumeration
enum class HandSide {
    LEFT,
    RIGHT,
    UNKNOWN
};

// Palm regions for detailed analysis
enum class PalmRegion {
    THENAR,          // Thumb base (mount of Venus)
    HYPOTHENAR,      // Pinky side
    INTERDIGITAL_1,  // Between thumb and index
    INTERDIGITAL_2,  // Between index and middle
    INTERDIGITAL_3,  // Between middle and ring
    INTERDIGITAL_4,  // Between ring and pinky
    CENTRAL,         // Center of palm
    WRIST,           // Wrist area
    FULL_PALM        // Entire palm
};

// Palm line types
enum class PalmLineType {
    HEART_LINE,
    HEAD_LINE,
    LIFE_LINE,
    FATE_LINE,
    SUN_LINE,
    MERCURY_LINE,
    FLEXION_CREASES,
    SECONDARY_CREASES,
    MINOR_LINES,
    UNKNOWN
};

// Line detection quality
enum class LineQuality {
    EXCELLENT,
    GOOD,
    FAIR,
    POOR,
    INVALID
};

// Individual palm line structure
struct PalmLine {
    PalmLineType type;
    std::vector<cv::Point2f> points;
    std::vector<float> curvature_profile;
    float length;
    float avg_curvature;
    float max_curvature;
    float depth;
    float prominence;
    cv::Vec4f line_params;
    cv::Moments moments;
    LineQuality quality;
    float confidence;
    std::vector<cv::Point2f> branch_points;
    std::vector<cv::Point2f> intersection_points;
    
    PalmLine() : type(PalmLineType::UNKNOWN), length(0.0f), 
                 avg_curvature(0.0f), max_curvature(0.0f), 
                 depth(0.0f), prominence(0.0f), 
                 quality(LineQuality::INVALID), confidence(0.0f) {}
};

// Palm texture features
struct PalmTexture {
    cv::Mat texture_map;
    cv::Mat orientation_field;
    cv::Mat frequency_field;
    std::vector<float> gabor_features;
    std::vector<float> lbp_features;
    std::vector<float> wavelet_features;
    std::vector<float> glcm_features;
    float texture_complexity;
    float ridge_density;
    float ridge_quality;
    float average_frequency;
    float frequency_variance;
    std::vector<cv::Mat> multiscale_textures;
    
    PalmTexture() : texture_complexity(0.0f), ridge_density(0.0f),
                    ridge_quality(0.0f), average_frequency(0.0f),
                    frequency_variance(0.0f) {}
};

// Palm geometric features
struct PalmGeometry {
    cv::Point2f palm_center;
    cv::Point2f centroid;
    float palm_width;
    float palm_height;
    float palm_area;
    float palm_perimeter;
    float aspect_ratio;
    float circularity;
    float solidity;
    float convexity;
    std::vector<cv::Point2f> finger_bases;
    std::vector<cv::Point2f> finger_tips;
    std::vector<cv::Point2f> key_points;
    std::vector<cv::Point2f> anatomical_landmarks;
    cv::RotatedRect bounding_box;
    cv::RotatedRect min_area_rect;
    std::vector<cv::Point> convex_hull;
    std::vector<cv::Vec4i> convexity_defects;
    std::vector<float> geometric_features;
    Eigen::Matrix3f affine_transform;
    std::vector<float> shape_context;
    std::vector<float> curvature_scale_space;
    
    PalmGeometry() : palm_width(0.0f), palm_height(0.0f), palm_area(0.0f),
                     palm_perimeter(0.0f), aspect_ratio(0.0f), 
                     circularity(0.0f), solidity(0.0f), convexity(0.0f) {}
};

// Vascular pattern features (for enhanced security)
struct VascularFeatures {
    cv::Mat vein_map;
    std::vector<cv::Point2f> bifurcation_points;
    std::vector<cv::Point2f> endpoint_points;
    std::vector<std::vector<cv::Point2f>> vein_paths;
    float vascular_density;
    std::vector<float> vein_features;
    
    VascularFeatures() : vascular_density(0.0f) {}
};

// Complete palm print data
struct PalmPrintData {
    // Raw images
    cv::Mat original_image;
    cv::Mat palm_image;
    cv::Mat palm_roi;
    cv::Mat normalized_palm;
    cv::Mat enhanced_palm;
    cv::Mat hand_mask;
    cv::Mat palm_mask;
    
    // Features
    std::vector<PalmLine> palm_lines;
    PalmTexture texture;
    PalmGeometry geometry;
    VascularFeatures vascular;
    std::vector<float> deep_features;
    std::vector<float> fusion_features;
    
    // Quality metrics
    float overall_quality;
    float image_quality;
    float contrast_score;
    float sharpness_score;
    float illumination_score;
    float noise_level;
    float line_clarity;
    float texture_quality;
    bool is_valid;
    bool is_live;
    
    // Metadata
    HandSide hand_side;
    std::chrono::system_clock::time_point capture_time;
    std::string scan_id;
    std::string device_id;
    cv::Size original_size;
    cv::Rect roi_rect;
    float capture_confidence;
    std::unordered_map<std::string, float> extended_metrics;
    
    PalmPrintData() : overall_quality(0.0f), image_quality(0.0f),
                      contrast_score(0.0f), sharpness_score(0.0f),
                      illumination_score(0.0f), noise_level(0.0f),
                      line_clarity(0.0f), texture_quality(0.0f),
                      is_valid(false), is_live(false),
                      hand_side(HandSide::UNKNOWN),
                      capture_confidence(0.0f) {}
};

// Palm print template for storage
struct PalmPrintTemplate {
    std::string template_id;
    std::string user_id;
    HandSide hand_side;
    
    // Multi-level features
    std::vector<float> level1_features;  // Coarse features
    std::vector<float> level2_features;  // Medium features
    std::vector<float> level3_features;  // Fine features
    std::vector<float> texture_features;
    std::vector<float> geometric_features;
    std::vector<float> line_features;
    std::vector<float> vascular_features;
    std::vector<float> fusion_features;
    
    // Template metadata
    std::chrono::system_clock::time_point enrollment_time;
    std::chrono::system_clock::time_point last_update;
    int enrollment_quality;
    int update_count;
    std::vector<uint8_t> encrypted_data;
    std::vector<uint8_t> template_hash;
    
    // Quality and validation
    float template_quality;
    bool is_verified;
    std::string verification_method;
    
    PalmPrintTemplate() : hand_side(HandSide::UNKNOWN),
                          enrollment_quality(0), update_count(0),
                          template_quality(0.0f), is_verified(false) {}
};

// Matching result with detailed scores
struct PalmMatchResult {
    bool is_match;
    float overall_score;
    float confidence;
    
    // Component scores
    float line_similarity;
    float texture_similarity;
    float geometric_similarity;
    float vascular_similarity;
    float deep_feature_similarity;
    float fusion_score;
    
    // Match details
    std::string matched_template_id;
    std::string matched_user_id;
    HandSide matched_hand_side;
    int num_matched_minutiae;
    float match_time_ms;
    
    // Quality indicators
    float probe_quality;
    float gallery_quality;
    float match_reliability;
    
    // Forensic data
    std::vector<std::pair<cv::Point2f, cv::Point2f>> matched_points;
    cv::Mat alignment_transform;
    std::vector<float> score_breakdown;
    
    PalmMatchResult() : is_match(false), overall_score(0.0f), 
                        confidence(0.0f), line_similarity(0.0f),
                        texture_similarity(0.0f), geometric_similarity(0.0f),
                        vascular_similarity(0.0f), deep_feature_similarity(0.0f),
                        fusion_score(0.0f), matched_hand_side(HandSide::UNKNOWN),
                        num_matched_minutiae(0), match_time_ms(0.0f),
                        probe_quality(0.0f), gallery_quality(0.0f),
                        match_reliability(0.0f) {}
};

// Processing statistics
struct ProcessingStats {
    std::atomic<uint64_t> total_captures{0};
    std::atomic<uint64_t> successful_captures{0};
    std::atomic<uint64_t> failed_captures{0};
    std::atomic<uint64_t> total_matches{0};
    std::atomic<uint64_t> successful_matches{0};
    std::atomic<uint64_t> false_rejections{0};
    std::atomic<uint64_t> false_acceptances{0};
    
    std::atomic<double> avg_capture_time_ms{0.0};
    std::atomic<double> avg_matching_time_ms{0.0};
    std::atomic<double> avg_quality_score{0.0};
    
    std::chrono::system_clock::time_point start_time;
    
    void reset() {
        total_captures = 0;
        successful_captures = 0;
        failed_captures = 0;
        total_matches = 0;
        successful_matches = 0;
        false_rejections = 0;
        false_acceptances = 0;
        avg_capture_time_ms = 0.0;
        avg_matching_time_ms = 0.0;
        avg_quality_score = 0.0;
        start_time = std::chrono::system_clock::now();
    }
};

// Main Palm Print Analyzer Class
class PalmPrintAnalyzer {
public:
    PalmPrintAnalyzer();
    ~PalmPrintAnalyzer();
    
    // Initialization and shutdown
    bool initialize(const std::string& config_path);
    bool initializeWithModels(const std::string& palm_model_path,
                              const std::string& liveness_model_path);
    void shutdown();
    bool isInitialized() const { return initialized_.load(); }
    bool warmup();
    
    // Configuration
    bool loadConfiguration(const std::string& config_path);
    bool saveConfiguration(const std::string& config_path);
    void setMatchingThreshold(float threshold);
    void setQualityThreshold(float threshold);
    void setProcessingMode(const std::string& mode);
    void enableParallelProcessing(bool enable);
    void setNumThreads(int num_threads);
    
    // Palm capture and extraction
    PalmPrintData capturePalmPrint(const cv::Mat& full_hand_image);
    PalmPrintData capturePalmPrintAsync(const cv::Mat& full_hand_image);
    cv::Mat extractPalmRegion(const cv::Mat& hand_image);
    cv::Mat extractPalmRegionAdvanced(const cv::Mat& hand_image, 
                                       const cv::Mat& depth_map);
    cv::Mat alignPalm(const cv::Mat& palm_roi);
    cv::Mat alignPalmRobust(const cv::Mat& palm_roi, 
                             const std::vector<cv::Point2f>& landmarks);
    
    // Hand segmentation
    cv::Mat segmentHand(const cv::Mat& image);
    cv::Mat segmentHandDeep(const cv::Mat& image);
    std::vector<cv::Point> findHandContour(const cv::Mat& mask);
    cv::Rect computePalmBoundingBox(const std::vector<cv::Point>& contour);
    HandSide detectHandSide(const cv::Mat& hand_image, 
                            const std::vector<cv::Point>& contour);
    
    // Palm line detection and analysis
    std::vector<PalmLine> detectPalmLines(const cv::Mat& palm_image);
    std::vector<PalmLine> detectPalmLinesAdvanced(const cv::Mat& palm_image);
    PalmLine detectSpecificLine(const cv::Mat& palm_image, PalmLineType type);
    std::vector<cv::Point2f> traceLinePoints(const cv::Mat& edge_map, 
                                               cv::Point2f start_point);
    std::vector<cv::Point2f> traceLineRobust(const cv::Mat& enhanced_image,
                                               cv::Point2f start_point,
                                               cv::Point2f direction);
    void classifyPalmLines(std::vector<PalmLine>& lines, 
                           const PalmGeometry& geometry);
    void refinePalmLines(std::vector<PalmLine>& lines);
    void filterPalmLines(std::vector<PalmLine>& lines);
    float computeLineCurvature(const std::vector<cv::Point2f>& points);
    std::vector<float> computeCurvatureProfile(const std::vector<cv::Point2f>& points);
    
    // Texture analysis
    PalmTexture analyzePalmTexture(const cv::Mat& palm_image);
    PalmTexture analyzePalmTextureMultiscale(const cv::Mat& palm_image);
    std::vector<float> extractGaborFeatures(const cv::Mat& palm_image);
    std::vector<float> extractLBPFeatures(const cv::Mat& palm_image);
    std::vector<float> extractWaveletFeatures(const cv::Mat& palm_image);
    std::vector<float> extractGLCMFeatures(const cv::Mat& palm_image);
    cv::Mat computeTextureMap(const cv::Mat& palm_image);
    cv::Mat computeOrientationField(const cv::Mat& palm_image);
    cv::Mat computeFrequencyField(const cv::Mat& palm_image);
    cv::Mat enhancePalmRidges(const cv::Mat& palm_image);
    cv::Mat enhancePalmRidgesAdaptive(const cv::Mat& palm_image,
                                       const cv::Mat& orientation_field);
    
    // Geometric analysis
    PalmGeometry analyzePalmGeometry(const cv::Mat& palm_image, 
                                      const cv::Mat& hand_mask);
    PalmGeometry analyzePalmGeometryDetailed(const cv::Mat& palm_image,
                                              const cv::Mat& hand_mask,
                                              const std::vector<cv::Point>& contour);
    std::vector<cv::Point2f> detectKeyPoints(const cv::Mat& palm_image);
    std::vector<cv::Point2f> detectAnatomicalLandmarks(const cv::Mat& palm_image,
                                                         const cv::Mat& hand_mask);
    std::vector<cv::Point2f> detectFingerBases(const std::vector<cv::Point>& contour);
    std::vector<cv::Point2f> detectFingerTips(const std::vector<cv::Point>& contour);
    std::vector<float> extractGeometricFeatures(const PalmGeometry& geometry);
    std::vector<float> computeShapeContext(const std::vector<cv::Point>& contour);
    std::vector<float> computeCurvatureScaleSpace(const std::vector<cv::Point>& contour);
    
    // Vascular pattern analysis
    VascularFeatures extractVascularFeatures(const cv::Mat& palm_image);
    cv::Mat enhanceVeins(const cv::Mat& palm_image);
    std::vector<cv::Point2f> detectVeinBifurcations(const cv::Mat& vein_map);
    std::vector<std::vector<cv::Point2f>> traceVeinPaths(const cv::Mat& vein_map);
    
    // Deep learning features
    std::vector<float> extractDeepFeatures(const cv::Mat& palm_image);
    std::vector<float> extractMultiscaleDeepFeatures(const cv::Mat& palm_image);
    std::vector<float> fuseFeatures(const PalmPrintData& palm_data);
    
    // Liveness detection
    bool performLivenessDetection(const cv::Mat& palm_image);
    bool performLivenessDetectionAdvanced(const cv::Mat& palm_image,
                                           const cv::Mat& prev_frame);
    float computeLivenessScore(const cv::Mat& palm_image);
    
    // Quality assessment
    float assessPalmQuality(const PalmPrintData& palm_data);
    float assessPalmQualityDetailed(const PalmPrintData& palm_data,
                                     std::unordered_map<std::string, float>& metrics);
    float computeImageSharpness(const cv::Mat& image);
    float computeImageContrast(const cv::Mat& image);
    float computeIlluminationQuality(const cv::Mat& image);
    float computeNoiseLevel(const cv::Mat& image);
    bool validatePalmROI(const cv::Mat& palm_roi);
    bool validatePalmData(const PalmPrintData& palm_data);
    
    // Template operations
    PalmPrintTemplate createTemplate(const PalmPrintData& palm_data,
                                      const std::string& user_id);
    PalmPrintTemplate createTemplateMultiSample(
        const std::vector<PalmPrintData>& samples,
        const std::string& user_id);
    bool updateTemplate(PalmPrintTemplate& existing_template,
                        const PalmPrintData& new_data);
    bool saveTemplate(const PalmPrintTemplate& template_data,
                      const std::string& filepath);
    PalmPrintTemplate loadTemplate(const std::string& filepath);
    bool exportTemplate(const PalmPrintTemplate& template_data,
                        const std::string& format,
                        const std::string& filepath);
    
    // Matching operations
    PalmMatchResult matchPalmPrints(const PalmPrintData& probe,
                                     const PalmPrintTemplate& gallery);
    PalmMatchResult matchPalmPrintsRobust(const PalmPrintData& probe,
                                           const PalmPrintTemplate& gallery,
                                           int max_attempts = 3);
    std::vector<PalmMatchResult> matchAgainstDatabase(
        const PalmPrintData& probe,
        const std::vector<PalmPrintTemplate>& gallery_templates,
        int top_k = 5);
    float comparePalmLines(const std::vector<PalmLine>& lines1,
                           const std::vector<PalmLine>& lines2);
    float compareTextures(const PalmTexture& texture1,
                          const PalmTexture& texture2);
    float compareGeometry(const PalmGeometry& geom1,
                          const PalmGeometry& geom2);
    float compareVascular(const VascularFeatures& vasc1,
                          const VascularFeatures& vasc2);
    float computeFeatureSimilarity(const std::vector<float>& feat1,
                                    const std::vector<float>& feat2);
    float computeCosineSimilarity(const std::vector<float>& v1,
                                   const std::vector<float>& v2);
    float computeEuclideanDistance(const std::vector<float>& v1,
                                    const std::vector<float>& v2);
    
    // Preprocessing and enhancement
    cv::Mat preprocessPalmImage(const cv::Mat& palm_image);
    cv::Mat preprocessPalmImageAdvanced(const cv::Mat& palm_image,
                                         bool enhance_contrast = true,
                                         bool reduce_noise = true,
                                         bool normalize_illumination = true);
    cv::Mat normalizePalmImage(const cv::Mat& palm_image);
    cv::Mat normalizeIllumination(const cv::Mat& palm_image);
    cv::Mat reduceNoise(const cv::Mat& palm_image);
    cv::Mat enhanceContrast(const cv::Mat& palm_image);
    cv::Mat histogramEqualization(const cv::Mat& palm_image);
    cv::Mat adaptiveHistogramEqualization(const cv::Mat& palm_image);
    
    // Image processing utilities
    cv::Mat applyGaborBank(const cv::Mat& image, 
                            const std::vector<float>& wavelengths,
                            const std::vector<float>& orientations);
    cv::Mat applyGaborFilter(const cv::Mat& image, 
                              float wavelength, 
                              float orientation,
                              float sigma, 
                              float gamma = 0.5f);
    cv::Mat applyLBP(const cv::Mat& image, int radius = 1, int neighbors = 8);
    cv::Mat applyUniformLBP(const cv::Mat& image, int radius, int neighbors);
    cv::Mat computeEdgeMap(const cv::Mat& image);
    cv::Mat computeEdgeMapMultiscale(const cv::Mat& image);
    std::vector<cv::Vec4f> detectLines(const cv::Mat& edge_map);
    std::vector<cv::Vec4f> detectLinesHough(const cv::Mat& edge_map);
    std::vector<cv::Vec4f> detectLinesLSD(const cv::Mat& edge_map);
    
    // Visualization
    void visualizePalmLines(cv::Mat& image, 
                             const std::vector<PalmLine>& lines,
                             bool show_types = true);
    void visualizeKeyPoints(cv::Mat& image,
                             const PalmGeometry& geometry);
    void visualizeTextureMap(const cv::Mat& texture_map,
                              cv::Mat& output);
    void visualizeOrientationField(const cv::Mat& orientation_field,
                                    cv::Mat& output);
    void visualizeMatchResult(cv::Mat& probe_image,
                               cv::Mat& gallery_image,
                               const PalmMatchResult& result);
    cv::Mat createQualityVisualization(const PalmPrintData& palm_data);
    
    // Statistics and monitoring
    ProcessingStats getStatistics() const { return stats_; }
    void resetStatistics() { stats_.reset(); }
    std::string getStatusMessage() const { return status_message_; }
    std::vector<std::string> getErrorLog() const;
    void enableDetailedLogging(bool enable);
    
    // Performance tuning
    void optimizeForSpeed();
    void optimizeForAccuracy();
    void setGPUAcceleration(bool enable);
    bool isGPUAvailable() const;
    
    // Calibration
    bool calibrate(const std::vector<cv::Mat>& calibration_images);
    bool validateCalibration();
    
    // Export and reporting
    std::string generateQualityReport(const PalmPrintData& palm_data);
    std::string generateMatchReport(const PalmMatchResult& result);
    bool exportDiagnostics(const std::string& filepath);

private:
    // Internal initialization
    bool initializeModels();
    bool initializeProcessingPipeline();
    bool loadNeuralNetworks();
    void initializeFilters();
    void initializeThreadPool();
    
    // Advanced palm extraction
    cv::Mat refinePalmMask(const cv::Mat& initial_mask, 
                            const std::vector<cv::Point>& hand_contour);
    std::vector<cv::Vec4i> computeConvexityDefects(
        const std::vector<cv::Point>& contour,
        const std::vector<int>& hull);
    cv::Point2f findPalmCenter(const std::vector<cv::Point>& contour,
                                 const std::vector<cv::Point2f>& finger_bases);
    
    // Line processing helpers
    void mergeNearbyLines(std::vector<PalmLine>& lines, float distance_threshold);
    void removeShortLines(std::vector<PalmLine>& lines, float min_length);
    void smoothLinePoints(std::vector<cv::Point2f>& points, int window_size = 5);
    bool areLinesConnected(const PalmLine& line1, const PalmLine& line2,
                            float threshold);
    std::vector<cv::Point2f> findLineIntersections(const std::vector<PalmLine>& lines);
    
    // Feature extraction helpers
    std::vector<float> computeHOGFeatures(const cv::Mat& image);
    std::vector<float> computeSIFTDescriptor(const cv::Mat& image,
                                              const std::vector<cv::KeyPoint>& keypoints);
    std::vector<float> computeHaralickFeatures(const cv::Mat& image);
    cv::Mat computeGLCM(const cv::Mat& image, int dx, int dy, int levels = 256);
    
    // Neural network inference
    std::vector<float> runPalmInference(const cv::Mat& palm_image);
    std::vector<float> runLivenessInference(const cv::Mat& palm_image);
    cv::Mat preprocessForInference(const cv::Mat& image, cv::Size target_size);
    
    // Matching helpers
    cv::Mat estimateAffineTransform(const std::vector<cv::Point2f>& src_points,
                                     const std::vector<cv::Point2f>& dst_points);
    void alignFeatures(std::vector<float>& features1,
                       std::vector<float>& features2,
                       const cv::Mat& transform);
    float computeMatchConfidence(const PalmMatchResult& result);
    bool verifyMatchGeometry(const PalmGeometry& geom1,
                              const PalmGeometry& geom2,
                              float tolerance);
    
    // Security and encryption
    std::vector<uint8_t> encryptTemplate(const PalmPrintTemplate& template_data);
    PalmPrintTemplate decryptTemplate(const std::vector<uint8_t>& encrypted_data);
    std::vector<uint8_t> computeTemplateHash(const PalmPrintTemplate& template_data);
    bool verifyTemplateIntegrity(const PalmPrintTemplate& template_data);
    
    // Thread pool for parallel processing
    class ThreadPool {
    public:
        ThreadPool(size_t num_threads);
        ~ThreadPool();
        template<typename F, typename... Args>
        auto enqueue(F&& f, Args&&... args) 
            -> std::future<typename std::result_of<F(Args...)>::type>;
    private:
        std::vector<std::thread> workers_;
        std::queue<std::function<void()>> tasks_;
        std::mutex queue_mutex_;
        std::condition_variable condition_;
        bool stop_;
    };
    
    // Member variables
    std::atomic<bool> initialized_;
    std::atomic<bool> calibrated_;
    mutable std::mutex processing_mutex_;
    mutable std::mutex stats_mutex_;
    
    // Neural networks
    std::unique_ptr<NeuralNetwork> palm_recognition_net_;
    std::unique_ptr<NeuralNetwork> liveness_detection_net_;
    std::unique_ptr<NeuralNetwork> segmentation_net_;
    
    // Encryption
    std::unique_ptr<EncryptionEngine> encryption_engine_;
    
    // Thread pool
    std::unique_ptr<ThreadPool> thread_pool_;
    
    // Configuration
    float matching_threshold_;
    float quality_threshold_;
    int target_palm_width_;
    int target_palm_height_;
    bool parallel_processing_enabled_;
    bool gpu_acceleration_enabled_;
    bool detailed_logging_enabled_;
    int num_processing_threads_;
    std::string processing_mode_;
    
    // Processing parameters
    struct ProcessingParams {
        // Gabor parameters
        std::vector<float> gabor_wavelengths;
        std::vector<float> gabor_orientations;
        float gabor_sigma;
        float gabor_gamma;
        int gabor_kernel_size;
        
        // LBP parameters
        int lbp_radius;
        int lbp_neighbors;
        bool lbp_uniform;
        
        // Edge detection
        double canny_low_threshold;
        double canny_high_threshold;
        int canny_aperture_size;
        
        // Line detection
        double hough_rho;
        double hough_theta;
        int hough_threshold;
        int hough_min_line_length;
        int hough_max_line_gap;
        
        // Quality thresholds
        float min_contrast;
        float min_sharpness;
        float min_illumination;
        float max_noise_level;
        
        // Preprocessing
        bool auto_contrast;
        bool auto_illumination;
        bool noise_reduction;
        int bilateral_filter_d;
        double bilateral_filter_sigma_color;
        double bilateral_filter_sigma_space;
        
        // Template matching
        float line_weight;
        float texture_weight;
        float geometry_weight;
        float vascular_weight;
        float deep_feature_weight;
        
        ProcessingParams() :
            gabor_sigma(4.0f), gabor_gamma(0.5f), gabor_kernel_size(31),
            lbp_radius(2), lbp_neighbors(16), lbp_uniform(true),
            canny_low_threshold(50.0), canny_high_threshold(150.0), canny_aperture_size(3),
            hough_rho(1.0), hough_theta(CV_PI/180.0), hough_threshold(80),
            hough_min_line_length(30), hough_max_line_gap(10),
            min_contrast(0.3f), min_sharpness(0.4f), min_illumination(0.5f), max_noise_level(0.3f),
            auto_contrast(true), auto_illumination(true), noise_reduction(true),
            bilateral_filter_d(9), bilateral_filter_sigma_color(75.0), bilateral_filter_sigma_space(75.0),
            line_weight(0.25f), texture_weight(0.25f), geometry_weight(0.20f),
            vascular_weight(0.15f), deep_feature_weight(0.15f) {
            
            // Initialize Gabor parameters
            for (int i = 0; i < PalmConfig::GABOR_SCALES; i++) {
                gabor_wavelengths.push_back(4.0f * std::pow(2.0f, i));
            }
            for (int i = 0; i < PalmConfig::GABOR_ORIENTATIONS; i++) {
                gabor_orientations.push_back(i * M_PI / PalmConfig::GABOR_ORIENTATIONS);
            }
        }
    } params_;
    
    // Pre-computed filters
    std::vector<cv::Mat> gabor_filters_;
    cv::Mat gaussian_kernel_;
    cv::Mat bilateral_kernel_;
    
    // Statistics
    ProcessingStats stats_;
    std::string status_message_;
    std::vector<std::string> error_log_;
    std::chrono::system_clock::time_point last_capture_time_;
    
    // Performance metrics
    struct PerformanceMetrics {
        double preprocessing_time_ms;
        double extraction_time_ms;
        double feature_time_ms;
        double matching_time_ms;
        double total_time_ms;
        
        PerformanceMetrics() : preprocessing_time_ms(0), extraction_time_ms(0),
                               feature_time_ms(0), matching_time_ms(0), total_time_ms(0) {}
    };
    
    mutable PerformanceMetrics last_metrics_;
    
    // Cache for optimization
    struct FeatureCache {
        std::unordered_map<std::string, std::vector<float>> cached_features;
        std::unordered_map<std::string, cv::Mat> cached_images;
        std::mutex cache_mutex;
        size_t max_cache_size;
        
        FeatureCache() : max_cache_size(100) {}
        
        void clear() {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cached_features.clear();
            cached_images.clear();
        }
    };
    
    mutable FeatureCache feature_cache_;
};

// Inline utility functions
inline float cosineSimilarity(const std::vector<float>& v1, 
                              const std::vector<float>& v2) {
    if (v1.size() != v2.size() || v1.empty()) return 0.0f;
    
    float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
    for (size_t i = 0; i < v1.size(); i++) {
        dot += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    
    if (norm1 == 0.0f || norm2 == 0.0f) return 0.0f;
    return dot / (std::sqrt(norm1) * std::sqrt(norm2));
}

inline float euclideanDistance(const std::vector<float>& v1,
                                const std::vector<float>& v2) {
    if (v1.size() != v2.size() || v1.empty()) return std::numeric_limits<float>::max();
    
    float sum = 0.0f;
    for (size_t i = 0; i < v1.size(); i++) {
        float diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

inline std::string handSideToString(HandSide side) {
    switch(side) {
        case HandSide::LEFT: return "Left";
        case HandSide::RIGHT: return "Right";
        case HandSide::UNKNOWN: return "Unknown";
        default: return "Invalid";
    }
}

inline std::string palmRegionToString(PalmRegion region) {
    switch(region) {
        case PalmRegion::THENAR: return "Thenar";
        case PalmRegion::HYPOTHENAR: return "Hypothenar";
        case PalmRegion::INTERDIGITAL_1: return "Interdigital_1";
        case PalmRegion::INTERDIGITAL_2: return "Interdigital_2";
        case PalmRegion::INTERDIGITAL_3: return "Interdigital_3";
        case PalmRegion::INTERDIGITAL_4: return "Interdigital_4";
        case PalmRegion::CENTRAL: return "Central";
        case PalmRegion::WRIST: return "Wrist";
        case PalmRegion::FULL_PALM: return "Full Palm";
        default: return "Unknown";
    }
}

inline std::string palmLineTypeToString(PalmLineType type) {
    switch(type) {
        case PalmLineType::HEART_LINE: return "Heart Line";
        case PalmLineType::HEAD_LINE: return "Head Line";
        case PalmLineType::LIFE_LINE: return "Life Line";
        case PalmLineType::FATE_LINE: return "Fate Line";
        case PalmLineType::SUN_LINE: return "Sun Line";
        case PalmLineType::MERCURY_LINE: return "Mercury Line";
        case PalmLineType::FLEXION_CREASES: return "Flexion Creases";
        case PalmLineType::SECONDARY_CREASES: return "Secondary Creases";
        case PalmLineType::MINOR_LINES: return "Minor Lines";
        case PalmLineType::UNKNOWN: return "Unknown";
        default: return "Invalid";
    }
}

inline std::string lineQualityToString(LineQuality quality) {
    switch(quality) {
        case LineQuality::EXCELLENT: return "Excellent";
        case LineQuality::GOOD: return "Good";
        case LineQuality::FAIR: return "Fair";
        case LineQuality::POOR: return "Poor";
        case LineQuality::INVALID: return "Invalid";
        default: return "Unknown";
    }
}

// Template for ThreadPool enqueue
template<typename F, typename... Args>
auto PalmPrintAnalyzer::ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;
    
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (stop_) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }
        tasks_.emplace([task](){ (*task)(); });
    }
    condition_.notify_one();
    return res;
}

#endif // PALM_PRINT_ANALYZER_H
