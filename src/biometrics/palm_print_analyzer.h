// palm_print_analyzer.h - Comprehensive Palm Print Analysis for Whole Hand
#ifndef PALM_PRINT_ANALYZER_H
#define PALM_PRINT_ANALYZER_H

#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <chrono>

// Forward declarations
class NeuralNetwork;

// Palm regions for detailed analysis
enum class PalmRegion {
    THENAR,          // Thumb base (mount of Venus)
    HYPOTHENAR,      // Pinky side
    INTERDIGITAL,    // Between fingers
    CENTRAL,         // Center of palm
    FULL_PALM        // Entire palm
};

// Palm line types (palmistry-inspired but for biometric patterns)
enum class PalmLineType {
    HEART_LINE,
    HEAD_LINE,
    LIFE_LINE,
    FATE_LINE,
    MINOR_LINES,
    FLEXION_CREASES
};

// Individual palm line structure
struct PalmLine {
    PalmLineType type;
    std::vector<cv::Point> points;
    float length;
    float curvature;
    float depth;              // Line prominence
    cv::Vec4f line_params;    // Line equation parameters
};

// Palm texture features
struct PalmTexture {
    cv::Mat texture_map;             // Texture intensity map
    std::vector<float> gabor_features;   // Gabor filter responses
    std::vector<float> lbp_features;     // Local Binary Patterns
    float texture_complexity;         // Overall complexity measure
    float ridge_density;              // Ridge concentration
};

// Palm geometric features
struct PalmGeometry {
    cv::Point2f palm_center;
    float palm_width;
    float palm_height;
    float palm_area;
    std::vector<cv::Point2f> finger_bases;    // Where fingers meet palm
    std::vector<cv::Point2f> key_points;      // Anatomical landmarks
    cv::RotatedRect bounding_box;
    std::vector<float> geometric_features;    // 50D geometric descriptor
};

// Complete palm print data
struct PalmPrintData {
    cv::Mat palm_image;                  // Raw palm image (RGB)
    cv::Mat palm_roi;                    // Extracted palm region
    cv::Mat normalized_palm;             // Normalized and aligned
    
    // Features
    std::vector<PalmLine> palm_lines;
    PalmTexture texture;
    PalmGeometry geometry;
    std::vector<float> deep_features;    // 256D CNN embedding
    
    // Quality metrics
    float image_quality;                 // 0-100
    float contrast_score;
    float sharpness_score;
    bool is_valid;
    
    // Metadata
    HandSide hand_side;
    std::chrono::system_clock::time_point capture_time;
    std::string scan_id;
};

// Palm print template for storage
struct PalmPrintTemplate {
    std::string user_id;
    HandSide hand_side;
    
    // Multi-scale features
    std::vector<float> coarse_features;  // Low-res features
    std::vector<float> fine_features;    // High-res features
    std::vector<float> texture_features;
    std::vector<float> geometric_features;
    
    // Encrypted storage
    std::vector<uint8_t> encrypted_data;
    std::chrono::system_clock::time_point enrollment_time;
};

// Matching result
struct PalmMatchResult {
    bool is_match;
    float confidence_score;
    float line_similarity;
    float texture_similarity;
    float geometric_similarity;
    float combined_score;
    std::string matched_user_id;
};

class PalmPrintAnalyzer {
public:
    PalmPrintAnalyzer();
    ~PalmPrintAnalyzer();
    
    // Initialization
    bool initialize(const std::string& config_path);
    void shutdown();
    bool isInitialized() const { return initialized_; }
    
    // Palm capture and extraction
    PalmPrintData capturePalmPrint(const cv::Mat& full_hand_image);
    cv::Mat extractPalmRegion(const cv::Mat& hand_image);
    cv::Mat alignPalm(const cv::Mat& palm_roi);
    
    // Palm line detection
    std::vector<PalmLine> detectPalmLines(const cv::Mat& palm_image);
    PalmLine detectLifeLine(const cv::Mat& palm_image);
    PalmLine detectHeartLine(const cv::Mat& palm_image);
    PalmLine detectHeadLine(const cv::Mat& palm_image);
    std::vector<cv::Point> traceLinePoints(const cv::Mat& edge_map, 
                                           cv::Point start_point);
    
    // Texture analysis
    PalmTexture analyzePalmTexture(const cv::Mat& palm_image);
    std::vector<float> extractGaborFeatures(const cv::Mat& palm_image);
    std::vector<float> extractLBPFeatures(const cv::Mat& palm_image);
    cv::Mat computeTextureMap(const cv::Mat& palm_image);
    
    // Geometric analysis
    PalmGeometry analyzePalmGeometry(const cv::Mat& palm_image, 
                                     const cv::Mat& hand_mask);
    std::vector<cv::Point2f> detectKeyPoints(const cv::Mat& palm_image);
    std::vector<cv::Point2f> detectFingerBases(const cv::Mat& hand_contour);
    std::vector<float> extractGeometricFeatures(const PalmGeometry& geometry);
    
    // Deep learning features
    std::vector<float> extractDeepFeatures(const cv::Mat& palm_image);
    
    // Quality assessment
    float assessPalmQuality(const PalmPrintData& palm_data);
    float computeImageSharpness(const cv::Mat& image);
    float computeImageContrast(const cv::Mat& image);
    bool validatePalmROI(const cv::Mat& palm_roi);
    
    // Template operations
    PalmPrintTemplate createTemplate(const PalmPrintData& palm_data,
                                     const std::string& user_id);
    bool saveTemplate(const PalmPrintTemplate& template_data,
                     const std::string& filepath);
    PalmPrintTemplate loadTemplate(const std::string& filepath);
    
    // Matching operations
    PalmMatchResult matchPalmPrints(const PalmPrintData& probe,
                                    const PalmPrintTemplate& gallery);
    float comparePalmLines(const std::vector<PalmLine>& lines1,
                          const std::vector<PalmLine>& lines2);
    float compareTextures(const PalmTexture& texture1,
                         const PalmTexture& texture2);
    float compareGeometry(const PalmGeometry& geom1,
                         const PalmGeometry& geom2);
    float computeFeatureSimilarity(const std::vector<float>& feat1,
                                   const std::vector<float>& feat2);
    
    // Visualization
    void visualizePalmLines(cv::Mat& image, 
                           const std::vector<PalmLine>& lines);
    void visualizeKeyPoints(cv::Mat& image,
                           const PalmGeometry& geometry);
    void visualizeTextureMap(const cv::Mat& texture_map,
                            cv::Mat& output);
    
    // Preprocessing
    cv::Mat preprocessPalmImage(const cv::Mat& palm_image);
    cv::Mat enhancePalmRidges(const cv::Mat& palm_image);
    cv::Mat normalizePalmImage(const cv::Mat& palm_image);
    
    // Utilities
    std::string getStatusMessage() const { return status_message_; }
    void setMatchingThreshold(float threshold) { matching_threshold_ = threshold; }
    void setQualityThreshold(float threshold) { quality_threshold_ = threshold; }

private:
    // Hand segmentation helpers
    cv::Mat segmentHand(const cv::Mat& image);
    std::vector<cv::Point> findHandContour(const cv::Mat& image);
    cv::Rect computePalmBoundingBox(const std::vector<cv::Point>& hand_contour);
    
    // Image processing
    cv::Mat applyGaborBank(const cv::Mat& image, 
                          const std::vector<float>& wavelengths,
                          const std::vector<float>& orientations);
    cv::Mat computeOrientationField(const cv::Mat& image);
    cv::Mat applyLBP(const cv::Mat& image, int radius = 1, int neighbors = 8);
    
    // Line detection helpers
    cv::Mat computeEdgeMap(const cv::Mat& image);
    std::vector<cv::Vec4f> detectLines(const cv::Mat& edge_map);
    void filterPalmLines(std::vector<PalmLine>& lines);
    float computeLineCurvature(const std::vector<cv::Point>& line_points);
    
    // Feature processing
    std::vector<float> normalizeFeatures(const std::vector<float>& features);
    void fuseMultiscaleFeatures(std::vector<float>& coarse,
                               std::vector<float>& fine);
    
    // Neural network inference
    bool loadPalmCNN(const std::string& model_path);
    std::vector<float> runPalmInference(const cv::Mat& palm_image);
    
    // Security
    std::vector<uint8_t> encryptTemplate(const PalmPrintTemplate& template_data);
    PalmPrintTemplate decryptTemplate(const std::vector<uint8_t>& encrypted);
    
    // Member variables
    std::atomic<bool> initialized_;
    std::mutex analysis_mutex_;
    
    // AI/ML components
    std::unique_ptr<NeuralNetwork> palm_cnn_;
    
    // Configuration
    float matching_threshold_;        // Default: 0.80
    float quality_threshold_;         // Default: 60.0
    int target_palm_width_;          // Default: 224 pixels
    int target_palm_height_;         // Default: 224 pixels
    
    // Processing parameters
    struct ProcessingParams {
        // Gabor filter parameters
        std::vector<float> gabor_wavelengths;
        std::vector<float> gabor_orientations;
        float gabor_sigma;
        
        // LBP parameters
        int lbp_radius;
        int lbp_neighbors;
        
        // Edge detection
        double canny_low_thresh;
        double canny_high_thresh;
        
        // Line detection
        double hough_rho;
        double hough_theta;
        int hough_threshold;
    } params_;
    
    // Status
    std::string status_message_;
    
    // Performance metrics
    struct PerformanceMetrics {
        float avg_extraction_time_ms;
        float avg_matching_time_ms;
        int total_captures;
        int successful_captures;
    } metrics_;
};

// Helper functions
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

inline std::string palmRegionToString(PalmRegion region) {
    switch(region) {
        case PalmRegion::THENAR: return "Thenar";
        case PalmRegion::HYPOTHENAR: return "Hypothenar";
        case PalmRegion::INTERDIGITAL: return "Interdigital";
        case PalmRegion::CENTRAL: return "Central";
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
        case PalmLineType::MINOR_LINES: return "Minor Lines";
        case PalmLineType::FLEXION_CREASES: return "Flexion Creases";
        default: return "Unknown";
    }
}

#endif // PALM_PRINT_ANALYZER_H
