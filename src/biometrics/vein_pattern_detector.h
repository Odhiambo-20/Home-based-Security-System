// vein_pattern_detector.h - Comprehensive Hand Vein Pattern Detection
// Uses Near-Infrared (NIR) imaging for subcutaneous vein detection
#ifndef VEIN_PATTERN_DETECTOR_H
#define VEIN_PATTERN_DETECTOR_H

#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <chrono>

// Forward declarations
class NeuralNetwork;

// Vein pattern types
enum class VeinType {
    DORSAL_HAND,      // Back of hand veins
    PALMAR,           // Palm veins
    FINGER,           // Finger veins
    WRIST             // Wrist veins
};

// Individual vein structure
struct VeinPattern {
    std::vector<cv::Point> centerline;    // Vein centerline points
    float thickness;                       // Average vein thickness
    float length;                          // Total vein length
    float tortuosity;                      // Vein curvature measure
    cv::Vec4f main_direction;             // Principal direction
    int bifurcation_count;                // Number of branch points
    std::vector<cv::Point> bifurcations;  // Bifurcation locations
};

// Complete vein pattern data
struct VeinPatternData {
    cv::Mat nir_image;                    // Raw NIR image
    cv::Mat thermal_image;                // Optional thermal image
    cv::Mat vein_mask;                    // Binary vein segmentation
    cv::Mat enhanced_veins;               // Enhanced vein image
    
    std::vector<VeinPattern> vein_patterns;
    std::vector<float> deep_features;     // 256D CNN embedding
    std::vector<float> geometric_features; // Geometric descriptors
    
    VeinType vein_type;
    HandSide hand_side;
    
    // Quality metrics
    float image_quality;
    float vein_clarity;
    float contrast_ratio;
    int detected_vein_count;
    bool is_valid;
    
    // Liveness indicators
    bool blood_flow_detected;
    float temperature_profile;
    
    std::chrono::system_clock::time_point capture_time;
    std::string scan_id;
};

// Vein template for storage
struct VeinTemplate {
    std::string user_id;
    HandSide hand_side;
    VeinType vein_type;
    
    std::vector<float> vein_features;     // Encoded vein patterns
    std::vector<float> topology_features; // Vein network topology
    std::vector<uint8_t> encrypted_data;
    
    std::chrono::system_clock::time_point enrollment_time;
};

// Matching result
struct VeinMatchResult {
    bool is_match;
    float confidence_score;
    float pattern_similarity;
    float topology_similarity;
    float combined_score;
    std::string matched_user_id;
};

class VeinPatternDetector {
public:
    VeinPatternDetector();
    ~VeinPatternDetector();
    
    // Initialization
    bool initialize(const std::string& config_path);
    void shutdown();
    bool isInitialized() const { return initialized_; }
    
    // Vein capture
    VeinPatternData captureVeinPattern(VeinType type = VeinType::DORSAL_HAND);
    bool initializeNIRCamera();
    bool initializeThermalSensor();
    cv::Mat captureNIRImage();
    cv::Mat captureThermalImage();
    
    // Vein enhancement
    cv::Mat enhanceVeinPattern(const cv::Mat& nir_image);
    cv::Mat applyCLAHE(const cv::Mat& image);
    cv::Mat applyGaussianMatching(const cv::Mat& image);
    cv::Mat applyFrangiFilter(const cv::Mat& image);
    cv::Mat applyAdaptiveThreshold(const cv::Mat& image);
    
    // Vein segmentation
    cv::Mat segmentVeins(const cv::Mat& enhanced_image);
    cv::Mat regionGrowing(const cv::Mat& image, const std::vector<cv::Point>& seeds);
    cv::Mat maximumCurvature(const cv::Mat& image);
    std::vector<cv::Point> detectVeinSeeds(const cv::Mat& image);
    
    // Vein pattern extraction
    std::vector<VeinPattern> extractVeinPatterns(const cv::Mat& vein_mask);
    std::vector<cv::Point> extractVeinCenterline(const cv::Mat& vein_region);
    std::vector<cv::Point> detectBifurcationPoints(const cv::Mat& skeleton);
    float computeVeinTortuosity(const std::vector<cv::Point>& centerline);
    float computeVeinThickness(const cv::Mat& vein_region);
    
    // Morphological processing
    cv::Mat skeletonize(const cv::Mat& binary_image);
    cv::Mat pruneShortBranches(const cv::Mat& skeleton, int min_length = 10);
    void thinning(cv::Mat& image);
    
    // Feature extraction
    std::vector<float> extractVeinFeatures(const VeinPatternData& vein_data);
    std::vector<float> extractTopologyFeatures(const std::vector<VeinPattern>& patterns);
    std::vector<float> extractDeepFeatures(const cv::Mat& vein_image);
    std::vector<float> extractGeometricFeatures(const std::vector<VeinPattern>& patterns);
    
    // Liveness detection
    bool detectLiveness(const VeinPatternData& vein_data);
    bool detectBloodFlow(const std::vector<cv::Mat>& temporal_sequence);
    float analyzeThermalSignature(const cv::Mat& thermal_image);
    bool detectPulsation(const std::vector<cv::Mat>& nir_sequence);
    
    // Quality assessment
    float assessVeinQuality(const VeinPatternData& vein_data);
    float computeVeinContrast(const cv::Mat& nir_image);
    float computeVeinClarity(const cv::Mat& vein_mask);
    bool validateVeinPattern(const VeinPatternData& vein_data);
    
    // Template operations
    VeinTemplate createTemplate(const VeinPatternData& vein_data,
                               const std::string& user_id);
    bool saveTemplate(const VeinTemplate& template_data,
                     const std::string& filepath);
    VeinTemplate loadTemplate(const std::string& filepath);
    
    // Matching operations
    VeinMatchResult matchVeinPatterns(const VeinPatternData& probe,
                                      const VeinTemplate& gallery);
    float compareVeinTopology(const std::vector<VeinPattern>& veins1,
                             const std::vector<VeinPattern>& veins2);
    float compareCenterlines(const std::vector<cv::Point>& line1,
                            const std::vector<cv::Point>& line2);
    float compareBifurcations(const std::vector<cv::Point>& bif1,
                             const std::vector<cv::Point>& bif2);
    
    // Visualization
    void visualizeVeinPatterns(cv::Mat& image, 
                              const std::vector<VeinPattern>& patterns);
    void visualizeBifurcations(cv::Mat& image,
                              const std::vector<VeinPattern>& patterns);
    cv::Mat createVeinOverlay(const cv::Mat& original, 
                             const cv::Mat& vein_mask);
    
    // Utilities
    std::string getStatusMessage() const { return status_message_; }
    void setMatchingThreshold(float threshold) { matching_threshold_ = threshold; }
    void setQualityThreshold(float threshold) { quality_threshold_ = threshold; }
    
private:
    // Hardware interface
    int nir_camera_fd_;
    int thermal_sensor_fd_;
    void* nir_camera_handle_;
    void* thermal_handle_;
    
    // Image processing helpers
    cv::Mat preprocessNIRImage(const cv::Mat& nir_image);
    cv::Mat normalizeImage(const cv::Mat& image);
    cv::Mat removeBackground(const cv::Mat& image);
    
    // Frangi filter implementation (vesselness filter)
    cv::Mat computeFrangiVesselness(const cv::Mat& image, 
                                    const std::vector<double>& scales);
    cv::Mat computeHessianEigenvalues(const cv::Mat& image, double sigma);
    
    // Vein tracking
    void traceVein(const cv::Mat& vein_mask, cv::Point start,
                   std::vector<cv::Point>& centerline);
    
    // Network topology analysis
    struct VeinNode {
        cv::Point position;
        std::vector<int> connected_nodes;
        bool is_bifurcation;
    };
    
    std::vector<VeinNode> buildVeinGraph(const std::vector<VeinPattern>& patterns);
    std::vector<float> computeGraphFeatures(const std::vector<VeinNode>& graph);
    
    // Neural network inference
    bool loadVeinCNN(const std::string& model_path);
    std::vector<float> runVeinInference(const cv::Mat& vein_image);
    
    // Security
    std::vector<uint8_t> encryptTemplate(const VeinTemplate& template_data);
    VeinTemplate decryptTemplate(const std::vector<uint8_t>& encrypted);
    
    // Member variables
    std::atomic<bool> initialized_;
    std::mutex capture_mutex_;
    
    // AI/ML components
    std::unique_ptr<NeuralNetwork> vein_cnn_;
    
    // Configuration
    float matching_threshold_;        // Default: 0.90
    float quality_threshold_;         // Default: 70.0
    int nir_wavelength_;             // 850nm typical
    int image_width_;                // 640 pixels
    int image_height_;               // 480 pixels
    
    // Processing parameters
    struct ProcessingParams {
        // CLAHE parameters
        double clahe_clip_limit;
        int clahe_tile_size;
        
        // Frangi filter parameters
        std::vector<double> frangi_scales;
        double frangi_alpha;
        double frangi_beta;
        double frangi_c;
        
        // Segmentation parameters
        double adaptive_threshold_block_size;
        double adaptive_threshold_c;
        
        // Morphology parameters
        int morphology_kernel_size;
        int min_vein_length;
        int min_vein_thickness;
    } params_;
    
    // Status
    std::string status_message_;
    
    // Performance metrics
    struct PerformanceMetrics {
        float avg_capture_time_ms;
        float avg_processing_time_ms;
        int total_captures;
        int successful_captures;
    } metrics_;
    
    // Calibration
    cv::Mat nir_calibration_matrix_;
    cv::Mat thermal_calibration_matrix_;
};

// Helper functions
inline float hausdorffDistance(const std::vector<cv::Point>& set1,
                              const std::vector<cv::Point>& set2) {
    float max_dist = 0;
    for (const auto& p1 : set1) {
        float min_dist = std::numeric_limits<float>::max();
        for (const auto& p2 : set2) {
            float dist = cv::norm(p1 - p2);
            min_dist = std::min(min_dist, dist);
        }
        max_dist = std::max(max_dist, min_dist);
    }
    return max_dist;
}

inline std::string veinTypeToString(VeinType type) {
    switch(type) {
        case VeinType::DORSAL_HAND: return "Dorsal Hand";
        case VeinType::PALMAR: return "Palmar";
        case VeinType::FINGER: return "Finger";
        case VeinType::WRIST: return "Wrist";
        default: return "Unknown";
    }
}

#endif // VEIN_PATTERN_DETECTOR_H
