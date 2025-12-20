// multimodal_fusion.h - Multi-Modal Biometric Fusion for Whole Hand
// Combines fingerprint, palm print, and vein pattern for maximum security
#ifndef MULTIMODAL_FUSION_H
#define MULTIMODAL_FUSION_H

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <opencv2/opencv.hpp>

#include "fingerprint_scanner.h"
#include "palm_print_analyzer.h"
#include "vein_pattern_detector.h"

// Biometric modality weights
struct ModalityWeights {
    float fingerprint_weight;     // 0-1
    float palm_print_weight;      // 0-1
    float vein_pattern_weight;    // 0-1
    
    void normalize() {
        float sum = fingerprint_weight + palm_print_weight + vein_pattern_weight;
        if (sum > 0) {
            fingerprint_weight /= sum;
            palm_print_weight /= sum;
            vein_pattern_weight /= sum;
        }
    }
};

// Individual modality result
struct ModalityResult {
    bool available;              // Modality data was captured
    bool valid;                  // Quality passed threshold
    float quality_score;         // Quality metric 0-100
    float match_score;           // Matching score 0-1
    float confidence;            // Confidence in this modality
    std::chrono::milliseconds processing_time;
};

// Combined authentication result
struct MultimodalAuthResult {
    bool authenticated;
    float fusion_score;          // Final fused score 0-1
    float confidence;            // Overall confidence
    
    // Individual modality results
    std::map<std::string, ModalityResult> modality_results;
    
    // Performance metrics
    std::chrono::milliseconds total_time;
    
    // Decision details
    std::string decision_method;  // "score_fusion", "decision_fusion", etc.
    std::vector<std::string> active_modalities;
    
    // Security indicators
    bool liveness_verified;
    float spoofing_probability;
    
    // Matched identity
    std::string matched_user_id;
    float false_accept_rate;     // Estimated FAR at threshold
    float false_reject_rate;     // Estimated FRR at threshold
};

// Complete multi-modal template
struct MultimodalTemplate {
    std::string user_id;
    HandSide hand_side;
    
    // Individual modality templates
    std::vector<FingerprintTemplate> fingerprint_templates;  // All 5 fingers
    PalmPrintTemplate palm_template;
    VeinTemplate vein_template;
    
    // Fusion-level features
    std::vector<float> fused_features;
    
    // Quality indicators
    std::map<std::string, float> modality_qualities;
    
    // Metadata
    std::chrono::system_clock::time_point enrollment_time;
    int update_count;
    std::vector<uint8_t> encrypted_data;
};

// Fusion strategy
enum class FusionStrategy {
    WEIGHTED_SUM,        // Weighted average of scores
    PRODUCT_RULE,        // Product of probabilities
    MAX_RULE,            // Maximum score
    MIN_RULE,            // Minimum score (most conservative)
    ADAPTIVE,            // Quality-based adaptive weighting
    MACHINE_LEARNING,    // ML-based fusion (SVM, Random Forest)
    DEEP_FUSION          // Deep learning-based fusion
};

class MultimodalFusion {
public:
    MultimodalFusion();
    ~MultimodalFusion();
    
    // Initialization
    bool initialize(const std::string& config_path);
    void shutdown();
    bool isInitialized() const { return initialized_; }
    
    // Set fusion components
    void setFingerprintScanner(std::shared_ptr<FingerprintScanner> scanner);
    void setPalmAnalyzer(std::shared_ptr<PalmPrintAnalyzer> analyzer);
    void setVeinDetector(std::shared_ptr<VeinPatternDetector> detector);
    
    // Complete hand capture
    bool captureCompleteHand(HandScanData& fingerprint_data,
                            PalmPrintData& palm_data,
                            VeinPatternData& vein_data);
    
    // Enrollment
    MultimodalTemplate enrollUser(const std::string& user_id,
                                   const HandScanData& fingerprint_data,
                                   const PalmPrintData& palm_data,
                                   const VeinPatternData& vein_data);
    
    bool saveTemplate(const MultimodalTemplate& template_data,
                     const std::string& filepath);
    MultimodalTemplate loadTemplate(const std::string& filepath);
    
    // Authentication
    MultimodalAuthResult authenticateUser(
        const HandScanData& probe_fingerprint,
        const PalmPrintData& probe_palm,
        const VeinPatternData& probe_vein,
        const MultimodalTemplate& gallery_template);
    
    // Score-level fusion
    float fuseScores(const std::map<std::string, float>& modality_scores,
                    const ModalityWeights& weights);
    
    float weightedSumFusion(const std::map<std::string, float>& scores,
                           const ModalityWeights& weights);
    
    float productRuleFusion(const std::map<std::string, float>& scores);
    
    float adaptiveFusion(const std::map<std::string, float>& scores,
                        const std::map<std::string, float>& qualities);
    
    float mlBasedFusion(const std::map<std::string, std::vector<float>>& features);
    
    // Feature-level fusion
    std::vector<float> fuseFeatures(
        const std::vector<float>& fingerprint_features,
        const std::vector<float>& palm_features,
        const std::vector<float>& vein_features);
    
    std::vector<float> concatenateFeatures(
        const std::vector<float>& feat1,
        const std::vector<float>& feat2,
        const std::vector<float>& feat3);
    
    std::vector<float> pcaFusion(
        const std::vector<std::vector<float>>& feature_sets);
    
    // Decision-level fusion
    bool decisionFusion(const std::vector<bool>& individual_decisions,
                       const std::vector<float>& confidences);
    
    bool majorityVoting(const std::vector<bool>& decisions);
    
    bool weightedVoting(const std::vector<bool>& decisions,
                       const std::vector<float>& weights);
    
    // Quality-based weight adaptation
    ModalityWeights computeAdaptiveWeights(
        const std::map<std::string, float>& quality_scores);
    
    void updateWeightsFromQuality(ModalityWeights& weights,
                                  float fingerprint_quality,
                                  float palm_quality,
                                  float vein_quality);
    
    // Liveness verification across modalities
    bool verifyMultimodalLiveness(const HandScanData& fingerprint,
                                  const PalmPrintData& palm,
                                  const VeinPatternData& vein);
    
    bool crossModalityLivenessCheck(const cv::Mat& visible_image,
                                    const cv::Mat& nir_image,
                                    const cv::Mat& thermal_image);
    
    // Security analysis
    float estimateSpoofingProbability(const MultimodalAuthResult& result);
    
    bool detectPresentationAttack(const HandScanData& fingerprint,
                                  const PalmPrintData& palm,
                                  const VeinPatternData& vein);
    
    // Performance optimization
    void enableParallelProcessing(bool enable);
    void setFusionStrategy(FusionStrategy strategy);
    void setModalityWeights(const ModalityWeights& weights);
    
    // Configuration
    void setAuthenticationThreshold(float threshold) { 
        authentication_threshold_ = threshold; 
    }
    
    void setRequireAllModalities(bool require) { 
        require_all_modalities_ = require; 
    }
    
    void setMinimumModalities(int min_count) { 
        minimum_modalities_ = min_count; 
    }
    
    // Analytics
    std::map<std::string, float> getModalityContributions(
        const MultimodalAuthResult& result);
    
    std::string generateAuthenticationReport(const MultimodalAuthResult& result);
    
    // Utilities
    std::string getStatusMessage() const { return status_message_; }

private:
    // Component interfaces
    std::shared_ptr<FingerprintScanner> fingerprint_scanner_;
    std::shared_ptr<PalmPrintAnalyzer> palm_analyzer_;
    std::shared_ptr<VeinPatternDetector> vein_detector_;
    
    // Fusion models
    struct FusionModel {
        cv::Ptr<cv::ml::SVM> svm_classifier;
        cv::Ptr<cv::ml::RTrees> random_forest;
        cv::PCA pca_model;
        std::vector<float> fusion_weights;
    } fusion_model_;
    
    // Score normalization
    float normalizeScore(float raw_score, const std::string& modality);
    
    struct ScoreStatistics {
        float mean;
        float stddev;
        float min_val;
        float max_val;
    };
    
    std::map<std::string, ScoreStatistics> score_stats_;
    
    // Quality-aware processing
    bool shouldUseModality(float quality_score, const std::string& modality);
    
    float computeModalityReliability(float quality_score, 
                                    float historical_accuracy);
    
    // Cross-modality consistency check
    bool checkCrossModalityConsistency(
        const HandScanData& fingerprint,
        const PalmPrintData& palm,
        const VeinPatternData& vein);
    
    float computeGeometricConsistency(
        const HandScanData& fingerprint,
        const PalmPrintData& palm);
    
    // Template management
    void updateTemplate(MultimodalTemplate& template_data,
                       const HandScanData& fingerprint,
                       const PalmPrintData& palm,
                       const VeinPatternData& vein);
    
    // Security
    std::vector<uint8_t> encryptMultimodalTemplate(
        const MultimodalTemplate& template_data);
    
    MultimodalTemplate decryptMultimodalTemplate(
        const std::vector<uint8_t>& encrypted);
    
    // Machine learning helpers
    bool trainFusionClassifier(
        const std::vector<std::map<std::string, float>>& training_scores,
        const std::vector<bool>& labels);
    
    float predictWithML(const std::map<std::string, float>& scores);
    
    // Member variables
    std::atomic<bool> initialized_;
    std::mutex fusion_mutex_;
    
    // Configuration
    FusionStrategy fusion_strategy_;
    ModalityWeights default_weights_;
    bool adaptive_weighting_;
    bool parallel_processing_;
    bool require_all_modalities_;
    int minimum_modalities_;
    float authentication_threshold_;
    
    // Quality thresholds per modality
    float fingerprint_quality_threshold_;
    float palm_quality_threshold_;
    float vein_quality_threshold_;
    
    // Performance metrics
    struct FusionMetrics {
        int total_authentications;
        int successful_authentications;
        int failed_authentications;
        
        float avg_fusion_time_ms;
        float avg_fingerprint_time_ms;
        float avg_palm_time_ms;
        float avg_vein_time_ms;
        
        std::map<std::string, int> modality_usage_count;
        std::map<std::string, float> modality_accuracy;
    } metrics_;
    
    // Status
    std::string status_message_;
    
    // Logging
    void logAuthenticationAttempt(const MultimodalAuthResult& result);
    void logEnrollment(const MultimodalTemplate& template_data);
};

// Utility functions
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float softmax(const std::vector<float>& scores, int index) {
    float exp_sum = 0.0f;
    for (float score : scores) {
        exp_sum += std::exp(score);
    }
    return std::exp(scores[index]) / exp_sum;
}

inline std::string fusionStrategyToString(FusionStrategy strategy) {
    switch(strategy) {
        case FusionStrategy::WEIGHTED_SUM: return "Weighted Sum";
        case FusionStrategy::PRODUCT_RULE: return "Product Rule";
        case FusionStrategy::MAX_RULE: return "Maximum Rule";
        case FusionStrategy::MIN_RULE: return "Minimum Rule";
        case FusionStrategy::ADAPTIVE: return "Adaptive";
        case FusionStrategy::MACHINE_LEARNING: return "Machine Learning";
        case FusionStrategy::DEEP_FUSION: return "Deep Fusion";
        default: return "Unknown";
    }
}

#endif // MULTIMODAL_FUSION_H
