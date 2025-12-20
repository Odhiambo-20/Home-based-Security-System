// multimodal_fusion.cpp - Multi-Modal Biometric Fusion Implementation
#include "multimodal_fusion.h"
#include "../utils/logger.h"
#include <algorithm>
#include <numeric>
#include <thread>
#include <future>

MultimodalFusion::MultimodalFusion()
    : initialized_(false)
    , fusion_strategy_(FusionStrategy::ADAPTIVE)
    , adaptive_weighting_(true)
    , parallel_processing_(true)
    , require_all_modalities_(false)
    , minimum_modalities_(2)
    , authentication_threshold_(0.85f)
    , fingerprint_quality_threshold_(40.0f)
    , palm_quality_threshold_(60.0f)
    , vein_quality_threshold_(70.0f)
{
    // Initialize default weights (equal initially)
    default_weights_.fingerprint_weight = 0.40f;
    default_weights_.palm_print_weight = 0.35f;
    default_weights_.vein_pattern_weight = 0.25f;
    
    metrics_ = {0, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f, {}, {}};
}

MultimodalFusion::~MultimodalFusion() {
    shutdown();
}

bool MultimodalFusion::initialize(const std::string& config_path) {
    Logger::info("Initializing Multimodal Biometric Fusion System");
    
    try {
        // Initialize score normalization statistics
        score_stats_["fingerprint"] = {0.5f, 0.2f, 0.0f, 1.0f};
        score_stats_["palm"] = {0.5f, 0.2f, 0.0f, 1.0f};
        score_stats_["vein"] = {0.5f, 0.2f, 0.0f, 1.0f};
        
        // Initialize ML models (if using ML-based fusion)
        if (fusion_strategy_ == FusionStrategy::MACHINE_LEARNING) {
            fusion_model_.svm_classifier = cv::ml::SVM::create();
            fusion_model_.random_forest = cv::ml::RTrees::create();
        }
        
        initialized_ = true;
        status_message_ = "Multimodal fusion system ready";
        Logger::info("Multimodal fusion system initialized successfully");
        
        return true;
        
    } catch (const std::exception& e) {
        Logger::error(std::string("Multimodal fusion initialization failed: ") + e.what());
        return false;
    }
}

void MultimodalFusion::shutdown() {
    if (initialized_) {
        Logger::info("Shutting down multimodal fusion system");
        fingerprint_scanner_.reset();
        palm_analyzer_.reset();
        vein_detector_.reset();
        initialized_ = false;
    }
}

void MultimodalFusion::setFingerprintScanner(
    std::shared_ptr<FingerprintScanner> scanner) {
    fingerprint_scanner_ = scanner;
}

void MultimodalFusion::setPalmAnalyzer(
    std::shared_ptr<PalmPrintAnalyzer> analyzer) {
    palm_analyzer_ = analyzer;
}

void MultimodalFusion::setVeinDetector(
    std::shared_ptr<VeinPatternDetector> detector) {
    vein_detector_ = detector;
}

bool MultimodalFusion::captureCompleteHand(
    HandScanData& fingerprint_data,
    PalmPrintData& palm_data,
    VeinPatternData& vein_data) {
    
    Logger::info("Capturing complete hand biometrics");
    auto start_time = std::chrono::steady_clock::now();
    
    bool all_success = true;
    
    if (parallel_processing_) {
        // Capture all modalities in parallel
        std::future<HandScanData> fingerprint_future = std::async(
            std::launch::async,
            [this]() { 
                return fingerprint_scanner_->captureHandScan(); 
            }
        );
        
        std::future<VeinPatternData> vein_future = std::async(
            std::launch::async,
            [this]() { 
                return vein_detector_->captureVeinPattern(VeinType::DORSAL_HAND); 
            }
        );
        
        // Capture fingerprint and vein simultaneously
        fingerprint_data = fingerprint_future.get();
        vein_data = vein_future.get();
        
        // Palm print can be extracted from the same hand image
        if (!fingerprint_data.full_hand_image.empty()) {
            palm_data = palm_analyzer_->capturePalmPrint(
                fingerprint_data.full_hand_image);
        }
        
    } else {
        // Sequential capture
        fingerprint_data = fingerprint_scanner_->captureHandScan();
        
        if (!fingerprint_data.full_hand_image.empty()) {
            palm_data = palm_analyzer_->capturePalmPrint(
                fingerprint_data.full_hand_image);
        }
        
        vein_data = vein_detector_->captureVeinPattern(VeinType::DORSAL_HAND);
    }
    
    // Validate captures
    all_success = fingerprint_data.scan_complete && 
                  palm_data.is_valid && 
                  vein_data.is_valid;
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    Logger::info("Complete hand capture finished in " + 
                std::to_string(duration) + "ms");
    Logger::info("Fingerprint: " + 
                std::string(fingerprint_data.scan_complete ? "SUCCESS" : "FAILED"));
    Logger::info("Palm Print: " + 
                std::string(palm_data.is_valid ? "SUCCESS" : "FAILED"));
    Logger::info("Vein Pattern: " + 
                std::string(vein_data.is_valid ? "SUCCESS" : "FAILED"));
    
    return all_success || !require_all_modalities_;
}

MultimodalTemplate MultimodalFusion::enrollUser(
    const std::string& user_id,
    const HandScanData& fingerprint_data,
    const PalmPrintData& palm_data,
    const VeinPatternData& vein_data) {
    
    std::lock_guard<std::mutex> lock(fusion_mutex_);
    
    Logger::info("Enrolling user: " + user_id + " with multimodal biometrics");
    
    MultimodalTemplate template_data;
    template_data.user_id = user_id;
    template_data.hand_side = fingerprint_data.hand_side;
    template_data.enrollment_time = std::chrono::system_clock::now();
    template_data.update_count = 0;
    
    // Create individual modality templates
    
    // 1. Fingerprint templates (all 5 fingers)
    for (const auto& finger : fingerprint_data.fingers) {
        if (finger.is_valid) {
            // Create template for this finger (simplified - actual implementation
            // would use proper template creation from FingerprintScanner)
            FingerprintTemplate fp_template;
            fp_template.user_id = user_id;
            fp_template.hand_side = fingerprint_data.hand_side;
            fp_template.enrollment_time = std::chrono::system_clock::now();
            
            template_data.fingerprint_templates.push_back(fp_template);
            template_data.modality_qualities["fingerprint"] = 
                finger.quality.nfiq_score;
        }
    }
    
    // 2. Palm print template
    if (palm_data.is_valid) {
        template_data.palm_template = palm_analyzer_->createTemplate(
            palm_data, user_id);
        template_data.modality_qualities["palm"] = palm_data.image_quality;
    }
    
    // 3. Vein pattern template
    if (vein_data.is_valid) {
        template_data.vein_template = vein_detector_->createTemplate(
            vein_data, user_id);
        template_data.modality_qualities["vein"] = vein_data.image_quality;
    }
    
    // 4. Create fused features
    std::vector<float> all_features;
    
    // Collect features from all valid modalities
    for (const auto& finger : fingerprint_data.fingers) {
        if (finger.is_valid) {
            all_features.insert(all_features.end(),
                              finger.feature_vector.begin(),
                              finger.feature_vector.end());
        }
    }
    
    if (palm_data.is_valid) {
        all_features.insert(all_features.end(),
                          palm_data.deep_features.begin(),
                          palm_data.deep_features.end());
    }
    
    if (vein_data.is_valid) {
        all_features.insert(all_features.end(),
                          vein_data.deep_features.begin(),
                          vein_data.deep_features.end());
    }
    
    template_data.fused_features = all_features;
    
    // Encrypt template
    template_data.encrypted_data = encryptMultimodalTemplate(template_data);
    
    Logger::info("User " + user_id + " enrolled successfully");
    Logger::info("  - Fingerprints: " + 
                std::to_string(template_data.fingerprint_templates.size()));
    Logger::info("  - Palm print: " + 
                std::string(palm_data.is_valid ? "YES" : "NO"));
    Logger::info("  - Vein pattern: " + 
                std::string(vein_data.is_valid ? "YES" : "NO"));
    Logger::info("  - Fused features: " + 
                std::to_string(template_data.fused_features.size()) + "D");
    
    logEnrollment(template_data);
    
    return template_data;
}

MultimodalAuthResult MultimodalFusion::authenticateUser(
    const HandScanData& probe_fingerprint,
    const PalmPrintData& probe_palm,
    const VeinPatternData& probe_vein,
    const MultimodalTemplate& gallery_template) {
    
    std::lock_guard<std::mutex> lock(fusion_mutex_);
    auto start_time = std::chrono::steady_clock::now();
    
    MultimodalAuthResult result;
    result.authenticated = false;
    result.fusion_score = 0.0f;
    result.confidence = 0.0f;
    result.liveness_verified = false;
    result.matched_user_id = "";
    
    Logger::info("Starting multimodal authentication");
    
    // Perform liveness verification across modalities
    result.liveness_verified = verifyMultimodalLiveness(
        probe_fingerprint, probe_palm, probe_vein);
    
    if (!result.liveness_verified) {
        Logger::warning("Liveness verification FAILED");
        result.spoofing_probability = 0.95f;
        status_message_ = "Liveness check failed - possible spoofing attack";
        return result;
    }
    
    result.spoofing_probability = estimateSpoofingProbability(result);
    
    // Match each modality independently
    std::map<std::string, float> modality_scores;
    std::map<std::string, float> quality_scores;
    
    // 1. Fingerprint matching
    if (probe_fingerprint.scan_complete && 
        !gallery_template.fingerprint_templates.empty()) {
        
        auto fp_start = std::chrono::steady_clock::now();
        
        // Match all fingers and fuse
        std::vector<float> finger_scores;
        for (size_t i = 0; i < probe_fingerprint.fingers.size() && 
                        i < gallery_template.fingerprint_templates.size(); i++) {
            
            if (!probe_fingerprint.fingers[i].is_valid) continue;
            
            // Compute similarity (simplified - actual would use proper matching)
            float score = fingerprint_scanner_->computeSimilarity(
                probe_fingerprint.fingers[i].feature_vector,
                gallery_template.fingerprint_templates[i].finger_embeddings[0]
            );
            finger_scores.push_back(score);
        }
        
        float fp_score = finger_scores.empty() ? 0.0f :
            std::accumulate(finger_scores.begin(), finger_scores.end(), 0.0f) / 
            finger_scores.size();
        
        modality_scores["fingerprint"] = normalizeScore(fp_score, "fingerprint");
        quality_scores["fingerprint"] = probe_fingerprint.overall_quality;
        
        auto fp_end = std::chrono::steady_clock::now();
        
        ModalityResult fp_result;
        fp_result.available = true;
        fp_result.valid = probe_fingerprint.scan_complete;
        fp_result.quality_score = probe_fingerprint.overall_quality;
        fp_result.match_score = fp_score;
        fp_result.confidence = fp_score * (probe_fingerprint.overall_quality / 100.0f);
        fp_result.processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            fp_end - fp_start);
        
        result.modality_results["fingerprint"] = fp_result;
        result.active_modalities.push_back("fingerprint");
        
        Logger::info("Fingerprint match score: " + std::to_string(fp_score));
    }
    
    // 2. Palm print matching
    if (probe_palm.is_valid) {
        auto palm_start = std::chrono::steady_clock::now();
        
        PalmMatchResult palm_match = palm_analyzer_->matchPalmPrints(
            probe_palm, gallery_template.palm_template);
        
        modality_scores["palm"] = normalizeScore(
            palm_match.combined_score, "palm");
        quality_scores["palm"] = probe_palm.image_quality;
        
        auto palm_end = std::chrono::steady_clock::now();
        
        ModalityResult palm_result;
        palm_result.available = true;
        palm_result.valid = probe_palm.is_valid;
        palm_result.quality_score = probe_palm.image_quality;
        palm_result.match_score = palm_match.combined_score;
        palm_result.confidence = palm_match.confidence_score;
        palm_result.processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            palm_end - palm_start);
        
        result.modality_results["palm"] = palm_result;
        result.active_modalities.push_back("palm");
        
        Logger::info("Palm print match score: " + 
                    std::to_string(palm_match.combined_score));
    }
    
    // 3. Vein pattern matching
    if (probe_vein.is_valid) {
        auto vein_start = std::chrono::steady_clock::now();
        
        VeinMatchResult vein_match = vein_detector_->matchVeinPatterns(
            probe_vein, gallery_template.vein_template);
        
        modality_scores["vein"] = normalizeScore(
            vein_match.combined_score, "vein");
        quality_scores["vein"] = probe_vein.image_quality;
        
        auto vein_end = std::chrono::steady_clock::now();
        
        ModalityResult vein_result;
        vein_result.available = true;
        vein_result.valid = probe_vein.is_valid;
        vein_result.quality_score = probe_vein.image_quality;
        vein_result.match_score = vein_match.combined_score;
        vein_result.confidence = vein_match.confidence_score;
        vein_result.processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            vein_end - vein_start);
        
        result.modality_results["vein"] = vein_result;
        result.active_modalities.push_back("vein");
        
        Logger::info("Vein pattern match score: " + 
                    std::to_string(vein_match.combined_score));
    }
    
    // Check minimum modalities requirement
    if (result.active_modalities.size() < minimum_modalities_) {
        Logger::error("Insufficient modalities: " + 
                     std::to_string(result.active_modalities.size()) + " < " +
                     std::to_string(minimum_modalities_));
        status_message_ = "Insufficient biometric data captured";
        return result;
    }
    
    // Perform score fusion
    ModalityWeights weights = adaptive_weighting_ ? 
        computeAdaptiveWeights(quality_scores) : default_weights_;
    
    result.fusion_score = fuseScores(modality_scores, weights);
    result.decision_method = fusionStrategyToString(fusion_strategy_);
    
    // Make authentication decision
    result.authenticated = (result.fusion_score >= authentication_threshold_);
    
    if (result.authenticated) {
        result.matched_user_id = gallery_template.user_id;
        result.confidence = result.fusion_score;
        
        // Estimate FAR/FRR (simplified - actual would use statistical models)
        result.false_accept_rate = std::pow(10, -6) * 
            std::exp(-(result.fusion_score - 0.5) * 10);
        result.false_reject_rate = std::pow(10, -2) * 
            std::exp((result.fusion_score - 0.95) * 10);
    }
    
    auto end_time = std::chrono::steady_clock::now();
    result.total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    Logger::info("Authentication complete:");
    Logger::info("  - Result: " + std::string(result.authenticated ? "ACCEPT" : "REJECT"));
    Logger::info("  - Fusion score: " + std::to_string(result.fusion_score));
    Logger::info("  - Confidence: " + std::to_string(result.confidence));
    Logger::info("  - Time: " + std::to_string(result.total_time.count()) + "ms");
    Logger::info("  - Active modalities: " + 
                std::to_string(result.active_modalities.size()));
    
    // Update metrics
    metrics_.total_authentications++;
    if (result.authenticated) {
        metrics_.successful_authentications++;
    } else {
        metrics_.failed_authentications++;
    }
    
    metrics_.avg_fusion_time_ms = 
        (metrics_.avg_fusion_time_ms * (metrics_.total_authentications - 1) + 
         result.total_time.count()) / metrics_.total_authentications;
    
    logAuthenticationAttempt(result);
    
    status_message_ = result.authenticated ? 
        "Authentication successful" : "Authentication failed";
    
    return result;
}

float MultimodalFusion::fuseScores(
    const std::map<std::string, float>& modality_scores,
    const ModalityWeights& weights) {
    
    switch (fusion_strategy_) {
        case FusionStrategy::WEIGHTED_SUM:
            return weightedSumFusion(modality_scores, weights);
            
        case FusionStrategy::PRODUCT_RULE:
            return productRuleFusion(modality_scores);
            
        case FusionStrategy::MAX_RULE: {
            float max_score = 0.0f;
            for (const auto& pair : modality_scores) {
                max_score = std::max(max_score, pair.second);
            }
            return max_score;
        }
        
        case FusionStrategy::MIN_RULE: {
            float min_score = 1.0f;
            for (const auto& pair : modality_scores) {
                min_score = std::min(min_score, pair.second);
            }
            return min_score;
        }
        
        case FusionStrategy::ADAPTIVE:
        case FusionStrategy::MACHINE_LEARNING:
        case FusionStrategy::DEEP_FUSION:
        default:
            return weightedSumFusion(modality_scores, weights);
    }
}

float MultimodalFusion::weightedSumFusion(
    const std::map<std::string, float>& scores,
    const ModalityWeights& weights) {
    
    float fused_score = 0.0f;
    
    if (scores.count("fingerprint")) {
        fused_score += scores.at("fingerprint") * weights.fingerprint_weight;
    }
    if (scores.count("palm")) {
        fused_score += scores.at("palm") * weights.palm_print_weight;
    }
    if (scores.count("vein")) {
        fused_score += scores.at("vein") * weights.vein_pattern_weight;
    }
    
    return fused_score;
}

float MultimodalFusion::productRuleFusion(
    const std::map<std::string, float>& scores) {
    
    float product = 1.0f;
    int count = 0;
    
    for (const auto& pair : scores) {
        product *= pair.second;
        count++;
    }
    
    // Return geometric mean
    return count > 0 ? std::pow(product, 1.0f / count) : 0.0f;
}

ModalityWeights MultimodalFusion::computeAdaptiveWeights(
    const std::map<std::string, float>& quality_scores) {
    
    ModalityWeights weights;
    
    // Convert quality scores to weights using softmax
    std::vector<float> qualities;
    qualities.reserve(3);
    
    if (quality_scores.count("fingerprint")) {
        qualities.push_back(quality_scores.at("fingerprint") / 100.0f);
    } else {
        qualities.push_back(0.0f);
    }
    
    if (quality_scores.count("palm")) {
        qualities.push_back(quality_scores.at("palm") / 100.0f);
    } else {
        qualities.push_back(0.0f);
    }
    
    if (quality_scores.count("vein")) {
        qualities.push_back(quality_scores.at("vein") / 100.0f);
    } else {
        qualities.push_back(0.0f);
    }
    
    // Apply softmax
    weights.fingerprint_weight = softmax(qualities, 0);
    weights.palm_print_weight = softmax(qualities, 1);
    weights.vein_pattern_weight = softmax(qualities, 2);
    
    Logger::info("Adaptive weights computed:");
    Logger::info("  - Fingerprint: " + std::to_string(weights.fingerprint_weight));
    Logger::info("  - Palm: " + std::to_string(weights.palm_print_weight));
    Logger::info("  - Vein: " + std::to_string(weights.vein_pattern_weight));
    
    return weights;
}

// Additional helper implementations...
float MultimodalFusion::normalizeScore(float raw_score, 
                                       const std::string& modality) {
    // Z-score normalization
    auto stats = score_stats_[modality];
    return (raw_score - stats.mean) / stats.stddev * 0.2f + 0.5f;
}

void MultimodalFusion::logAuthenticationAttempt(const MultimodalAuthResult& result) {
    // Log to file or database for audit trail
    Logger::info("AUTH: " + result.matched_user_id + 
                " | Score: " + std::to_string(result.fusion_score) +
                " | Result: " + std::string(result.authenticated ? "ACCEPT" : "REJECT"));
}

void MultimodalFusion::logEnrollment(const MultimodalTemplate& template_data) {
    Logger::info("ENROLL: " + template_data.user_id);
}

// Remaining helper function implementations...
