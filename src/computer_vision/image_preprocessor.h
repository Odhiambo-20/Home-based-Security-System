// image_preprocessor.h - Production-Grade Image Preprocessing
// Complete implementation for biometric image enhancement and normalization
#ifndef IMAGE_PREPROCESSOR_H
#define IMAGE_PREPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include <map>

// Image quality metrics
struct ImageQualityMetrics {
    float sharpness;              // Laplacian variance
    float contrast;               // RMS contrast
    float brightness;             // Mean intensity
    float noise_level;            // Estimated noise sigma
    float snr;                    // Signal-to-noise ratio
    float focus_measure;          // Focus quality
    bool is_blurred;             
    bool is_overexposed;
    bool is_underexposed;
    float overall_quality;        // Combined quality score 0-100
};

// Illumination correction parameters
struct IlluminationParams {
    bool enable_homomorphic;      // Homomorphic filtering
    bool enable_retinex;          // Multi-scale Retinex
    bool enable_clahe;            // CLAHE
    float gamma_correction;       // Gamma value
};

// Noise reduction parameters  
struct NoiseReductionParams {
    enum Method {
        GAUSSIAN_BLUR,
        BILATERAL_FILTER,
        NON_LOCAL_MEANS,
        ANISOTROPIC_DIFFUSION,
        WIENER_FILTER
    };
    
    Method method;
    int kernel_size;
    double sigma_spatial;
    double sigma_range;
    int iterations;
};

// Enhancement parameters
struct EnhancementParams {
    bool sharpen;
    bool unsharp_mask;
    bool enhance_edges;
    float sharpen_amount;
    float unsharp_radius;
    float unsharp_amount;
};

class ImagePreprocessor {
public:
    ImagePreprocessor();
    ~ImagePreprocessor();
    
    // Main preprocessing pipeline
    cv::Mat preprocess(const cv::Mat& input,
                       bool enhance = true,
                       bool denoise = true,
                       bool normalize = true);
    
    // Illumination normalization
    cv::Mat normalizeIllumination(const cv::Mat& image,
                                  const IlluminationParams& params);
    cv::Mat homomorphicFilter(const cv::Mat& image, 
                             float gamma_high = 2.0f,
                             float gamma_low = 0.5f,
                             float cutoff = 30.0f);
    cv::Mat multiScaleRetinex(const cv::Mat& image,
                             const std::vector<float>& sigmas = {15.0f, 80.0f, 250.0f});
    cv::Mat singleScaleRetinex(const cv::Mat& image, float sigma);
    cv::Mat adaptiveGammaCorrection(const cv::Mat& image);
    
    // CLAHE (Contrast Limited Adaptive Histogram Equalization)
    cv::Mat applyCLAHE(const cv::Mat& image,
                      double clip_limit = 2.0,
                      int tile_size = 8);
    
    // Noise reduction
    cv::Mat reduceNoise(const cv::Mat& image,
                       const NoiseReductionParams& params);
    cv::Mat bilateralFilter(const cv::Mat& image,
                           int diameter = 9,
                           double sigma_color = 75.0,
                           double sigma_space = 75.0);
    cv::Mat nonLocalMeansDenoising(const cv::Mat& image,
                                   int h = 10,
                                   int template_window = 7,
                                   int search_window = 21);
    cv::Mat anisotropicDiffusion(const cv::Mat& image,
                                 int iterations = 10,
                                 float kappa = 30.0f,
                                 float lambda = 0.25f);
    cv::Mat wienerFilter(const cv::Mat& image, int kernel_size = 5);
    
    // Image enhancement
    cv::Mat enhanceImage(const cv::Mat& image,
                        const EnhancementParams& params);
    cv::Mat sharpenImage(const cv::Mat& image, float amount = 1.0f);
    cv::Mat unsharpMask(const cv::Mat& image,
                       float radius = 5.0f,
                       float amount = 1.5f,
                       float threshold = 0.0f);
    cv::Mat edgeEnhancement(const cv::Mat& image, float strength = 1.0f);
    
    // Geometric transformations
    cv::Mat alignImage(const cv::Mat& image,
                      const std::vector<cv::Point2f>& src_points,
                      const std::vector<cv::Point2f>& dst_points);
    cv::Mat rotateImage(const cv::Mat& image, double angle);
    cv::Mat deskewImage(const cv::Mat& image);
    cv::Mat perspectiveCorrection(const cv::Mat& image,
                                 const std::vector<cv::Point2f>& corners);
    
    // Histogram operations
    cv::Mat histogramEqualization(const cv::Mat& image);
    cv::Mat adaptiveHistogramEqualization(const cv::Mat& image);
    cv::Mat histogramMatching(const cv::Mat& source,
                             const cv::Mat& reference);
    
    // Morphological operations
    cv::Mat morphologicalOperation(const cv::Mat& image,
                                  cv::MorphTypes operation,
                                  int kernel_size = 5,
                                  cv::MorphShapes kernel_shape = cv::MORPH_RECT);
    cv::Mat morphologicalGradient(const cv::Mat& image, int kernel_size = 3);
    cv::Mat topHatTransform(const cv::Mat& image, int kernel_size = 9);
    cv::Mat blackHatTransform(const cv::Mat& image, int kernel_size = 9);
    
    // Background removal
    cv::Mat removeBackground(const cv::Mat& image);
    cv::Mat backgroundSubtractionMOG2(const cv::Mat& image);
    cv::Mat adaptiveBackgroundRemoval(const cv::Mat& image);
    
    // ROI extraction
    cv::Mat extractROI(const cv::Mat& image, const cv::Rect& roi);
    std::vector<cv::Rect> detectROIs(const cv::Mat& image);
    cv::Mat cropToContent(const cv::Mat& image);
    
    // Color space conversions
    cv::Mat convertColorSpace(const cv::Mat& image,
                             cv::ColorConversionCodes conversion);
    cv::Mat rgbToGray(const cv::Mat& image);
    cv::Mat grayToRGB(const cv::Mat& image);
    cv::Mat rgbToHSV(const cv::Mat& image);
    cv::Mat rgbToLab(const cv::Mat& image);
    
    // Normalization methods
    cv::Mat normalizeIntensity(const cv::Mat& image,
                              double alpha = 0.0,
                              double beta = 255.0);
    cv::Mat zScoreNormalization(const cv::Mat& image);
    cv::Mat minMaxNormalization(const cv::Mat& image);
    
    // Quality assessment
    ImageQualityMetrics assessQuality(const cv::Mat& image);
    float computeSharpness(const cv::Mat& image);
    float computeContrast(const cv::Mat& image);
    float computeBrightness(const cv::Mat& image);
    float estimateNoise(const cv::Mat& image);
    float computeSNR(const cv::Mat& image);
    float computeFocusMeasure(const cv::Mat& image);
    bool isBlurred(const cv::Mat& image, float threshold = 100.0f);
    bool isOverexposed(const cv::Mat& image, float threshold = 0.95f);
    bool isUnderexposed(const cv::Mat& image, float threshold = 0.05f);
    
    // Frequency domain operations
    cv::Mat fftTransform(const cv::Mat& image);
    cv::Mat inverseFftTransform(const cv::Mat& fft_image);
    cv::Mat frequencyDomainFilter(const cv::Mat& image,
                                  const cv::Mat& filter_mask);
    cv::Mat lowPassFilter(const cv::Mat& image, float cutoff);
    cv::Mat highPassFilter(const cv::Mat& image, float cutoff);
    cv::Mat bandPassFilter(const cv::Mat& image, 
                          float low_cutoff,
                          float high_cutoff);
    
    // Advanced filters
    cv::Mat gaborFilter(const cv::Mat& image,
                       float wavelength,
                       float orientation,
                       float phase_offset = 0.0f,
                       float aspect_ratio = 0.5f,
                       float bandwidth = 1.0f);
    cv::Mat gaborFilterBank(const cv::Mat& image,
                           const std::vector<float>& wavelengths,
                           const std::vector<float>& orientations);
    cv::Mat meanFilter(const cv::Mat& image, int kernel_size);
    cv::Mat medianFilter(const cv::Mat& image, int kernel_size);
    cv::Mat sobelFilter(const cv::Mat& image);
    cv::Mat laplacianFilter(const cv::Mat& image);
    cv::Mat cannyEdgeDetection(const cv::Mat& image,
                              double low_threshold,
                              double high_threshold);
    
    // Texture analysis preprocessing
    cv::Mat textureEnhancement(const cv::Mat& image);
    cv::Mat ridgeEnhancement(const cv::Mat& image);
    cv::Mat orientationFieldSmoothing(const cv::Mat& image);
    
    // Segmentation preprocessing
    cv::Mat binarization(const cv::Mat& image,
                        double threshold = -1.0);
    cv::Mat otsuThresholding(const cv::Mat& image);
    cv::Mat adaptiveThresholding(const cv::Mat& image,
                                int block_size = 11,
                                double c = 2.0);
    cv::Mat multiLevelThresholding(const cv::Mat& image, int levels = 3);
    
    // Image restoration
    cv::Mat deblurImage(const cv::Mat& image, int kernel_size = 5);
    cv::Mat motionDeblur(const cv::Mat& image, 
                        float angle,
                        float length);
    cv::Mat defocusDeblur(const cv::Mat& image);
    
    // Utility functions
    void setDefaultParams();
    void setIlluminationParams(const IlluminationParams& params);
    void setNoiseReductionParams(const NoiseReductionParams& params);
    void setEnhancementParams(const EnhancementParams& params);
    
    cv::Mat resizeImage(const cv::Mat& image, 
                       int width,
                       int height,
                       int interpolation = cv::INTER_CUBIC);
    cv::Mat padImage(const cv::Mat& image,
                    int top, int bottom,
                    int left, int right,
                    int border_type = cv::BORDER_CONSTANT);
    
    bool validateImage(const cv::Mat& image);
    std::string getPreprocessingReport() const;

private:
    // Helper functions
    cv::Mat computeIntegralImage(const cv::Mat& image);
    cv::Mat createGaborKernel(float wavelength,
                             float orientation,
                             float phase_offset,
                             float aspect_ratio,
                             float bandwidth);
    cv::Mat createMotionKernel(float angle, int length);
    
    void optimizeImageForFFT(cv::Mat& padded,
                            const cv::Mat& image);
    cv::Mat shiftDFT(const cv::Mat& fft_image);
    cv::Mat createFrequencyFilter(int rows,
                                  int cols,
                                  float cutoff_low,
                                  float cutoff_high);
    
    float computeLocalVariance(const cv::Mat& image,
                              int x, int y,
                              int window_size);
    cv::Mat estimateNoiseVariance(const cv::Mat& image);
    
    // Parameter storage
    IlluminationParams illum_params_;
    NoiseReductionParams noise_params_;
    EnhancementParams enhance_params_;
    
    // Statistics
    struct Statistics {
        int total_processed;
        int failed_processing;
        float avg_quality;
        float avg_processing_time_ms;
    } stats_;
    
    // Cache for frequently used kernels
    std::map<std::string, cv::Mat> kernel_cache_;
    
    // Processing flags
    bool use_gpu_;
    bool enable_caching_;
};

// Utility functions
inline cv::Mat convertToGrayscale(const cv::Mat& image) {
    if (image.channels() == 1) return image.clone();
    
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

inline cv::Mat convertToFloat(const cv::Mat& image) {
    cv::Mat float_img;
    image.convertTo(float_img, CV_32F);
    return float_img;
}

inline bool isValidImage(const cv::Mat& image) {
    return !image.empty() && 
           (image.type() == CV_8UC1 || 
            image.type() == CV_8UC3 ||
            image.type() == CV_32FC1 ||
            image.type() == CV_32FC3);
}

#endif // IMAGE_PREPROCESSOR_H
