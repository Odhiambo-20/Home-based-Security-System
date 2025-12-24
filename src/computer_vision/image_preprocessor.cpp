// image_preprocessor.cpp - COMPLETE PRODUCTION IMPLEMENTATION
// Full industrial-grade image preprocessing for biometric systems
#include "image_preprocessor.h"
#include "../utils/logger.h"
#include <cmath>
#include <algorithm>
#include <numeric>

ImagePreprocessor::ImagePreprocessor()
    : use_gpu_(false)
    , enable_caching_(true)
{
    setDefaultParams();
    stats_ = {0, 0, 0.0f, 0.0f};
}

ImagePreprocessor::~ImagePreprocessor() {
    kernel_cache_.clear();
}

void ImagePreprocessor::setDefaultParams() {
    // Illumination parameters
    illum_params_.enable_homomorphic = false;
    illum_params_.enable_retinex = true;
    illum_params_.enable_clahe = true;
    illum_params_.gamma_correction = 1.0f;
    
    // Noise reduction parameters
    noise_params_.method = NoiseReductionParams::BILATERAL_FILTER;
    noise_params_.kernel_size = 9;
    noise_params_.sigma_spatial = 75.0;
    noise_params_.sigma_range = 75.0;
    noise_params_.iterations = 1;
    
    // Enhancement parameters
    enhance_params_.sharpen = true;
    enhance_params_.unsharp_mask = false;
    enhance_params_.enhance_edges = true;
    enhance_params_.sharpen_amount = 1.0f;
    enhance_params_.unsharp_radius = 5.0f;
    enhance_params_.unsharp_amount = 1.5f;
}

cv::Mat ImagePreprocessor::preprocess(const cv::Mat& input,
                                      bool enhance,
                                      bool denoise,
                                      bool normalize) {
    if (!validateImage(input)) {
        Logger::error("Invalid input image for preprocessing");
        return cv::Mat();
    }
    
    auto start = std::chrono::steady_clock::now();
    
    cv::Mat processed = input.clone();
    
    // Convert to grayscale if needed
    if (processed.channels() == 3) {
        processed = rgbToGray(processed);
    }
    
    // Illumination normalization
    if (normalize) {
        processed = normalizeIllumination(processed, illum_params_);
    }
    
    // Noise reduction
    if (denoise) {
        processed = reduceNoise(processed, noise_params_);
    }
    
    // Enhancement
    if (enhance) {
        processed = enhanceImage(processed, enhance_params_);
    }
    
    // Final intensity normalization
    processed = normalizeIntensity(processed, 0, 255);
    
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    stats_.total_processed++;
    stats_.avg_processing_time_ms = 
        (stats_.avg_processing_time_ms * (stats_.total_processed - 1) + duration) / 
        stats_.total_processed;
    
    return processed;
}

cv::Mat ImagePreprocessor::normalizeIllumination(const cv::Mat& image,
                                                 const IlluminationParams& params) {
    cv::Mat normalized = image.clone();
    
    if (params.enable_homomorphic) {
        normalized = homomorphicFilter(normalized);
    }
    
    if (params.enable_retinex) {
        normalized = multiScaleRetinex(normalized);
    }
    
    if (params.enable_clahe) {
        normalized = applyCLAHE(normalized, 2.0, 8);
    }
    
    if (params.gamma_correction != 1.0f) {
        normalized = adaptiveGammaCorrection(normalized);
    }
    
    return normalized;
}

cv::Mat ImagePreprocessor::homomorphicFilter(const cv::Mat& image,
                                             float gamma_high,
                                             float gamma_low,
                                             float cutoff) {
    // Convert to float
    cv::Mat float_img;
    image.convertTo(float_img, CV_32F);
    float_img += 1.0f; // Avoid log(0)
    
    // Take logarithm
    cv::log(float_img, float_img);
    
    // Apply FFT
    cv::Mat fft_img = fftTransform(float_img);
    
    // Create homomorphic filter
    int rows = fft_img.rows;
    int cols = fft_img.cols;
    cv::Mat filter = cv::Mat::zeros(rows, cols, CV_32F);
    
    float cutoff_sq = cutoff * cutoff;
    int center_x = cols / 2;
    int center_y = rows / 2;
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float dx = j - center_x;
            float dy = i - center_y;
            float d_sq = dx * dx + dy * dy;
            
            float h = (gamma_high - gamma_low) * (1.0f - std::exp(-d_sq / (2.0f * cutoff_sq))) + gamma_low;
            filter.at<float>(i, j) = h;
        }
    }
    
    // Apply filter
    cv::Mat filtered = frequencyDomainFilter(float_img, filter);
    
    // Take exponential
    cv::exp(filtered, filtered);
    filtered -= 1.0f;
    
    // Convert back to 8-bit
    cv::Mat result;
    filtered.convertTo(result, CV_8U);
    
    return result;
}

cv::Mat ImagePreprocessor::multiScaleRetinex(const cv::Mat& image,
                                             const std::vector<float>& sigmas) {
    cv::Mat result = cv::Mat::zeros(image.size(), CV_32F);
    
    for (float sigma : sigmas) {
        cv::Mat ssr = singleScaleRetinex(image, sigma);
        result += ssr;
    }
    
    result /= static_cast<float>(sigmas.size());
    
    // Normalize to 0-255
    cv::normalize(result, result, 0, 255, cv::NORM_MINMAX);
    
    cv::Mat output;
    result.convertTo(output, CV_8U);
    
    return output;
}

cv::Mat ImagePreprocessor::singleScaleRetinex(const cv::Mat& image, float sigma) {
    cv::Mat float_img;
    image.convertTo(float_img, CV_32F);
    float_img += 1.0f;
    
    // Gaussian blur
    cv::Mat blurred;
    int ksize = static_cast<int>(6 * sigma + 1);
    if (ksize % 2 == 0) ksize++;
    cv::GaussianBlur(float_img, blurred, cv::Size(ksize, ksize), sigma);
    
    // Log of original
    cv::Mat log_img;
    cv::log(float_img, log_img);
    
    // Log of blurred
    cv::Mat log_blurred;
    cv::log(blurred, log_blurred);
    
    // Subtract
    cv::Mat retinex = log_img - log_blurred;
    
    return retinex;
}

cv::Mat ImagePreprocessor::adaptiveGammaCorrection(const cv::Mat& image) {
    // Compute optimal gamma based on image histogram
    cv::Mat hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    
    cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    
    // Calculate mean intensity
    double mean_intensity = cv::mean(image)[0];
    
    // Adaptive gamma: darker images get higher gamma
    float gamma = 1.0f;
    if (mean_intensity < 85) {
        gamma = 1.5f;
    } else if (mean_intensity < 128) {
        gamma = 1.2f;
    } else if (mean_intensity > 170) {
        gamma = 0.8f;
    }
    
    // Apply gamma correction
    cv::Mat lut(1, 256, CV_8U);
    for (int i = 0; i < 256; i++) {
        lut.at<uchar>(i) = cv::saturate_cast<uchar>(std::pow(i / 255.0, gamma) * 255.0);
    }
    
    cv::Mat corrected;
    cv::LUT(image, lut, corrected);
    
    return corrected;
}

cv::Mat ImagePreprocessor::applyCLAHE(const cv::Mat& image,
                                      double clip_limit,
                                      int tile_size) {
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(clip_limit);
    clahe->setTilesGridSize(cv::Size(tile_size, tile_size));
    
    cv::Mat enhanced;
    clahe->apply(image, enhanced);
    
    return enhanced;
}

cv::Mat ImagePreprocessor::reduceNoise(const cv::Mat& image,
                                       const NoiseReductionParams& params) {
    switch (params.method) {
        case NoiseReductionParams::BILATERAL_FILTER:
            return bilateralFilter(image, params.kernel_size,
                                  params.sigma_range, params.sigma_spatial);
        
        case NoiseReductionParams::NON_LOCAL_MEANS:
            return nonLocalMeansDenoising(image, 10, 7, 21);
        
        case NoiseReductionParams::ANISOTROPIC_DIFFUSION:
            return anisotropicDiffusion(image, params.iterations, 30.0f, 0.25f);
        
        case NoiseReductionParams::WIENER_FILTER:
            return wienerFilter(image, params.kernel_size);
        
        case NoiseReductionParams::GAUSSIAN_BLUR:
        default: {
            cv::Mat blurred;
            cv::GaussianBlur(image, blurred, 
                           cv::Size(params.kernel_size, params.kernel_size),
                           params.sigma_spatial);
            return blurred;
        }
    }
}

cv::Mat ImagePreprocessor::bilateralFilter(const cv::Mat& image,
                                           int diameter,
                                           double sigma_color,
                                           double sigma_space) {
    cv::Mat filtered;
    cv::bilateralFilter(image, filtered, diameter, sigma_color, sigma_space);
    return filtered;
}

cv::Mat ImagePreprocessor::nonLocalMeansDenoising(const cv::Mat& image,
                                                  int h,
                                                  int template_window,
                                                  int search_window) {
    cv::Mat denoised;
    cv::fastNlMeansDenoising(image, denoised, h, template_window, search_window);
    return denoised;
}

cv::Mat ImagePreprocessor::anisotropicDiffusion(const cv::Mat& image,
                                                int iterations,
                                                float kappa,
                                                float lambda) {
    cv::Mat input, output;
    image.convertTo(input, CV_32F);
    output = input.clone();
    
    for (int iter = 0; iter < iterations; iter++) {
        cv::Mat gradN = cv::Mat::zeros(input.size(), CV_32F);
        cv::Mat gradS = cv::Mat::zeros(input.size(), CV_32F);
        cv::Mat gradE = cv::Mat::zeros(input.size(), CV_32F);
        cv::Mat gradW = cv::Mat::zeros(input.size(), CV_32F);
        
        // Compute gradients
        for (int i = 1; i < input.rows - 1; i++) {
            for (int j = 1; j < input.cols - 1; j++) {
                float center = input.at<float>(i, j);
                
                gradN.at<float>(i, j) = input.at<float>(i-1, j) - center;
                gradS.at<float>(i, j) = input.at<float>(i+1, j) - center;
                gradE.at<float>(i, j) = input.at<float>(i, j+1) - center;
                gradW.at<float>(i, j) = input.at<float>(i, j-1) - center;
            }
        }
        
        // Compute diffusion coefficients
        cv::Mat cN, cS, cE, cW;
        float kappa_sq = kappa * kappa;
        
        cv::exp(-(gradN.mul(gradN)) / kappa_sq, cN);
        cv::exp(-(gradS.mul(gradS)) / kappa_sq, cS);
        cv::exp(-(gradE.mul(gradE)) / kappa_sq, cE);
        cv::exp(-(gradW.mul(gradW)) / kappa_sq, cW);
        
        // Update image
        cv::Mat update = lambda * (cN.mul(gradN) + cS.mul(gradS) + 
                                  cE.mul(gradE) + cW.mul(gradW));
        
        output = input + update;
        output.copyTo(input);
    }
    
    cv::Mat result;
    output.convertTo(result, CV_8U);
    
    return result;
}

cv::Mat ImagePreprocessor::wienerFilter(const cv::Mat& image, int kernel_size) {
    cv::Mat float_img;
    image.convertTo(float_img, CV_32F);
    
    // Estimate noise variance
    cv::Mat noise_var = estimateNoiseVariance(image);
    
    // Local mean
    cv::Mat mean_img;
    cv::blur(float_img, mean_img, cv::Size(kernel_size, kernel_size));
    
    // Local variance
    cv::Mat mean_sq;
    cv::blur(float_img.mul(float_img), mean_sq, cv::Size(kernel_size, kernel_size));
    cv::Mat var_img = mean_sq - mean_img.mul(mean_img);
    
    // Wiener filter
    cv::Mat wiener = cv::Mat::zeros(image.size(), CV_32F);
    float global_noise_var = cv::mean(noise_var)[0];
    
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            float local_var = var_img.at<float>(i, j);
            float ratio = std::max(0.0f, (local_var - global_noise_var) / (local_var + 1e-6f));
            
            wiener.at<float>(i, j) = mean_img.at<float>(i, j) + 
                                     ratio * (float_img.at<float>(i, j) - mean_img.at<float>(i, j));
        }
    }
    
    cv::Mat result;
    wiener.convertTo(result, CV_8U);
    
    return result;
}

cv::Mat ImagePreprocessor::estimateNoiseVariance(const cv::Mat& image) {
    // Estimate noise using median absolute deviation
    cv::Mat lap;
    cv::Laplacian(image, lap, CV_32F);
    
    cv::Mat abs_lap = cv::abs(lap);
    
    // Flatten and compute median
    cv::Mat flat = abs_lap.reshape(1, abs_lap.total());
    std::vector<float> values(flat.begin<float>(), flat.end<float>());
    std::sort(values.begin(), values.end());
    
    float median = values[values.size() / 2];
    float sigma = median / 0.6745f;
    
    cv::Mat noise_var = cv::Mat::ones(image.size(), CV_32F) * (sigma * sigma);
    
    return noise_var;
}

cv::Mat ImagePreprocessor::enhanceImage(const cv::Mat& image,
                                        const EnhancementParams& params) {
    cv::Mat enhanced = image.clone();
    
    if (params.sharpen) {
        enhanced = sharpenImage(enhanced, params.sharpen_amount);
    }
    
    if (params.unsharp_mask) {
        enhanced = unsharpMask(enhanced, params.unsharp_radius, params.unsharp_amount);
    }
    
    if (params.enhance_edges) {
        enhanced = edgeEnhancement(enhanced, 1.0f);
    }
    
    return enhanced;
}

cv::Mat ImagePreprocessor::sharpenImage(const cv::Mat& image, float amount) {
    // Sharpening kernel
    cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
        0, -amount, 0,
        -amount, 1 + 4*amount, -amount,
        0, -amount, 0
    );
    
    cv::Mat sharpened;
    cv::filter2D(image, sharpened, -1, kernel);
    
    return sharpened;
}

cv::Mat ImagePreprocessor::unsharpMask(const cv::Mat& image,
                                       float radius,
                                       float amount,
                                       float threshold) {
    // Gaussian blur
    cv::Mat blurred;
    int ksize = static_cast<int>(6 * radius + 1);
    if (ksize % 2 == 0) ksize++;
    cv::GaussianBlur(image, blurred, cv::Size(ksize, ksize), radius);
    
    // Compute mask
    cv::Mat mask = image - blurred;
    
    // Apply threshold
    if (threshold > 0) {
        mask.setTo(0, cv::abs(mask) < threshold);
    }
    
    // Add weighted mask
    cv::Mat result = image + amount * mask;
    
    // Clip values
    cv::Mat output;
    result.convertTo(output, CV_8U);
    
    return output;
}

cv::Mat ImagePreprocessor::edgeEnhancement(const cv::Mat& image, float strength) {
    // Detect edges
    cv::Mat edges;
    cv::Laplacian(image, edges, CV_32F);
    
    // Enhance edges
    cv::Mat float_img;
    image.convertTo(float_img, CV_32F);
    
    cv::Mat enhanced = float_img - strength * edges;
    
    cv::Mat result;
    enhanced.convertTo(result, CV_8U);
    
    return result;
}

// FFT operations
cv::Mat ImagePreprocessor::fftTransform(const cv::Mat& image) {
    cv::Mat padded;
    optimizeImageForFFT(padded, image);
    
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complex_img;
    cv::merge(planes, 2, complex_img);
    
    cv::dft(complex_img, complex_img);
    
    return complex_img;
}

cv::Mat ImagePreprocessor::inverseFftTransform(const cv::Mat& fft_image) {
    cv::Mat inverse;
    cv::idft(fft_image, inverse, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    
    return inverse;
}

void ImagePreprocessor::optimizeImageForFFT(cv::Mat& padded, const cv::Mat& image) {
    int m = cv::getOptimalDFTSize(image.rows);
    int n = cv::getOptimalDFTSize(image.cols);
    
    cv::copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols,
                      cv::BORDER_CONSTANT, cv::Scalar::all(0));
}

cv::Mat ImagePreprocessor::frequencyDomainFilter(const cv::Mat& image,
                                                 const cv::Mat& filter_mask) {
    // Apply filter in frequency domain
    cv::Mat filtered;
    cv::mulSpectrums(image, filter_mask, filtered, 0);
    
    // Inverse FFT
    cv::Mat result = inverseFftTransform(filtered);
    
    return result;
}

// Gabor filter
cv::Mat ImagePreprocessor::gaborFilter(const cv::Mat& image,
                                       float wavelength,
                                       float orientation,
                                       float phase_offset,
                                       float aspect_ratio,
                                       float bandwidth) {
    // Create Gabor kernel
    cv::Mat kernel = createGaborKernel(wavelength, orientation, phase_offset,
                                       aspect_ratio, bandwidth);
    
    // Apply filter
    cv::Mat filtered;
    cv::filter2D(image, filtered, CV_32F, kernel);
    
    // Normalize
    cv::normalize(filtered, filtered, 0, 255, cv::NORM_MINMAX);
    
    cv::Mat result;
    filtered.convertTo(result, CV_8U);
    
    return result;
}

cv::Mat ImagePreprocessor::createGaborKernel(float wavelength,
                                             float orientation,
                                             float phase_offset,
                                             float aspect_ratio,
                                             float bandwidth) {
    float sigma = wavelength * bandwidth / M_PI;
    int ksize = static_cast<int>(6 * sigma + 1);
    if (ksize % 2 == 0) ksize++;
    
    cv::Mat kernel = cv::Mat::zeros(ksize, ksize, CV_32F);
    
    int center = ksize / 2;
    float sigma_x = sigma;
    float sigma_y = sigma / aspect_ratio;
    
    float cos_theta = std::cos(orientation);
    float sin_theta = std::sin(orientation);
    
    for (int i = 0; i < ksize; i++) {
        for (int j = 0; j < ksize; j++) {
            float x = j - center;
            float y = i - center;
            
            // Rotate coordinates
            float x_rot = x * cos_theta + y * sin_theta;
            float y_rot = -x * sin_theta + y * cos_theta;
            
            // Gabor function
            float gaussian = std::exp(-(x_rot * x_rot / (2 * sigma_x * sigma_x) +
                                       y_rot * y_rot / (2 * sigma_y * sigma_y)));
            
            float sinusoid = std::cos(2 * M_PI * x_rot / wavelength + phase_offset);
            
            kernel.at<float>(i, j) = gaussian * sinusoid;
        }
    }
    
    // Normalize kernel
    cv::Scalar sum = cv::sum(kernel);
    if (sum[0] != 0) {
        kernel /= sum[0];
    }
    
    return kernel;
}

// Quality assessment
ImageQualityMetrics ImagePreprocessor::assessQuality(const cv::Mat& image) {
    ImageQualityMetrics metrics;
    
    metrics.sharpness = computeSharpness(image);
    metrics.contrast = computeContrast(image);
    metrics.brightness = computeBrightness(image);
    metrics.noise_level = estimateNoise(image);
    metrics.snr = computeSNR(image);
    metrics.focus_measure = computeFocusMeasure(image);
    
    metrics.is_blurred = isBlurred(image, 100.0f);
    metrics.is_overexposed = isOverexposed(image, 0.95f);
    metrics.is_underexposed = isUnderexposed(image, 0.05f);
    
    // Compute overall quality (0-100)
    float quality = 0.0f;
    
    // Sharpness contribution (0-30)
    quality += std::min(30.0f, metrics.sharpness / 10.0f);
    
    // Contrast contribution (0-25)
    quality += metrics.contrast * 25.0f;
    
    // Brightness contribution (0-20)
    float bright_score = 20.0f * (1.0f - std::abs(metrics.brightness - 127.5f) / 127.5f);
    quality += bright_score;
    
    // SNR contribution (0-15)
    quality += std::min(15.0f, metrics.snr / 3.0f);
    
    // Focus contribution (0-10)
    quality += std::min(10.0f, metrics.focus_measure);
    
    // Penalties
    if (metrics.is_blurred) quality *= 0.7f;
    if (metrics.is_overexposed || metrics.is_underexposed) quality *= 0.8f;
    
    metrics.overall_quality = quality;
    
    return metrics;
}

float ImagePreprocessor::computeSharpness(const cv::Mat& image) {
    cv::Mat lap;
    cv::Laplacian(image, lap, CV_64F);
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev);
    
    return static_cast<float>(stddev[0] * stddev[0]);
}

float ImagePreprocessor::computeContrast(const cv::Mat& image) {
    cv::Scalar mean, stddev;
    cv::meanStdDev(image, mean, stddev);
    
    // RMS contrast
    return static_cast<float>(stddev[0] / (mean[0] + 1e-6));
}

float ImagePreprocessor::computeBrightness(const cv::Mat& image) {
    return static_cast<float>(cv::mean(image)[0]);
}

float ImagePreprocessor::estimateNoise(const cv::Mat& image) {
    cv::Mat lap;
    cv::Laplacian(image, lap, CV_32F);
    
    cv::Mat abs_lap = cv::abs(lap);
    
    cv::Scalar mean = cv::mean(abs_lap);
    
    // Robust noise estimation
    float sigma = static_cast<float>(mean[0]) / 0.6745f;
    
    return sigma;
}

float ImagePreprocessor::computeSNR(const cv::Mat& image) {
    float signal = computeBrightness(image);
    float noise = estimateNoise(image);
    
    if (noise < 1e-6f) return 100.0f;
    
    return 20.0f * std::log10(signal / noise);
}

float ImagePreprocessor::computeFocusMeasure(const cv::Mat& image) {
    // Tenengrad focus measure
    cv::Mat gx, gy;
    cv::Sobel(image, gx, CV_32F, 1, 0, 3);
    cv::Sobel(image, gy, CV_32F, 0, 1, 3);
    
    cv::Mat gradient = gx.mul(gx) + gy.mul(gy);
    
    return static_cast<float>(cv::mean(gradient)[0]);
}

bool ImagePreprocessor::isBlurred(const cv::Mat& image, float threshold) {
    return computeSharpness(image) < threshold;
}

bool ImagePreprocessor::isOverexposed(const cv::Mat& image, float threshold) {
    int total_pixels = image.rows * image.cols;
    int bright_pixels = cv::countNonZero(image > 250);
    
    return (static_cast<float>(bright_pixels) / total_pixels) > threshold * 0.1f;
}

bool ImagePreprocessor::isUnderexposed(const cv::Mat& image, float threshold) {
    int total_pixels = image.rows * image.cols;
    int dark_pixels = cv::countNonZero(image < 5);
    
    return (static_cast<float>(dark_pixels) / total_pixels) > threshold * 0.1f;
}

// Color conversions
cv::Mat ImagePreprocessor::rgbToGray(const cv::Mat& image) {
    if (image.channels() == 1) return image.clone();
    
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

cv::Mat ImagePreprocessor::grayToRGB(const cv::Mat& image) {
    if (image.channels() == 3) return image.clone();
    
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_GRAY2BGR);
    return rgb;
}

bool ImagePreprocessor::validateImage(const cv::Mat& image) {
    return !image.empty() && (image.type() == CV_8UC1 || image.type() == CV_8UC3);
}

std::string ImagePreprocessor::getPreprocessingReport() const {
    std::stringstream ss;
    ss << "=== Image Preprocessing Statistics ===" << std::endl;
    ss << "Total Processed: " << stats_.total_processed << std::endl;
    ss << "Failed: " << stats_.failed_processing << std::endl;
    ss << "Average Quality: " << stats_.avg_quality << std::endl;
    ss << "Average Processing Time: " << stats_.avg_processing_time_ms << " ms" << std::endl;
    
    return ss.str();
}
