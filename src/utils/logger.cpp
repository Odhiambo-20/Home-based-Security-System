// logger.cpp - Complete Industrial Logger Implementation
// Copyright (c) 2025 Biometric Security Systems

#include "logger.h"
#include <iostream>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <algorithm>
#include <sys/stat.h>

#ifdef __unix__
#include <syslog.h>
#endif

// ============================================================================
// STANDARD FORMATTER IMPLEMENTATION
// ============================================================================

StandardFormatter::StandardFormatter(bool include_timestamp, bool include_level,
                                    bool include_thread, bool include_location)
    : include_timestamp_(include_timestamp)
    , include_level_(include_level)
    , include_thread_(include_thread)
    , include_location_(include_location)
    , color_enabled_(true)
    , date_format_("%Y-%m-%d %H:%M:%S")
{}

std::string StandardFormatter::format(const LogEntry& entry) {
    std::ostringstream oss;
    
    // Timestamp
    if (include_timestamp_) {
        auto time_t_value = std::chrono::system_clock::to_time_t(entry.timestamp);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            entry.timestamp.time_since_epoch()) % 1000;
        
        std::tm tm_buf;
        localtime_r(&time_t_value, &tm_buf);
        
        oss << "[" << std::put_time(&tm_buf, date_format_.c_str())
            << "." << std::setfill('0') << std::setw(3) << ms.count() << "] ";
    }
    
    // Log level with color
    if (include_level_) {
        if (color_enabled_) {
            oss << getColorCode(entry.level);
        }
        oss << "[" << std::setw(8) << std::left << levelToString(entry.level) << "]";
        if (color_enabled_) {
            oss << resetColor();
        }
        oss << " ";
    }
    
    // Thread ID
    if (include_thread_) {
        oss << "[Thread " << entry.thread_id << "] ";
    }
    
    // Category
    if (!entry.category.empty()) {
        oss << "[" << entry.category << "] ";
    }
    
    // Message
    oss << entry.message;
    
    // Location
    if (include_location_ && !entry.file.empty()) {
        oss << " (" << entry.file << ":" << entry.line;
        if (!entry.function.empty()) {
            oss << " in " << entry.function;
        }
        oss << ")";
    }
    
    // Metadata
    if (!entry.metadata.empty()) {
        oss << " {";
        bool first = true;
        for (const auto& pair : entry.metadata) {
            if (!first) oss << ", ";
            oss << pair.first << "=" << pair.second;
            first = false;
        }
        oss << "}";
    }
    
    return oss.str();
}

void StandardFormatter::setDateFormat(const std::string& format) {
    date_format_ = format;
}

void StandardFormatter::setColorEnabled(bool enabled) {
    color_enabled_ = enabled;
}

std::string StandardFormatter::getColorCode(LogLevel level) {
    switch (level) {
        case LogLevel::TRACE: return "\033[37m";      // White
        case LogLevel::DEBUG: return "\033[36m";      // Cyan
        case LogLevel::INFO: return "\033[32m";       // Green
        case LogLevel::WARNING: return "\033[33m";    // Yellow
        case LogLevel::ERROR: return "\033[31m";      // Red
        case LogLevel::CRITICAL: return "\033[35m";   // Magenta
        case LogLevel::FATAL: return "\033[1;31m";    // Bold Red
        default: return "";
    }
}

std::string StandardFormatter::resetColor() {
    return "\033[0m";
}

std::string StandardFormatter::levelToString(LogLevel level) {
    switch (level) {
        case LogLevel::TRACE: return "TRACE";
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::CRITICAL: return "CRITICAL";
        case LogLevel::FATAL: return "FATAL";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// JSON FORMATTER IMPLEMENTATION
// ============================================================================

JsonFormatter::JsonFormatter(bool pretty_print)
    : pretty_print_(pretty_print)
{}

std::string JsonFormatter::format(const LogEntry& entry) {
    std::ostringstream oss;
    std::string indent = pretty_print_ ? "  " : "";
    std::string newline = pretty_print_ ? "\n" : "";
    
    oss << "{" << newline;
    
    // Timestamp
    auto time_t_value = std::chrono::system_clock::to_time_t(entry.timestamp);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        entry.timestamp.time_since_epoch()) % 1000;
    
    std::tm tm_buf;
    gmtime_r(&time_t_value, &tm_buf);
    
    char time_str[100];
    strftime(time_str, sizeof(time_str), "%Y-%m-%dT%H:%M:%S", &tm_buf);
    
    oss << indent << "\"timestamp\": \"" << time_str << "." 
        << std::setfill('0') << std::setw(3) << ms.count() << "Z\"," << newline;
    
    // Level
    oss << indent << "\"level\": \"";
    switch (entry.level) {
        case LogLevel::TRACE: oss << "TRACE"; break;
        case LogLevel::DEBUG: oss << "DEBUG"; break;
        case LogLevel::INFO: oss << "INFO"; break;
        case LogLevel::WARNING: oss << "WARNING"; break;
        case LogLevel::ERROR: oss << "ERROR"; break;
        case LogLevel::CRITICAL: oss << "CRITICAL"; break;
        case LogLevel::FATAL: oss << "FATAL"; break;
    }
    oss << "\"," << newline;
    
    // Message
    oss << indent << "\"message\": \"" << escapeJson(entry.message) << "\"";
    
    // Category
    if (!entry.category.empty()) {
        oss << "," << newline << indent << "\"category\": \"" 
            << escapeJson(entry.category) << "\"";
    }
    
    // Thread ID
    oss << "," << newline << indent << "\"thread_id\": \"" << entry.thread_id << "\"";
    
    // Location
    if (!entry.file.empty()) {
        oss << "," << newline << indent << "\"file\": \"" 
            << escapeJson(entry.file) << "\"," << newline;
        oss << indent << "\"line\": " << entry.line;
        
        if (!entry.function.empty()) {
            oss << "," << newline << indent << "\"function\": \"" 
                << escapeJson(entry.function) << "\"";
        }
    }
    
    // Metadata
    if (!entry.metadata.empty()) {
        oss << "," << newline << indent << "\"metadata\": {" << newline;
        bool first = true;
        for (const auto& pair : entry.metadata) {
            if (!first) oss << "," << newline;
            oss << indent << indent << "\"" << escapeJson(pair.first) << "\": \"" 
                << escapeJson(pair.second) << "\"";
            first = false;
        }
        oss << newline << indent << "}";
    }
    
    oss << newline << "}";
    
    return oss.str();
}

std::string JsonFormatter::escapeJson(const std::string& str) {
    std::ostringstream oss;
    for (char c : str) {
        switch (c) {
            case '"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\b': oss << "\\b"; break;
            case '\f': oss << "\\f"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (c < 32) {
                    oss << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
                } else {
                    oss << c;
                }
        }
    }
    return oss.str();
}

// ============================================================================
// CONSOLE SINK IMPLEMENTATION
// ============================================================================

ConsoleSink::ConsoleSink(std::shared_ptr<ILogFormatter> formatter)
    : formatter_(formatter)
    , output_stream_(&std::cout)
    , enabled_(true)
    , min_level_(LogLevel::TRACE)
{}

void ConsoleSink::write(const LogEntry& entry) {
    if (!enabled_.load() || entry.level < min_level_.load()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(write_mutex_);
    std::string formatted = formatter_->format(entry);
    *output_stream_ << formatted << std::endl;
}

void ConsoleSink::flush() {
    std::lock_guard<std::mutex> lock(write_mutex_);
    output_stream_->flush();
}

void ConsoleSink::setOutputStream(std::ostream* stream) {
    std::lock_guard<std::mutex> lock(write_mutex_);
    output_stream_ = stream;
}

// ============================================================================
// FILE SINK IMPLEMENTATION
// ============================================================================

FileSink::FileSink(const std::string& filepath, 
                   std::shared_ptr<ILogFormatter> formatter,
                   bool append)
    : filepath_(filepath)
    , formatter_(formatter)
    , enabled_(true)
    , min_level_(LogLevel::TRACE)
    , rotation_enabled_(false)
    , max_file_size_(10 * 1024 * 1024)  // 10 MB default
    , max_files_(5)
    , current_size_(0)
{
    auto mode = append ? (std::ios::out | std::ios::app) : std::ios::out;
    file_.open(filepath_, mode);
    
    if (file_.is_open() && append) {
        file_.seekp(0, std::ios::end);
        current_size_ = file_.tellp();
    }
}

FileSink::~FileSink() {
    if (file_.is_open()) {
        file_.close();
    }
}

void FileSink::write(const LogEntry& entry) {
    if (!enabled_.load() || !file_.is_open() || entry.level < min_level_.load()) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(write_mutex_);
    
    std::string formatted = formatter_->format(entry);
    file_ << formatted << std::endl;
    current_size_ += formatted.length() + 1;
    
    if (rotation_enabled_) {
        checkRotation();
    }
}

void FileSink::flush() {
    std::lock_guard<std::mutex> lock(write_mutex_);
    if (file_.is_open()) {
        file_.flush();
    }
}

void FileSink::setMaxFileSize(size_t max_bytes) {
    max_file_size_ = max_bytes;
}

void FileSink::setMaxFiles(int max_files) {
    max_files_ = max_files;
}

void FileSink::enableRotation(bool enable) {
    rotation_enabled_ = enable;
}

void FileSink::rotateNow() {
    std::lock_guard<std::mutex> lock(write_mutex_);
    performRotation();
}

void FileSink::checkRotation() {
    if (current_size_ >= max_file_size_) {
        performRotation();
    }
}

void FileSink::performRotation() {
    if (file_.is_open()) {
        file_.close();
    }
    
    // Rotate existing files
    for (int i = max_files_ - 1; i > 0; --i) {
        std::string old_name = getRotatedFileName(i - 1);
        std::string new_name = getRotatedFileName(i);
        std::rename(old_name.c_str(), new_name.c_str());
    }
    
    // Move current file to .1
    std::rename(filepath_.c_str(), getRotatedFileName(0).c_str());
    
    // Open new file
    file_.open(filepath_, std::ios::out);
    current_size_ = 0;
}

std::string FileSink::getRotatedFileName(int index) {
    return filepath_ + "." + std::to_string(index + 1);
}

// ============================================================================
// SYSLOG SINK IMPLEMENTATION
// ============================================================================

#ifdef __unix__
SyslogSink::SyslogSink(const std::string& ident, int facility)
    : ident_(ident)
    , enabled_(true)
    , min_level_(LogLevel::TRACE)
    , syslog_opened_(false)
{
    openlog(ident_.c_str(), LOG_PID | LOG_CONS, facility);
    syslog_opened_ = true;
}

SyslogSink::~SyslogSink() {
    if (syslog_opened_) {
        closelog();
    }
}

void SyslogSink::write(const LogEntry& entry) {
    if (!enabled_.load() || !syslog_opened_ || entry.level < min_level_.load()) {
        return;
    }
    
    int priority = levelToSyslogPriority(entry.level);
    syslog(priority, "%s", entry.message.c_str());
}

void SyslogSink::flush() {
    // Syslog auto-flushes
}

int SyslogSink::levelToSyslogPriority(LogLevel level) {
    switch (level) {
        case LogLevel::TRACE:
        case LogLevel::DEBUG: return LOG_DEBUG;
        case LogLevel::INFO: return LOG_INFO;
        case LogLevel::WARNING: return LOG_WARNING;
        case LogLevel::ERROR: return LOG_ERR;
        case LogLevel::CRITICAL: return LOG_CRIT;
        case LogLevel::FATAL: return LOG_ALERT;
        default: return LOG_INFO;
    }
}
#else
SyslogSink::SyslogSink(const std::string& ident, int facility)
    : ident_(ident), enabled_(false), min_level_(LogLevel::TRACE), syslog_opened_(false) {}
SyslogSink::~SyslogSink() {}
void SyslogSink::write(const LogEntry&) {}
void SyslogSink::flush() {}
int SyslogSink::levelToSyslogPriority(LogLevel) { return 0; }
#endif

// ============================================================================
// CALLBACK SINK IMPLEMENTATION
// ============================================================================

CallbackSink::CallbackSink(CallbackFunc callback)
    : callback_(callback)
    , enabled_(true)
    , min_level_(LogLevel::TRACE)
{}

void CallbackSink::write(const LogEntry& entry) {
    if (!enabled_.load() || entry.level < min_level_.load()) {
        return;
    }
    
    if (callback_) {
        callback_(entry);
    }
}

void CallbackSink::flush() {
    // Callbacks flush immediately
}

// ============================================================================
// LOGGER IMPLEMENTATION
// ============================================================================

Logger::Logger()
    : initialized_(false)
    , shutdown_requested_(false)
    , async_mode_(false)
    , global_min_level_(LogLevel::INFO)
    , max_queue_size_(10000)
    , category_filtering_enabled_(false)
{
    stats_.reset();
}

Logger::~Logger() {
    shutdown();
}

Logger& Logger::getInstance() {
    static Logger instance;
    return instance;
}

bool Logger::initialize(const std::string& config_file) {
    if (initialized_.load()) {
        return true;
    }
    
    std::lock_guard<std::mutex> lock(sinks_mutex_);
    
    // Create default console sink
    auto console_formatter = std::make_shared<StandardFormatter>(true, true, false, false);
    auto console_sink = std::make_shared<ConsoleSink>(console_formatter);
    sinks_.push_back(console_sink);
    
    // Create default file sink
    auto file_formatter = std::make_shared<StandardFormatter>(true, true, true, true);
    auto file_sink = std::make_shared<FileSink>("/var/log/biometric_security.log", file_formatter, true);
    file_sink->setMaxFileSize(50 * 1024 * 1024);  // 50 MB
    file_sink->setMaxFiles(10);
    file_sink->enableRotation(true);
    sinks_.push_back(file_sink);
    
    // Start async worker if enabled
    if (async_mode_.load()) {
        worker_thread_ = std::thread(&Logger::processQueue, this);
    }
    
    initialized_ = true;
    stats_.start_time = std::chrono::system_clock::now();
    
    info("Logger initialized successfully");
    
    return true;
}

void Logger::shutdown() {
    if (!initialized_.load()) {
        return;
    }
    
    info("Logger shutting down");
    
    shutdown_requested_ = true;
    
    if (async_mode_.load() && worker_thread_.joinable()) {
        queue_cv_.notify_all();
        worker_thread_.join();
    }
    
    flush();
    
    std::lock_guard<std::mutex> lock(sinks_mutex_);
    sinks_.clear();
    
    initialized_ = false;
}

void Logger::addSink(std::shared_ptr<ILogSink> sink) {
    std::lock_guard<std::mutex> lock(sinks_mutex_);
    sinks_.push_back(sink);
}

void Logger::removeSink(std::shared_ptr<ILogSink> sink) {
    std::lock_guard<std::mutex> lock(sinks_mutex_);
    sinks_.erase(std::remove(sinks_.begin(), sinks_.end(), sink), sinks_.end());
}

void Logger::clearSinks() {
    std::lock_guard<std::mutex> lock(sinks_mutex_);
    sinks_.clear();
}

size_t Logger::getSinkCount() const {
    std::lock_guard<std::mutex> lock(sinks_mutex_);
    return sinks_.size();
}

void Logger::setGlobalMinLevel(LogLevel level) {
    global_min_level_ = level;
}

void Logger::setAsyncMode(bool async) {
    if (async == async_mode_.load()) {
        return;
    }
    
    async_mode_ = async;
    
    if (async && initialized_.load() && !worker_thread_.joinable()) {
        worker_thread_ = std::thread(&Logger::processQueue, this);
    }
}

void Logger::setQueueSize(size_t size) {
    max_queue_size_ = size;
}

void Logger::enableCategoryFiltering(bool enable) {
    category_filtering_enabled_ = enable;
}

void Logger::setAllowedCategories(const std::vector<std::string>& categories) {
    std::lock_guard<std::mutex> lock(category_mutex_);
    allowed_categories_ = categories;
}

void Logger::setBlockedCategories(const std::vector<std::string>& categories) {
    std::lock_guard<std::mutex> lock(category_mutex_);
    blocked_categories_ = categories;
}

void Logger::log(LogLevel level, const std::string& message,
                const char* file, int line, const char* function,
                const std::string& category) {
    logWithMetadata(level, message, {}, file, line, function, category);
}

void Logger::logWithMetadata(LogLevel level, const std::string& message,
                            const std::unordered_map<std::string, std::string>& metadata,
                            const char* file, int line, const char* function,
                            const std::string& category) {
    if (!shouldLog(level, category)) {
        return;
    }
    
    LogEntry entry;
    entry.timestamp = std::chrono::system_clock::now();
    entry.level = level;
    entry.message = message;
    entry.file = file ? file : "";
    entry.line = line;
    entry.function = function ? function : "";
    entry.thread_id = std::this_thread::get_id();
    entry.category = category;
    entry.metadata = metadata;
    
    // Update statistics
    stats_.total_entries++;
    switch (level) {
        case LogLevel::TRACE: stats_.trace_count++; break;
        case LogLevel::DEBUG: stats_.debug_count++; break;
        case LogLevel::INFO: stats_.info_count++; break;
        case LogLevel::WARNING: stats_.warning_count++; break;
        case LogLevel::ERROR: stats_.error_count++; break;
        case LogLevel::CRITICAL: stats_.critical_count++; break;
        case LogLevel::FATAL: stats_.fatal_count++; break;
    }
    
    if (async_mode_.load()) {
        enqueueEntry(entry);
    } else {
        std::lock_guard<std::mutex> lock(sinks_mutex_);
        for (auto& sink : sinks_) {
            if (sink && sink->isEnabled()) {
                sink->write(entry);
            }
        }
    }
}

void Logger::trace(const std::string& message, const std::string& category) {
    getInstance().log(LogLevel::TRACE, message, nullptr, 0, nullptr, category);
}

void Logger::debug(const std::string& message, const std::string& category) {
    getInstance().log(LogLevel::DEBUG, message, nullptr, 0, nullptr, category);
}

void Logger::info(const std::string& message, const std::string& category) {
    getInstance().log(LogLevel::INFO, message, nullptr, 0, nullptr, category);
}

void Logger::warning(const std::string& message, const std::string& category) {
    getInstance().log(LogLevel::WARNING, message, nullptr, 0, nullptr, category);
}

void Logger::error(const std::string& message, const std::string& category) {
    getInstance().log(LogLevel::ERROR, message, nullptr, 0, nullptr, category);
}

void Logger::critical(const std::string& message, const std::string& category) {
    getInstance().log(LogLevel::CRITICAL, message, nullptr, 0, nullptr, category);
}

void Logger::fatal(const std::string& message, const std::string& category) {
    getInstance().log(LogLevel::FATAL, message, nullptr, 0, nullptr, category);
}

void Logger::flush() {
    std::lock_guard<std::mutex> lock(sinks_mutex_);
    for (auto& sink : sinks_) {
        if (sink && sink->isEnabled()) {
            sink->flush();
        }
    }
}

void Logger::processQueue() {
    while (!shutdown_requested_.load()) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait_for(lock, std::chrono::milliseconds(100), 
                          [this] { return !log_queue_.empty() || shutdown_requested_.load(); });
        
        while (!log_queue_.empty()) {
            LogEntry entry = log_queue_.front();
            log_queue_.pop();
            lock.unlock();
            
            std::lock_guard<std::mutex> sinks_lock(sinks_mutex_);
            for (auto& sink : sinks_) {
                if (sink && sink->isEnabled()) {
                    sink->write(entry);
                }
            }
            
            lock.lock();
        }
    }
    
    // Process remaining entries
    std::lock_guard<std::mutex> lock(queue_mutex_);
    while (!log_queue_.empty()) {
        LogEntry entry = log_queue_.front();
        log_queue_.pop();
        
        std::lock_guard<std::mutex> sinks_lock(sinks_mutex_);
        for (auto& sink : sinks_) {
            if (sink && sink->isEnabled()) {
                sink->write(entry);
            }
        }
    }
}

void Logger::enqueueEntry(const LogEntry& entry) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    
    if (log_queue_.size() >= max_queue_size_) {
        stats_.dropped_entries++;
        return;
    }
    
    log_queue_.push(entry);
    queue_cv_.notify_one();
}

bool Logger::shouldLog(LogLevel level, const std::string& category) const {
    if (level < global_min_level_.load()) {
        return false;
    }
    
    if (!category_filtering_enabled_ || category.empty()) {
        return true;
    }
    
    std::lock_guard<std::mutex> lock(category_mutex_);
    
    // Check blocked categories
    if (std::find(blocked_categories_.begin(), blocked_categories_.end(), category) 
        != blocked_categories_.end()) {
        return false;
    }
    
    // Check allowed categories
    if (!allowed_categories_.empty()) {
        return std::find(allowed_categories_.begin(), allowed_categories_.end(), category) 
               != allowed_categories_.end();
    }
    
    return true;
}

// ============================================================================
// SCOPED LOGGER IMPLEMENTATION
// ============================================================================

ScopedLogger::ScopedLogger(const std::string& function_name, 
                           const char* file, int line)
    : function_name_(function_name)
    , file_(file)
    , line_(line)
    , start_time_(std::chrono::high_resolution_clock::now())
{
    Logger::getInstance().log(LogLevel::TRACE, 
                             "Entering " + function_name_, 
                             file_, line_, function_name_.c_str());
}

ScopedLogger::~ScopedLogger() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time_).count();
    
    std::unordered_map<std::string, std::string> metadata = metadata_;
    metadata["duration_us"] = std::to_string(duration);
    
    Logger::getInstance().logWithMetadata(LogLevel::TRACE,
                                         "Exiting " + function_name_,
                                         metadata,
                                         file_, line_, function_name_.c_str());
}

void ScopedLogger::addMetadata(const std::string& key, const std::string& value) {
    metadata_[key] = value;
}

// ============================================================================
// PERFORMANCE LOGGER IMPLEMENTATION
// ============================================================================

PerformanceLogger::PerformanceLogger(const std::string& operation_name)
    : operation_name_(operation_name)
    , start_time_(std::chrono::high_resolution_clock::now())
{}

PerformanceLogger::~PerformanceLogger() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time_).count();
    
    std::ostringstream oss;
    oss << "Performance: " << operation_name_ << " took " 
        << (total_duration / 1000.0) << "ms";
    
    if (!checkpoints_.empty()) {
        oss << " [";
        for (size_t i = 0; i < checkpoints_.size(); ++i) {
            if (i > 0) oss << ", ";
            
            auto checkpoint_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                checkpoints_[i].second - start_time_).count();
            
            oss << checkpoints_[i].first << ": " << (checkpoint_duration / 1000.0) << "ms";
        }
        oss << "]";
    }
    
    if (!metrics_.empty()) {
        oss << " Metrics: {";
        bool first = true;
        for (const auto& metric : metrics_) {
            if (!first) oss << ", ";
            oss << metric.first << ": " << metric.second;
            first = false;
        }
        oss << "}";
    }
    
    Logger::getInstance().log(LogLevel::DEBUG, oss.str());
}

void PerformanceLogger::checkpoint(const std::string& name) {
    checkpoints_.emplace_back(name, std::chrono::high_resolution_clock::now());
}

void PerformanceLogger::addMetric(const std::string& name, double value) {
    metrics_[name] = value;
}
