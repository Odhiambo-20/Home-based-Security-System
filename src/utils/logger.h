// logger.h - Industrial Grade Logging System
// Thread-safe, high-performance logging with multiple outputs
// Copyright (c) 2025 Biometric Security Systems
// Version: 2.0.0 - Production Release

#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <fstream>
#include <memory>
#include <mutex>
#include <vector>
#include <queue>
#include <thread>
#include <condition_variable>
#include <chrono>
#include <sstream>
#include <atomic>
#include <functional>
#include <unordered_map>

// Log levels
enum class LogLevel {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARNING = 3,
    ERROR = 4,
    CRITICAL = 5,
    FATAL = 6
};

// Log output destinations
enum class LogOutput {
    CONSOLE,
    FILE,
    SYSLOG,
    NETWORK,
    CALLBACK
};

// Log entry structure
struct LogEntry {
    std::chrono::system_clock::time_point timestamp;
    LogLevel level;
    std::string message;
    std::string file;
    int line;
    std::string function;
    std::thread::id thread_id;
    std::string category;
    std::unordered_map<std::string, std::string> metadata;
    
    LogEntry() : line(0), level(LogLevel::INFO) {}
};

// Log formatter interface
class ILogFormatter {
public:
    virtual ~ILogFormatter() = default;
    virtual std::string format(const LogEntry& entry) = 0;
};

// Standard log formatter
class StandardFormatter : public ILogFormatter {
public:
    StandardFormatter(bool include_timestamp = true,
                     bool include_level = true,
                     bool include_thread = false,
                     bool include_location = false);
    
    std::string format(const LogEntry& entry) override;
    void setDateFormat(const std::string& format);
    void setColorEnabled(bool enabled);
    
private:
    bool include_timestamp_;
    bool include_level_;
    bool include_thread_;
    bool include_location_;
    bool color_enabled_;
    std::string date_format_;
    
    std::string getColorCode(LogLevel level);
    std::string resetColor();
    std::string levelToString(LogLevel level);
};

// JSON log formatter
class JsonFormatter : public ILogFormatter {
public:
    JsonFormatter(bool pretty_print = false);
    std::string format(const LogEntry& entry) override;
    
private:
    bool pretty_print_;
    std::string escapeJson(const std::string& str);
};

// Log sink interface
class ILogSink {
public:
    virtual ~ILogSink() = default;
    virtual void write(const LogEntry& entry) = 0;
    virtual void flush() = 0;
    virtual bool isEnabled() const = 0;
    virtual void setEnabled(bool enabled) = 0;
    virtual void setMinLevel(LogLevel level) = 0;
};

// Console sink
class ConsoleSink : public ILogSink {
public:
    ConsoleSink(std::shared_ptr<ILogFormatter> formatter);
    ~ConsoleSink() override = default;
    
    void write(const LogEntry& entry) override;
    void flush() override;
    bool isEnabled() const override { return enabled_; }
    void setEnabled(bool enabled) override { enabled_ = enabled; }
    void setMinLevel(LogLevel level) override { min_level_ = level; }
    void setOutputStream(std::ostream* stream);
    
private:
    std::shared_ptr<ILogFormatter> formatter_;
    std::ostream* output_stream_;
    std::atomic<bool> enabled_;
    std::atomic<LogLevel> min_level_;
    mutable std::mutex write_mutex_;
};

// File sink
class FileSink : public ILogSink {
public:
    FileSink(const std::string& filepath, 
             std::shared_ptr<ILogFormatter> formatter,
             bool append = true);
    ~FileSink() override;
    
    void write(const LogEntry& entry) override;
    void flush() override;
    bool isEnabled() const override { return enabled_ && file_.is_open(); }
    void setEnabled(bool enabled) override { enabled_ = enabled; }
    void setMinLevel(LogLevel level) override { min_level_ = level; }
    
    // Rotation settings
    void setMaxFileSize(size_t max_bytes);
    void setMaxFiles(int max_files);
    void enableRotation(bool enable);
    void rotateNow();
    
    std::string getFilePath() const { return filepath_; }
    size_t getCurrentSize() const { return current_size_; }
    
private:
    std::string filepath_;
    std::ofstream file_;
    std::shared_ptr<ILogFormatter> formatter_;
    std::atomic<bool> enabled_;
    std::atomic<LogLevel> min_level_;
    mutable std::mutex write_mutex_;
    
    // Rotation
    bool rotation_enabled_;
    size_t max_file_size_;
    int max_files_;
    size_t current_size_;
    
    void checkRotation();
    void performRotation();
    std::string getRotatedFileName(int index);
};

// Syslog sink (Unix/Linux)
class SyslogSink : public ILogSink {
public:
    SyslogSink(const std::string& ident, int facility = 0);
    ~SyslogSink() override;
    
    void write(const LogEntry& entry) override;
    void flush() override;
    bool isEnabled() const override { return enabled_; }
    void setEnabled(bool enabled) override { enabled_ = enabled; }
    void setMinLevel(LogLevel level) override { min_level_ = level; }
    
private:
    std::string ident_;
    std::atomic<bool> enabled_;
    std::atomic<LogLevel> min_level_;
    bool syslog_opened_;
    
    int levelToSyslogPriority(LogLevel level);
};

// Callback sink
class CallbackSink : public ILogSink {
public:
    using CallbackFunc = std::function<void(const LogEntry&)>;
    
    CallbackSink(CallbackFunc callback);
    ~CallbackSink() override = default;
    
    void write(const LogEntry& entry) override;
    void flush() override;
    bool isEnabled() const override { return enabled_; }
    void setEnabled(bool enabled) override { enabled_ = enabled; }
    void setMinLevel(LogLevel level) override { min_level_ = level; }
    
private:
    CallbackFunc callback_;
    std::atomic<bool> enabled_;
    std::atomic<LogLevel> min_level_;
};

// Main Logger class
class Logger {
public:
    // Singleton access
    static Logger& getInstance();
    
    // Initialization
    bool initialize(const std::string& config_file = "");
    void shutdown();
    bool isInitialized() const { return initialized_.load(); }
    
    // Sink management
    void addSink(std::shared_ptr<ILogSink> sink);
    void removeSink(std::shared_ptr<ILogSink> sink);
    void clearSinks();
    size_t getSinkCount() const;
    
    // Configuration
    void setGlobalMinLevel(LogLevel level);
    LogLevel getGlobalMinLevel() const { return global_min_level_; }
    void setAsyncMode(bool async);
    bool isAsyncMode() const { return async_mode_; }
    void setQueueSize(size_t size);
    void enableCategoryFiltering(bool enable);
    void setAllowedCategories(const std::vector<std::string>& categories);
    void setBlockedCategories(const std::vector<std::string>& categories);
    
    // Logging methods
    void log(LogLevel level, const std::string& message,
            const char* file = nullptr, int line = 0,
            const char* function = nullptr,
            const std::string& category = "");
    
    void logWithMetadata(LogLevel level, const std::string& message,
                        const std::unordered_map<std::string, std::string>& metadata,
                        const char* file = nullptr, int line = 0,
                        const char* function = nullptr,
                        const std::string& category = "");
    
    // Convenience methods
    static void trace(const std::string& message, const std::string& category = "");
    static void debug(const std::string& message, const std::string& category = "");
    static void info(const std::string& message, const std::string& category = "");
    static void warning(const std::string& message, const std::string& category = "");
    static void error(const std::string& message, const std::string& category = "");
    static void critical(const std::string& message, const std::string& category = "");
    static void fatal(const std::string& message, const std::string& category = "");
    
    // Flush all sinks
    void flush();
    
    // Statistics
    struct Statistics {
        std::atomic<uint64_t> total_entries{0};
        std::atomic<uint64_t> trace_count{0};
        std::atomic<uint64_t> debug_count{0};
        std::atomic<uint64_t> info_count{0};
        std::atomic<uint64_t> warning_count{0};
        std::atomic<uint64_t> error_count{0};
        std::atomic<uint64_t> critical_count{0};
        std::atomic<uint64_t> fatal_count{0};
        std::atomic<uint64_t> dropped_entries{0};
        std::chrono::system_clock::time_point start_time;
        
        void reset() {
            total_entries = 0;
            trace_count = 0;
            debug_count = 0;
            info_count = 0;
            warning_count = 0;
            error_count = 0;
            critical_count = 0;
            fatal_count = 0;
            dropped_entries = 0;
            start_time = std::chrono::system_clock::now();
        }
    };
    
    Statistics getStatistics() const { return stats_; }
    void resetStatistics() { stats_.reset(); }
    
private:
    Logger();
    ~Logger();
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    // Async processing
    void processQueue();
    void enqueueEntry(const LogEntry& entry);
    
    // Filtering
    bool shouldLog(LogLevel level, const std::string& category) const;
    
    // Member variables
    std::atomic<bool> initialized_;
    std::atomic<bool> shutdown_requested_;
    std::atomic<bool> async_mode_;
    std::atomic<LogLevel> global_min_level_;
    
    std::vector<std::shared_ptr<ILogSink>> sinks_;
    mutable std::mutex sinks_mutex_;
    
    // Async queue
    std::queue<LogEntry> log_queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::thread worker_thread_;
    size_t max_queue_size_;
    
    // Category filtering
    bool category_filtering_enabled_;
    std::vector<std::string> allowed_categories_;
    std::vector<std::string> blocked_categories_;
    mutable std::mutex category_mutex_;
    
    // Statistics
    Statistics stats_;
};

// Convenience macros
#define LOG_TRACE(msg) Logger::getInstance().log(LogLevel::TRACE, msg, __FILE__, __LINE__, __FUNCTION__)
#define LOG_DEBUG(msg) Logger::getInstance().log(LogLevel::DEBUG, msg, __FILE__, __LINE__, __FUNCTION__)
#define LOG_INFO(msg) Logger::getInstance().log(LogLevel::INFO, msg, __FILE__, __LINE__, __FUNCTION__)
#define LOG_WARNING(msg) Logger::getInstance().log(LogLevel::WARNING, msg, __FILE__, __LINE__, __FUNCTION__)
#define LOG_ERROR(msg) Logger::getInstance().log(LogLevel::ERROR, msg, __FILE__, __LINE__, __FUNCTION__)
#define LOG_CRITICAL(msg) Logger::getInstance().log(LogLevel::CRITICAL, msg, __FILE__, __LINE__, __FUNCTION__)
#define LOG_FATAL(msg) Logger::getInstance().log(LogLevel::FATAL, msg, __FILE__, __LINE__, __FUNCTION__)

// Category logging macros
#define LOG_TRACE_CAT(cat, msg) Logger::getInstance().log(LogLevel::TRACE, msg, __FILE__, __LINE__, __FUNCTION__, cat)
#define LOG_DEBUG_CAT(cat, msg) Logger::getInstance().log(LogLevel::DEBUG, msg, __FILE__, __LINE__, __FUNCTION__, cat)
#define LOG_INFO_CAT(cat, msg) Logger::getInstance().log(LogLevel::INFO, msg, __FILE__, __LINE__, __FUNCTION__, cat)
#define LOG_WARNING_CAT(cat, msg) Logger::getInstance().log(LogLevel::WARNING, msg, __FILE__, __LINE__, __FUNCTION__, cat)
#define LOG_ERROR_CAT(cat, msg) Logger::getInstance().log(LogLevel::ERROR, msg, __FILE__, __LINE__, __FUNCTION__, cat)

// Scoped logger for function entry/exit
class ScopedLogger {
public:
    ScopedLogger(const std::string& function_name, 
                 const char* file = nullptr, 
                 int line = 0);
    ~ScopedLogger();
    
    void addMetadata(const std::string& key, const std::string& value);
    
private:
    std::string function_name_;
    const char* file_;
    int line_;
    std::chrono::high_resolution_clock::time_point start_time_;
    std::unordered_map<std::string, std::string> metadata_;
};

#define SCOPED_LOG() ScopedLogger __scoped_logger__(__FUNCTION__, __FILE__, __LINE__)

// Performance logger
class PerformanceLogger {
public:
    PerformanceLogger(const std::string& operation_name);
    ~PerformanceLogger();
    
    void checkpoint(const std::string& name);
    void addMetric(const std::string& name, double value);
    
private:
    std::string operation_name_;
    std::chrono::high_resolution_clock::time_point start_time_;
    std::vector<std::pair<std::string, std::chrono::high_resolution_clock::time_point>> checkpoints_;
    std::unordered_map<std::string, double> metrics_;
};

#define PERF_LOG(name) PerformanceLogger __perf_logger__(name)

#endif // LOGGER_H
