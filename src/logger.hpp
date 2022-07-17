#ifndef KL_CHALLENGE_LOGGER_H
#define KL_CHALLENGE_LOGGER_H

#include <iostream>

class Logger {

private:

    typedef enum {
        KL_LOG_ERROR = 0,
        KL_LOG_WARNING,
        KL_LOG_INFO,
        KL_LOG_DEBUG
    } loglevel;

    static const loglevel current_level = KL_LOG_INFO;

    static void log(const loglevel level, const std::string &message) {
        const char *type;

        switch (level) {
            case KL_LOG_ERROR:
                type = "ERROR: ";
                break;
            case KL_LOG_WARNING:
                type = "WARNING: ";
                break;
            case KL_LOG_INFO:
                type = "INFO: ";
                break;
            case KL_LOG_DEBUG:
                type = "DEBUG: ";
                break;
        }

        std::cout << type << message << std::endl;
    }

public:
    static void error(const std::string &message) {
        log(KL_LOG_ERROR, message);
    }
    static void warn(const std::string &message) {
        log(KL_LOG_WARNING, message);
    }
    static void info(const std::string &message) {
        log(KL_LOG_INFO, message);
    }
    static void debug(const std::string &message) {
        log(KL_LOG_DEBUG, message);
    }
};

#endif //KL_CHALLENGE_LOGGER_H
