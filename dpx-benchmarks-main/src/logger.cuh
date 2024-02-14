#ifndef LOGGER_CUH
#define LOGGER_CUH

#include <iostream>

#define LOG_ERROR(str, ...) {\
    fprintf(stdout, "[ERROR] "); \
    fprintf(stdout, str, ##__VA_ARGS__); \
    fprintf(stdout, "\n"); \
}

#ifndef NO_PROGRESS_BAR

#define UPDATE_PROGRESS_BAR(txt, curr_txt, current, total) {\
    const int bar_width = 50; \
    const float progress = (float)(current) / (total); \
    const int position = progress * bar_width; \
    std::cout << txt << " ["; \
    for (int i=0; i<bar_width; i++) { \
        if (i < position) std::cout << "="; \
        else if (i == position) std::cout << ">"; \
        else std::cout << " "; \
    } \
    std::cout << "] " << (int)(progress * 100) << "% (" << curr_txt << ")            \r"; \
    std::cout.flush(); \
}

#define END_PROGRESS_BAR() std::cout << std::endl;

#else // SILENT
#define UPDATE_PROGRESS_BAR(txt, curr_txt, current, total);
#define END_PROGRESS_BAR();
#endif // SILENT

#endif // LOGGER_CUH