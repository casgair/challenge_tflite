#include "model.hpp"

#include "opencv2/core.hpp"

#include "kl_error.hpp"

KLError Model::init(const char *model_path) {
    // Implementation missing...
    return KLError::NONE;
}

float Model::inference(const char *img_path) {
    // Implementation missing...
    return 0.0;
}

void Model::convert_image(const cv::Mat &src, float *dest) {
    // Implementation missing...
}
