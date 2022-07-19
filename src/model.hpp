#ifndef KL_CHALLENGE_MODEL_H
#define KL_CHALLENGE_MODEL_H

#include "kl_error.hpp"

#include "opencv2/core.hpp"

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"

class Model {

public:

    KLError init(const char *model_path);
    float inference(const char *img_path);

private:

    static const int CHANNELS = 3;
    static const int HEIGHT = 80;
    static const int WIDTH = 80;

    void convert_image(const cv::Mat &src, float *dest);
    static void softmax(float* input, size_t size);

    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
};

#endif //KL_CHALLENGE_MODEL_H
