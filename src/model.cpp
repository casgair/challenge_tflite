#include <stdexcept>
#include <iostream>
#include "model.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"

using namespace cv;

KLError Model::init(const char *model_path) {

    // Load the model
    model_ = tflite::FlatBufferModel::BuildFromFile(model_path);

    // Return model loading error if model not loaded
    if(model_ == nullptr){
        return KLError::MODEL_LOAD_ERROR;
    }

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver_;
    tflite::InterpreterBuilder(*model_, resolver_)(&interpreter_);
    interpreter_->AllocateTensors();

    return KLError::NONE;
}

float Model::inference(const char *img_path) {

    // Read image
    Mat img = imread(img_path);

    if(img.empty())
    {
        throw std::invalid_argument("Could not read input image from disk.");
    }

    cv::Mat inputImg;
    // Transform image to float32, 3 channel format
    img.convertTo(inputImg, CV_32FC3);
    // Convert opencv image representation from HWC to CHW format, destination flat float array
    float imageCHW[this->CHANNELS*this->HEIGHT*this->WIDTH];
    this->convert_image(inputImg, imageCHW);

    // Input and output layers
    float* input = interpreter_->typed_input_tensor<float>(0);
    float* output = interpreter_->typed_output_tensor<float>(0);

    // Copy image to model input buffer
    memcpy(input, imageCHW, this->CHANNELS * this->HEIGHT * this->WIDTH * sizeof(float));

    // Do prediction
    int tflite_status = interpreter_->Invoke();

    // Basic error checking of model invokation
    if(tflite_status != 0){
        throw std::runtime_error("Invoking tflite model failed, check model input.");
    }

    // Calculate softmax values from prediction output
    this->softmax(output, this->CHANNELS);

    // Return only genuine class score (Real face score)
    return output[1];
}

void Model::convert_image(const cv::Mat &src, float * dest) {

    // I'm not too proud of this hacky and probably slow solution, I'm sure there is a better way to do this (or circumvent it)
    for(int i=0; i<this->HEIGHT;i++){
        for(int j=0; j<this->WIDTH;j++){
            float b = src.ptr<float>(i,j)[0];
            float g = src.ptr<float>(i,j)[1];
            float r = src.ptr<float>(i,j)[2];
            *(dest+i*src.cols+j) = b;
            *(dest+src.cols*src.rows+i*src.cols+j) = g;
            *(dest+2*src.cols*src.rows+i*src.cols+j)= r;
        }
    }
}

// Added here for this exercise, should probably be moved to utils file
// From here: https://slaystudy.com/implementation-of-softmax-activation-function-in-c-c/
void Model::softmax(float* input, size_t size) {

	assert(0 <= size <= sizeof(input) / sizeof(float));

	unsigned int i;
	float m, sum, constant;

	m = -INFINITY;
	for (i = 0; i < size; ++i) {
		if (m < input[i]) {
			m = input[i];
		}
	}

	sum = 0.0;
	for (i = 0; i < size; ++i) {
		sum += exp(input[i] - m);
	}

	constant = m + log(sum);
	for (i = 0; i < size; ++i) {
		input[i] = exp(input[i] - constant);
	}

}