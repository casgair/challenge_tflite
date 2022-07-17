#include "../src/model.hpp"
#include "../src/logger.hpp"

int main() {

    Model instance;

    // Initialize TFLite model
    // Current folder is PROJECT_ROOT/test
    instance.init("../assets/models/2.7_80x80_MiniFASNetV2.tflite");

    std::vector<std::string> test_images = {
        "fixtures/image_F1.bmp",
        "fixtures/image_F2.bmp",
        "fixtures/image_T1.bmp"
    };

    for(auto img_path : test_images) {
        float res = instance.inference(img_path.c_str());
        Logger::info("Genuine score for " + img_path + " is: " + std::to_string(res));
    }

    return 0;
}