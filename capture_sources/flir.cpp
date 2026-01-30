/**
 * @file flir.cpp
 * @author Jeffrey Qian
 * @brief 
 * @version 1.0
 * @date 2025-03-01
 * 
 * 
 * I read the Flir documentation https://www.teledynevisionsolutions.com/products/spinnaker-sdk/?model=Spinnaker%20SDK&vertical=machine%20vision&segment=iis
 * so you don't have to.
 * 
 * Every GenICam compliant camera has an XML description file which contains camera features, and min/max camera parameters. 
 * The elements of a camera description file are represented in software objects called **Nodes**. A **Node Map** is a list
 * of nodes created dynamically at runtime.
 * 
 */

#include <filesystem>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>

#include <Spinnaker.h>

#include "libshm/c/shm.h"
#include "fmt/format.h"
#include "toml/toml.h"
#include "conf/parse.hpp"
#include "include/capture_source.hpp"

namespace fs = std::filesystem;

struct FlirConfig {
    const std::string serial_number;
    const std::string direction;
    const uint32_t width;
    const uint32_t height;
    const uint32_t fps;
    const uint32_t binning_factor_horizontal;
    const uint32_t binning_factor_vertical;


    FlirConfig(std::string&& serial_number, std::string&& direction, const uint32_t width, const uint32_t height, const uint32_t fps, const uint32_t binning_factor_horizontal, const uint32_t binning_factor_vertical) 
    : serial_number(std::move(serial_number))
    , direction(std::move(direction))
    , width(width)
    , height(height)
    , fps(fps)
    , binning_factor_horizontal(binning_factor_horizontal) 
    , binning_factor_vertical(binning_factor_vertical) {}
};

std::vector<FlirConfig> parse_flir_config(const fs::path& toml_file) {
    const toml::ParseResult toml_result = toml::parseFile(toml_file);
    const toml::Value& root = toml_result.value;

    if (!root.find("camera")) {
        throw std::runtime_error("Expected '[[camera]]' array table in toml file");
    }

    const toml::Value val = *root.find("camera");
    if (!val.is<toml::Array>()) {
        throw std::runtime_error("Expected array of cameras.");
    }

    const toml::Array& arr = val.as<toml::Array>();
    std::vector<FlirConfig> config_arr;
    for(size_t i = 0; i < arr.size(); i++) {
        toml::Value root = arr[i];
        if (!root.is<toml::Table>()) {
            throw std::runtime_error("Expected camera config to be a table.");
        }

        PARSE_STRING(serial_number);
        PARSE_STRING(direction);
        PARSE_INTEGER(width);
        PARSE_INTEGER(height);
        PARSE_INTEGER(fps);
        PARSE_INTEGER(binning_factor_horizontal);
        PARSE_INTEGER(binning_factor_vertical);

        config_arr.emplace_back(std::move(serial_number), std::move(direction), width, height, fps, binning_factor_horizontal, binning_factor_vertical);
    }

    return config_arr;
}


class Flir {
private:
    Spinnaker::CameraPtr _cam;
    std::string _direction;
    uint32_t _fps;

public:
    Flir(Spinnaker::CameraPtr cam, const FlirConfig &config);
    ~Flir();

    inline Spinnaker::CameraPtr camera() const {return _cam;}
    inline const std::string& direction() const {return _direction;}
    inline const uint32_t& fps() const {return _fps;}

    Flir(const Flir& flir) = delete;
    Flir& operator=(const Flir& flir) = delete;
};

Flir::Flir(Spinnaker::CameraPtr cam, const FlirConfig &config): _cam(cam), _direction(config.direction), _fps(config.fps) { 
    // throw an error if at any part, camera initialization fails
    _cam->Init();

    _cam->Width.SetValue(config.width);
    _cam->Height.SetValue(config.height);

    _cam->BinningHorizontal.SetValue(config.binning_factor_horizontal);
    _cam->BinningVertical.SetValue(config.binning_factor_vertical);

    _cam->AcquisitionFrameRateEnable.SetValue(true);
    _cam->AcquisitionFrameRate.SetValue(config.fps);


    // ouput values
    uint32_t max_width = cam->WidthMax();
    uint32_t max_height = cam->HeightMax();
    uint32_t min_fps = cam->AcquisitionFrameRate.GetMin();
    uint32_t max_fps = cam->AcquisitionFrameRate.GetMax();
    fmt::print("Successfully initialized Flir '{}' for '{}'\n", config.serial_number, config.direction);
    fmt::print("   * Current resolution: {}x{}\n", _cam->Width(), _cam->Height());
    fmt::print("   * Max resolution: {}x{}\n", max_width, max_height);
    fmt::print("   * Current frame rate: {} fps\n", _cam->AcquisitionFrameRate());
    fmt::print("   * Supported frame rate: {}-{} fps\n", min_fps, max_fps);
    fmt::print("   * Horizontal binning {}\n", _cam->BinningHorizontal());
    fmt::print("   * Vertical binning {}\n", _cam->BinningVertical());

    _cam->BeginAcquisition();
}

Flir::~Flir() {
    std::cout << "destructor called" << std::endl;
    _cam->EndAcquisition();
    _cam->DeInit();
}

void flir_capture_udl(capture_source::CaptureSource &cs, capture_source::QuitFlag &quit_flag, std::shared_ptr<Flir> flir) {
    capture_source::FpsLimiter limiter(flir->fps());
	Spinnaker::ImageProcessor processor;
 
    // use bilinear over knn to reduce noise
	processor.SetColorProcessing(Spinnaker::SPINNAKER_COLOR_PROCESSING_ALGORITHM_BILINEAR);

    while(!quit_flag.load()) {
        uint64_t acquisition_time_ms = limiter.tick();
        Spinnaker::ImagePtr image = flir->camera()->GetNextImage(1000);

        if (image->IsIncomplete()) {
            // Retrieve and print the image status description
            std::cout << "Image incomplete: " << Spinnaker::Image::GetImageStatusDescription(image->GetImageStatus()) << std::endl;
            image->Release();
            continue;
        }

        Spinnaker::ImagePtr convertedImage = processor.Convert(image, Spinnaker::PixelFormat_BGR8);
        unsigned char *data = static_cast<unsigned char *>(convertedImage->GetData());
        cs.write_image<unsigned char>(
            flir->direction(),
            acquisition_time_ms,
            convertedImage->GetWidth(),
            convertedImage->GetHeight(),
            convertedImage->GetNumChannels(),
            data
        );
        image->Release();
    }
}

void flir_param_update_udl(capture_source::CaptureSource &cs, capture_source::QuitFlag &quit_flag, std::shared_ptr<Flir> flir) {
    capture_source::FpsLimiter limiter(1);
    watcher_t watcher = create_watcher();
    shm_watch(flir_calibration, watcher);
    flir->camera()->ExposureAuto.SetValue(Spinnaker::ExposureAutoEnums::ExposureAuto_Off);

    //Set exposure mode to "Timed"
    flir->camera()->ExposureMode.SetValue(Spinnaker::ExposureModeEnums::ExposureMode_Timed);
    while (!quit_flag) 
    {
        limiter.tick();
        if (watcher_has_changed(watcher)) {
            std::cout << "I have updated!!" << std::endl;
            flir->camera()->ExposureTime.SetValue(shm->flir_calibration.g.flir_exposure);
        }
    }

    destroy_watcher(watcher);
}

int main() {
    shm_init();

    // flir.cpp should work for an arbitrary number of flir cameras
    // as such, cameras are configured in vision/capture_sources/configs/flir.conf

    const char* software_path = std::getenv("CUAUV_SOFTWARE");
    if(software_path == nullptr) { 
        throw std::runtime_error("Please set environment variable 'CUAUV_SOFTWARE'");
    }
    const fs::path config_file = fs::path(std::getenv("CUAUV_SOFTWARE")) / "vision/capture_sources/configs/flir.conf";
    const std::vector<FlirConfig> flir_config = parse_flir_config(config_file);

    // run a first pass to detect all flir cameras on the system
    // and output camera capabilities
    Spinnaker::SystemPtr system = Spinnaker::System::GetInstance();
    Spinnaker::CameraList cam_list = system->GetCameras();

    std::cout << "Detect Flir cameras:" << std::endl;
    for(size_t i = 0; i < cam_list.GetSize(); i++) {
        Spinnaker::CameraPtr cam = cam_list.GetByIndex(0);
        Spinnaker::GenApi::CStringPtr serial_number_ptr = cam->GetTLDeviceNodeMap().GetNode("DeviceSerialNumber");
        std::string serial_number = std::string(serial_number_ptr->GetValue());
        fmt::print(" {}. Detected Flir camera with serial: {}\n", i + 1, serial_number);
    }
    std::cout<<std::endl;
    
    // bind each flir camera defined in flir.conf to an acquisition thread
    capture_source::CaptureSource cs;
    for(const auto& config : flir_config) {
        Spinnaker::CameraPtr cam = cam_list.GetBySerial(config.serial_number);
        std::shared_ptr<Flir> flir = std::make_shared<Flir>(cam, config);
        cs.register_udl(fmt::format("flir-{}", config.serial_number), flir_capture_udl, flir);
        cs.register_udl("test of configuration update", flir_param_update_udl, flir);
    }

    cs.run_until_complete();
    cam_list.Clear();
    system->ReleaseInstance();
    return 0;
}