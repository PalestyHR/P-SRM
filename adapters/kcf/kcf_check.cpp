#include <opencv2/tracking.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

struct Observation {
    bool available = false, resized = false;
    cv::Mat response;
    cv::Rect2d roi;
    cv::Rect candidate;
    cv::Point peak;
    double score = std::numeric_limits<double>::quiet_NaN();
    float threshold = 0;
};
static Observation obs;

static void observe(const cv::Mat& response, const cv::Rect2d& oldroi,
                    bool resized, const cv::Point& peak, double score,
                    float threshold, const cv::Size& imageSize) {
    obs.available = true;
    obs.response = response.clone();
    obs.roi = oldroi; obs.resized = resized; obs.peak = peak;
    obs.score = score; obs.threshold = threshold;
    // Same decoder as trackerKCF.cpp; work on a local copy, not native state.
    cv::Rect2d roi = oldroi;
    roi.x += peak.x - roi.width / 2 + 1;
    roi.y += peak.y - roi.height / 2 + 1;
    cv::Rect2d box;
    box.x = (resized ? roi.x*2 : roi.x) + (resized ? roi.width*2 : roi.width)/4;
    box.y = (resized ? roi.y*2 : roi.y) + (resized ? roi.height*2 : roi.height)/4;
    box.width = (resized ? roi.width*2 : roi.width)/2;
    box.height = (resized ? roi.height*2 : roi.height)/2;
    int x1 = cvRound(box.x), y1 = cvRound(box.y);
    int x2 = cvRound(box.x+box.width), y2 = cvRound(box.y+box.height);
    obs.candidate = cv::Rect(x1,y1,x2-x1,y2-y1) & cv::Rect(cv::Point(0,0),imageSize);
}

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "usage: kcf_check frames.txt x y w h output_dir observe[0/1]\n";
        return 2;
    }
    try {
        cv::setNumThreads(1); cv::setRNGSeed(42);
        std::ifstream list(argv[1]);
        std::vector<std::string> paths; std::string line;
        while (std::getline(list,line)) { if (!line.empty() && line.back()=='\r') line.pop_back(); if (!line.empty()) paths.push_back(line); }
        if (paths.empty()) throw std::runtime_error("no frames");
        cv::Rect box(cvRound(std::stod(argv[2])),cvRound(std::stod(argv[3])),
                     cvRound(std::stod(argv[4])),cvRound(std::stod(argv[5])));
        bool enabled = std::stoi(argv[7]) != 0;
        std::filesystem::create_directories(argv[6]);
        auto out = std::filesystem::path(argv[6]);
        std::ofstream rows(out/"rows.csv");
        std::ofstream evidence;
        if (enabled) evidence.open(out/"rejected_responses.f32",std::ios::binary);
        rows << std::setprecision(17);
        rows << "index,native_valid,has_response,native_x,native_y,native_w,native_h,candidate_x,candidate_y,candidate_w,candidate_h,peak,threshold,response_h,response_w,offset_bytes,roi_x,roi_y,roi_w,roi_h,resized,peak_x,peak_y,decoder_equal\n";
        cv::TrackerKCF::Params params; // Unchanged OpenCV defaults.
        auto tracker = cv::TrackerKCF::create(params);
        cv::Mat image = cv::imread(paths[0]);
        if (image.empty()) throw std::runtime_error("unreadable first frame");
        tracker->init(image,box);
        cv::setKCFObservationCallback(enabled ? observe : nullptr);
        for (size_t i=0; i<paths.size(); ++i) {
            obs = Observation();
            bool accepted = true;
            if (i) {
                image = cv::imread(paths[i]);
                if (image.empty()) throw std::runtime_error("unreadable frame: " + paths[i]);
                accepted = tracker->update(image,box);
            }
            long long offset = -1;
            if (enabled && !accepted && obs.available) {
                if (obs.response.type()!=CV_32FC1) throw std::runtime_error("unexpected response type");
                offset = static_cast<long long>(evidence.tellp());
                for(int r=0;r<obs.response.rows;++r)
                    evidence.write(reinterpret_cast<const char*>(obs.response.ptr<float>(r)),obs.response.cols*sizeof(float));
            }
            int equal = accepted && obs.available ? int(box==obs.candidate) : -1;
            rows << i << ',' << accepted << ',' << obs.available << ','
                 << box.x << ',' << box.y << ',' << box.width << ',' << box.height << ','
                 << obs.candidate.x << ',' << obs.candidate.y << ',' << obs.candidate.width << ',' << obs.candidate.height << ','
                 << obs.score << ',' << obs.threshold << ',' << obs.response.rows << ',' << obs.response.cols << ',' << offset << ','
                 << obs.roi.x << ',' << obs.roi.y << ',' << obs.roi.width << ',' << obs.roi.height << ',' << obs.resized << ','
                 << obs.peak.x << ',' << obs.peak.y << ',' << equal << '\n';
        }
        cv::setKCFObservationCallback(nullptr);
        std::cout << "completed " << paths.size() << " frames\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
