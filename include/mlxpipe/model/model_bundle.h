// Reads MediaPipe model bundles (.task files): uncompressed zip archives that
// hold the .tflite models of a task, e.g. hand_landmarker.task contains
// hand_detector.tflite and hand_landmarks_detector.tflite.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace mlxpipe {

class ModelBundle {
public:
    // Throws std::runtime_error if the file cannot be read or is not a
    // stored (uncompressed) zip archive.
    static ModelBundle open(const std::string& path);

    bool contains(const std::string& name) const { return files_.count(name) > 0; }
    // Contents of `name`; throws std::out_of_range if absent.
    const std::vector<uint8_t>& file(const std::string& name) const;
    std::vector<std::string> file_names() const;
    const std::string& path() const { return path_; }

private:
    std::string path_;
    std::map<std::string, std::vector<uint8_t>> files_;
};

// Reads a whole file into memory; throws std::runtime_error on failure.
std::vector<uint8_t> read_file(const std::string& path);

}  // namespace mlxpipe
