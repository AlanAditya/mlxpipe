#include "mlxpipe/model/model_bundle.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace mlxpipe {

namespace {

template <typename T>
T read_le(const std::vector<uint8_t>& data, size_t offset) {
    if (offset + sizeof(T) > data.size()) throw std::runtime_error("Truncated zip archive");
    T value;
    std::memcpy(&value, data.data() + offset, sizeof(T));
    return value;
}

}  // namespace

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open " + path);
    std::vector<uint8_t> data(static_cast<size_t>(file.tellg()));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(data.data()), data.size());
    if (!file) throw std::runtime_error("Cannot read " + path);
    return data;
}

ModelBundle ModelBundle::open(const std::string& path) {
    std::vector<uint8_t> data = read_file(path);
    constexpr uint32_t kEndOfCentralDirectory = 0x06054b50;
    constexpr uint32_t kCentralDirectoryEntry = 0x02014b50;
    constexpr uint32_t kLocalFileHeader = 0x04034b50;

    // The end-of-central-directory record is within the last 64 KiB + 22 bytes.
    if (data.size() < 22) throw std::runtime_error(path + " is not a zip archive");
    size_t eocd = std::string::npos;
    size_t search_start = data.size() > 0xFFFF + 22 ? data.size() - 0xFFFF - 22 : 0;
    for (size_t i = data.size() - 22 + 1; i-- > search_start;) {
        if (read_le<uint32_t>(data, i) == kEndOfCentralDirectory) {
            eocd = i;
            break;
        }
    }
    if (eocd == std::string::npos) throw std::runtime_error(path + " is not a zip archive");

    uint16_t num_entries = read_le<uint16_t>(data, eocd + 10);
    size_t entry = read_le<uint32_t>(data, eocd + 16);

    ModelBundle bundle;
    bundle.path_ = path;
    for (uint16_t i = 0; i < num_entries; ++i) {
        if (read_le<uint32_t>(data, entry) != kCentralDirectoryEntry) {
            throw std::runtime_error("Corrupt central directory in " + path);
        }
        uint16_t method = read_le<uint16_t>(data, entry + 10);
        uint32_t compressed_size = read_le<uint32_t>(data, entry + 20);
        uint16_t name_length = read_le<uint16_t>(data, entry + 28);
        uint16_t extra_length = read_le<uint16_t>(data, entry + 30);
        uint16_t comment_length = read_le<uint16_t>(data, entry + 32);
        uint32_t local_header = read_le<uint32_t>(data, entry + 42);
        std::string name(reinterpret_cast<const char*>(data.data() + entry + 46), name_length);
        entry += 46 + name_length + extra_length + comment_length;

        if (method != 0) throw std::runtime_error(name + " in " + path + " is compressed; expected a stored entry");
        if (read_le<uint32_t>(data, local_header) != kLocalFileHeader) {
            throw std::runtime_error("Corrupt local header for " + name + " in " + path);
        }
        size_t content = local_header + 30 + read_le<uint16_t>(data, local_header + 26) +
                         read_le<uint16_t>(data, local_header + 28);
        if (content + compressed_size > data.size()) throw std::runtime_error("Truncated entry " + name);
        bundle.files_[name] =
            std::vector<uint8_t>(data.begin() + content, data.begin() + content + compressed_size);
    }
    return bundle;
}

const std::vector<uint8_t>& ModelBundle::file(const std::string& name) const {
    auto it = files_.find(name);
    if (it == files_.end()) throw std::out_of_range(name + " not found in " + path_);
    return it->second;
}

std::vector<std::string> ModelBundle::file_names() const {
    std::vector<std::string> names;
    for (const auto& [name, _] : files_) names.push_back(name);
    return names;
}

}  // namespace mlxpipe
