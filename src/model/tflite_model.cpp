#include "mlxpipe/model/tflite_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>

#include "mlxpipe/model/model_bundle.h"

namespace mlxpipe {

namespace mx = mlx::core;

namespace {

// --- Minimal FlatBuffers reader ---------------------------------------------
// Field ids below follow tensorflow/lite/schema/schema.fbs.

class FlatBuffer {
public:
    explicit FlatBuffer(const std::vector<uint8_t>& data) : data_(data) {}
    template <typename T>
    T read(size_t offset) const {
        if (offset + sizeof(T) > data_.size()) throw std::runtime_error("Malformed TFLite flatbuffer");
        T value;
        std::memcpy(&value, data_.data() + offset, sizeof(T));
        return value;
    }
    const uint8_t* bytes(size_t offset) const { return data_.data() + offset; }
    size_t size() const { return data_.size(); }

private:
    const std::vector<uint8_t>& data_;
};

class Table {
public:
    Table() = default;
    Table(const FlatBuffer* fb, size_t pos) : fb_(fb), pos_(pos) {}
    bool valid() const { return fb_ != nullptr; }

    template <typename T>
    T scalar(int id, T fallback) const {
        size_t f = field(id);
        return f ? fb_->read<T>(f) : fallback;
    }
    Table table(int id) const {
        size_t f = field(id);
        return f ? Table(fb_, f + fb_->read<uint32_t>(f)) : Table();
    }
    std::string string(int id) const {
        size_t f = field(id);
        if (!f) return "";
        size_t s = f + fb_->read<uint32_t>(f);
        return std::string(reinterpret_cast<const char*>(fb_->bytes(s + 4)), fb_->read<uint32_t>(s));
    }
    // Returns (start of elements, length).
    std::pair<size_t, uint32_t> vector(int id) const {
        size_t f = field(id);
        if (!f) return {0, 0};
        size_t s = f + fb_->read<uint32_t>(f);
        return {s + 4, fb_->read<uint32_t>(s)};
    }
    template <typename T>
    std::vector<T> scalars(int id) const {
        auto [start, length] = vector(id);
        std::vector<T> out(length);
        for (uint32_t i = 0; i < length; ++i) out[i] = fb_->read<T>(start + i * sizeof(T));
        return out;
    }
    std::vector<Table> tables(int id) const {
        auto [start, length] = vector(id);
        std::vector<Table> out;
        for (uint32_t i = 0; i < length; ++i) {
            size_t q = start + 4 * i;
            out.emplace_back(fb_, q + fb_->read<uint32_t>(q));
        }
        return out;
    }

private:
    size_t field(int id) const {
        int64_t vtable = static_cast<int64_t>(pos_) - fb_->read<int32_t>(pos_);
        uint16_t vtable_size = fb_->read<uint16_t>(vtable);
        size_t slot = 4 + 2 * id;
        if (slot >= vtable_size) return 0;
        uint16_t offset = fb_->read<uint16_t>(vtable + slot);
        return offset ? pos_ + offset : 0;
    }

    const FlatBuffer* fb_ = nullptr;
    size_t pos_ = 0;
};

// BuiltinOperator values used by MediaPipe's vision models.
enum OpCode : int {
    kAdd = 0,
    kAveragePool2D = 1,
    kConcatenation = 2,
    kConv2D = 3,
    kDepthwiseConv2D = 4,
    kDequantize = 6,
    kFullyConnected = 9,
    kLogistic = 14,
    kMaxPool2D = 17,
    kMul = 18,
    kRelu = 19,
    kRelu6 = 21,
    kReshape = 22,
    kResizeBilinear = 23,
    kTanh = 28,
    kPad = 34,
    kTranspose = 39,
    kMean = 40,
    kSub = 41,
    kDiv = 42,
    kStridedSlice = 45,
    kPRelu = 54,
    kNeg = 59,
    kSum = 74,
    kSqrt = 75,
    kRsqrt = 76,
    kSquaredDifference = 99,
};

const char* op_name(int code) {
    switch (code) {
        case kAdd: return "ADD";
        case kAveragePool2D: return "AVERAGE_POOL_2D";
        case kConcatenation: return "CONCATENATION";
        case kConv2D: return "CONV_2D";
        case kDepthwiseConv2D: return "DEPTHWISE_CONV_2D";
        case kDequantize: return "DEQUANTIZE";
        case kFullyConnected: return "FULLY_CONNECTED";
        case kLogistic: return "LOGISTIC";
        case kMaxPool2D: return "MAX_POOL_2D";
        case kMul: return "MUL";
        case kRelu: return "RELU";
        case kRelu6: return "RELU6";
        case kReshape: return "RESHAPE";
        case kResizeBilinear: return "RESIZE_BILINEAR";
        case kTanh: return "TANH";
        case kPad: return "PAD";
        case kTranspose: return "TRANSPOSE";
        case kMean: return "MEAN";
        case kSub: return "SUB";
        case kDiv: return "DIV";
        case kStridedSlice: return "STRIDED_SLICE";
        case kPRelu: return "PRELU";
        case kNeg: return "NEG";
        case kSum: return "SUM";
        case kSqrt: return "SQRT";
        case kRsqrt: return "RSQRT";
        case kSquaredDifference: return "SQUARED_DIFFERENCE";
        default: return nullptr;
    }
}

enum Activation : int { kActNone = 0, kActRelu = 1, kActReluN1To1 = 2, kActRelu6 = 3, kActTanh = 4 };
enum Padding : int { kPaddingSame = 0, kPaddingValid = 1 };

// Parsed operator with the builtin options this runtime understands.
struct Op {
    int code = 0;
    std::vector<int> inputs;
    std::vector<int> outputs;
    int padding = kPaddingSame;
    int stride_w = 1, stride_h = 1;
    int dilation_w = 1, dilation_h = 1;
    int filter_w = 1, filter_h = 1;
    int activation = kActNone;
    int axis = 0;
    bool keep_dims = false;
    bool keep_num_dims = false;
    bool align_corners = false;
    bool half_pixel_centers = false;
    std::vector<int> new_shape;
    int begin_mask = 0, end_mask = 0, ellipsis_mask = 0, new_axis_mask = 0, shrink_axis_mask = 0;
    // Weights rearranged once at load time, indexed by input slot.
    std::vector<std::optional<mx::array>> prepared;
    bool folded = false;
};

mx::Dtype tflite_dtype(int type) {
    switch (type) {
        case 0: return mx::float32;
        case 1: return mx::float16;
        case 2: return mx::int32;
        case 3: return mx::uint8;
        case 4: return mx::int64;
        case 6: return mx::bool_;
        case 7: return mx::int16;
        case 9: return mx::int8;
        default: throw std::runtime_error("Unsupported TFLite tensor type " + std::to_string(type));
    }
}

template <typename T>
mx::array array_from_bytes(const uint8_t* bytes, size_t count, const mx::Shape& shape, mx::Dtype dtype) {
    std::vector<T> values(count);
    std::memcpy(values.data(), bytes, count * sizeof(T));
    return mx::array(values.begin(), shape, dtype);
}

mx::array constant_from_bytes(const uint8_t* bytes, size_t length, const mx::Shape& shape, mx::Dtype dtype) {
    size_t count = 1;
    for (int d : shape) count *= d;
    if (count * mx::size_of(dtype) != length) throw std::runtime_error("TFLite buffer size does not match tensor shape");
    if (dtype == mx::float32) return array_from_bytes<float>(bytes, count, shape, dtype);
    if (dtype == mx::float16) return array_from_bytes<mx::float16_t>(bytes, count, shape, dtype);
    if (dtype == mx::int32) return array_from_bytes<int32_t>(bytes, count, shape, dtype);
    if (dtype == mx::int64) return array_from_bytes<int64_t>(bytes, count, shape, dtype);
    if (dtype == mx::uint8) return array_from_bytes<uint8_t>(bytes, count, shape, dtype);
    if (dtype == mx::int8) return array_from_bytes<int8_t>(bytes, count, shape, dtype);
    if (dtype == mx::int16) return array_from_bytes<int16_t>(bytes, count, shape, dtype);
    if (dtype == mx::bool_) return mx::astype(array_from_bytes<uint8_t>(bytes, count, shape, mx::uint8), mx::bool_);
    throw std::runtime_error("Unsupported constant dtype");
}

// Reads an int32/int64 constant as host integers.
std::vector<int> host_ints(const mx::array& a) {
    mx::array c = mx::astype(a, mx::int32);
    mx::eval(c);
    const int32_t* p = c.data<int32_t>();
    return std::vector<int>(p, p + c.size());
}

mx::array apply_activation(mx::array x, int activation) {
    switch (activation) {
        case kActNone: return x;
        case kActRelu: return mx::maximum(x, mx::array(0.0f));
        case kActReluN1To1: return mx::clip(x, mx::array(-1.0f), mx::array(1.0f));
        case kActRelu6: return mx::clip(x, mx::array(0.0f), mx::array(6.0f));
        case kActTanh: return mx::tanh(x);
        default: throw std::runtime_error("Unsupported fused activation " + std::to_string(activation));
    }
}

// TFLite SAME padding for one spatial dimension: (before, after).
std::pair<int, int> same_padding(int in, int filter, int stride, int dilation) {
    int out = (in + stride - 1) / stride;
    int effective = (filter - 1) * dilation + 1;
    int total = std::max((out - 1) * stride + effective - in, 0);
    return {total / 2, total - total / 2};
}

int normalize_axis(int axis, int rank) { return axis < 0 ? axis + rank : axis; }

// Bilinear resize along one axis using TFLite's ResizeBilinear sampling.
mx::array resize_axis(const mx::array& x, int axis, int out_size, bool align_corners, bool half_pixel) {
    int in_size = x.shape(axis);
    if (in_size == out_size) return x;
    float scale = (align_corners && out_size > 1) ? static_cast<float>(in_size - 1) / (out_size - 1)
                                                  : static_cast<float>(in_size) / out_size;
    std::vector<int32_t> lower(out_size), upper(out_size);
    std::vector<float> w_lower(out_size), w_upper(out_size);
    for (int o = 0; o < out_size; ++o) {
        float s = half_pixel ? (o + 0.5f) * scale - 0.5f : o * scale;
        int lo = std::max(static_cast<int>(std::floor(s)), 0);
        int hi = std::min(static_cast<int>(std::ceil(s)), in_size - 1);
        float frac = s - lo;
        lower[o] = lo;
        upper[o] = hi;
        w_lower[o] = 1.0f - frac;
        w_upper[o] = frac;
    }
    mx::Shape weight_shape(x.ndim(), 1);
    weight_shape[axis] = out_size;
    auto lo = mx::take(x, mx::array(lower.begin(), {out_size}, mx::int32), axis);
    auto hi = mx::take(x, mx::array(upper.begin(), {out_size}, mx::int32), axis);
    auto wl = mx::array(w_lower.begin(), weight_shape, mx::float32);
    auto wu = mx::array(w_upper.begin(), weight_shape, mx::float32);
    return mx::add(mx::multiply(lo, wl), mx::multiply(hi, wu));
}

// 2D pooling over NHWC.
mx::array pool2d(const mx::array& x, const Op& op, bool is_max) {
    int h = x.shape(1), w = x.shape(2);
    std::pair<int, int> pad_h{0, 0}, pad_w{0, 0};
    if (op.padding == kPaddingSame) {
        pad_h = same_padding(h, op.filter_h, op.stride_h, 1);
        pad_w = same_padding(w, op.filter_w, op.stride_w, 1);
    }
    bool padded = pad_h.first || pad_h.second || pad_w.first || pad_w.second;
    if (!is_max && padded) throw std::runtime_error("AVERAGE_POOL_2D with SAME padding is not implemented");

    mx::array input = x;
    if (padded) {
        input = mx::pad(x, {1, 2}, {pad_h.first, pad_w.first}, {pad_h.second, pad_w.second},
                        mx::array(-std::numeric_limits<float>::infinity()));
    }
    int ph = input.shape(1), pw = input.shape(2);
    int out_h = (ph - op.filter_h) / op.stride_h + 1;
    int out_w = (pw - op.filter_w) / op.stride_w + 1;
    int n = input.shape(0), c = input.shape(3);

    mx::array windows = input;
    std::vector<int> reduce_axes;
    if (op.filter_h == op.stride_h && op.filter_w == op.stride_w) {
        // Non-overlapping windows: a reshape exposes them directly.
        windows = mx::slice(input, {0, 0, 0, 0}, {n, out_h * op.stride_h, out_w * op.stride_w, c});
        windows = mx::reshape(windows, {n, out_h, op.filter_h, out_w, op.filter_w, c});
        reduce_axes = {2, 4};
    } else {
        input = mx::copy(input);
        windows = mx::as_strided(input, {n, out_h, out_w, op.filter_h, op.filter_w, c},
                                 {static_cast<int64_t>(ph) * pw * c, static_cast<int64_t>(op.stride_h) * pw * c,
                                  static_cast<int64_t>(op.stride_w) * c, static_cast<int64_t>(pw) * c, c, 1},
                                 0);
        reduce_axes = {3, 4};
    }
    mx::array out = is_max ? mx::max(windows, reduce_axes) : mx::mean(windows, reduce_axes);
    return apply_activation(out, op.activation);
}

}  // namespace

struct TfLiteModel::Impl {
    std::string name;
    std::vector<mx::Shape> tensor_shapes;
    std::vector<std::string> tensor_names;
    std::vector<int> inputs;
    std::vector<int> outputs;
    std::vector<Op> ops;
    // Constant tensors (weights and folded subgraphs), indexed by tensor id.
    std::vector<std::optional<mx::array>> constants;
    int declared_batch = 1;
    std::optional<TfLiteModel::Normalization> normalization;

    void parse(const std::vector<uint8_t>& data);
    void fold_constants();
    void prepare_weights();
    std::vector<mx::array> forward(const std::vector<mx::array>& in) const;
    void run_op(const Op& op, std::vector<std::optional<mx::array>>& values, int batch) const;
};

void TfLiteModel::Impl::parse(const std::vector<uint8_t>& data) {
    FlatBuffer fb(data);
    Table model(&fb, fb.read<uint32_t>(0));

    std::vector<Table> subgraphs = model.tables(2);
    if (subgraphs.empty()) throw std::runtime_error(name + ": model has no subgraphs");
    if (subgraphs.size() > 1) throw std::runtime_error(name + ": models with multiple subgraphs are not supported");
    Table subgraph = subgraphs[0];
    std::vector<Table> buffers = model.tables(4);

    std::vector<int> op_codes;
    std::vector<std::string> custom_codes;
    for (const Table& code : model.tables(1)) {
        int deprecated = code.scalar<int8_t>(0, 0);
        int builtin = code.scalar<int32_t>(3, 0);
        op_codes.push_back(std::max(deprecated, builtin));
        custom_codes.push_back(code.string(1));
    }

    std::vector<Table> tensors = subgraph.tables(0);
    tensor_shapes.resize(tensors.size());
    tensor_names.resize(tensors.size());
    constants.resize(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        const Table& t = tensors[i];
        std::vector<int32_t> shape = t.scalars<int32_t>(0);
        tensor_shapes[i] = mx::Shape(shape.begin(), shape.end());
        tensor_names[i] = t.string(3);
        uint32_t buffer_index = t.scalar<uint32_t>(2, 0);
        if (buffer_index == 0 || buffer_index >= buffers.size()) continue;
        const Table& buffer = buffers[buffer_index];
        auto [start, length] = buffer.vector(0);
        const uint8_t* bytes = length ? fb.bytes(start) : nullptr;
        uint64_t offset = buffer.scalar<uint64_t>(1, 0);
        uint64_t size = buffer.scalar<uint64_t>(2, 0);
        if (!length && offset > 1 && size) {  // large models store data after the flatbuffer
            if (offset + size > data.size()) throw std::runtime_error(name + ": buffer out of range");
            bytes = data.data() + offset;
            length = static_cast<uint32_t>(size);
        }
        if (!bytes) continue;
        constants[i] = constant_from_bytes(bytes, length, tensor_shapes[i], tflite_dtype(t.scalar<uint8_t>(1, 0)));
    }

    inputs = subgraph.scalars<int32_t>(1);
    outputs = subgraph.scalars<int32_t>(2);
    if (!inputs.empty() && !tensor_shapes[inputs[0]].empty()) declared_batch = tensor_shapes[inputs[0]][0];

    for (const Table& o : subgraph.tables(3)) {
        Op op;
        uint32_t code_index = o.scalar<uint32_t>(0, 0);
        op.code = op_codes.at(code_index);
        if (!op_name(op.code)) {
            std::string what = op.code == 32 ? "custom op " + custom_codes[code_index]
                                             : "builtin op " + std::to_string(op.code);
            throw std::runtime_error(name + ": unsupported " + what);
        }
        op.inputs = o.scalars<int32_t>(1);
        op.outputs = o.scalars<int32_t>(2);
        Table options = o.table(4);
        if (options.valid()) {
            switch (op.code) {
                case kConv2D:
                    op.padding = options.scalar<int8_t>(0, 0);
                    op.stride_w = options.scalar<int32_t>(1, 1);
                    op.stride_h = options.scalar<int32_t>(2, 1);
                    op.activation = options.scalar<int8_t>(3, 0);
                    op.dilation_w = options.scalar<int32_t>(4, 1);
                    op.dilation_h = options.scalar<int32_t>(5, 1);
                    break;
                case kDepthwiseConv2D:
                    op.padding = options.scalar<int8_t>(0, 0);
                    op.stride_w = options.scalar<int32_t>(1, 1);
                    op.stride_h = options.scalar<int32_t>(2, 1);
                    op.activation = options.scalar<int8_t>(4, 0);
                    op.dilation_w = options.scalar<int32_t>(5, 1);
                    op.dilation_h = options.scalar<int32_t>(6, 1);
                    break;
                case kMaxPool2D:
                case kAveragePool2D:
                    op.padding = options.scalar<int8_t>(0, 0);
                    op.stride_w = options.scalar<int32_t>(1, 1);
                    op.stride_h = options.scalar<int32_t>(2, 1);
                    op.filter_w = options.scalar<int32_t>(3, 1);
                    op.filter_h = options.scalar<int32_t>(4, 1);
                    op.activation = options.scalar<int8_t>(5, 0);
                    break;
                case kConcatenation:
                    op.axis = options.scalar<int32_t>(0, 0);
                    op.activation = options.scalar<int8_t>(1, 0);
                    break;
                case kReshape:
                    for (int32_t d : options.scalars<int32_t>(0)) op.new_shape.push_back(d);
                    break;
                case kResizeBilinear:
                    op.align_corners = options.scalar<uint8_t>(2, 0);
                    op.half_pixel_centers = options.scalar<uint8_t>(3, 0);
                    break;
                case kFullyConnected:
                    op.activation = options.scalar<int8_t>(0, 0);
                    if (options.scalar<int8_t>(1, 0) != 0) throw std::runtime_error(name + ": shuffled FC weights");
                    op.keep_num_dims = options.scalar<uint8_t>(2, 0);
                    break;
                case kAdd:
                case kMul:
                case kSub:
                case kDiv:
                    op.activation = options.scalar<int8_t>(0, 0);
                    break;
                case kMean:
                case kSum:
                    op.keep_dims = options.scalar<uint8_t>(0, 0);
                    break;
                case kStridedSlice:
                    op.begin_mask = options.scalar<int32_t>(0, 0);
                    op.end_mask = options.scalar<int32_t>(1, 0);
                    op.ellipsis_mask = options.scalar<int32_t>(2, 0);
                    op.new_axis_mask = options.scalar<int32_t>(3, 0);
                    op.shrink_axis_mask = options.scalar<int32_t>(4, 0);
                    if (op.ellipsis_mask || op.new_axis_mask) {
                        throw std::runtime_error(name + ": STRIDED_SLICE ellipsis/new-axis masks are not supported");
                    }
                    break;
                default:
                    break;
            }
        }
        op.prepared.resize(op.inputs.size());
        ops.push_back(std::move(op));
    }

    // TFLite metadata: ModelMetadata.subgraph_metadata[0].input_tensor_metadata[0]
    // .process_units[] with NormalizationOptions (field ids from
    // tensorflow_lite_support metadata_schema.fbs).
    for (const Table& metadata : model.tables(6)) {
        if (metadata.string(0) != "TFLITE_METADATA") continue;
        uint32_t buffer_index = metadata.scalar<uint32_t>(1, 0);
        if (buffer_index >= buffers.size()) continue;
        auto [start, length] = buffers[buffer_index].vector(0);
        if (!length) continue;
        std::vector<uint8_t> bytes(fb.bytes(start), fb.bytes(start) + length);
        FlatBuffer meta_fb(bytes);
        Table model_metadata(&meta_fb, meta_fb.read<uint32_t>(0));
        std::vector<Table> subgraph_metadata = model_metadata.tables(3);
        if (subgraph_metadata.empty()) continue;
        std::vector<Table> input_metadata = subgraph_metadata[0].tables(2);
        if (input_metadata.empty()) continue;
        for (const Table& unit : input_metadata[0].tables(4)) {
            constexpr uint8_t kNormalizationOptions = 1;
            if (unit.scalar<uint8_t>(0, 0) != kNormalizationOptions) continue;
            Table options = unit.table(1);
            std::vector<float> mean = options.scalars<float>(0);
            std::vector<float> stddev = options.scalars<float>(1);
            if (!mean.empty() && !stddev.empty()) normalization = TfLiteModel::Normalization{mean[0], stddev[0]};
        }
    }
}

// Evaluates operators whose inputs are all constant (e.g. DEQUANTIZE of
// float16 weights) once, at load time.
void TfLiteModel::Impl::fold_constants() {
    for (Op& op : ops) {
        bool all_constant = !op.inputs.empty();
        for (int idx : op.inputs) {
            if (idx >= 0 && !constants[idx]) all_constant = false;
        }
        if (!all_constant) continue;
        run_op(op, constants, declared_batch);
        for (int idx : op.outputs) mx::eval(*constants[idx]);
        op.folded = true;
    }
}

// Lays out weights in MLX's formats once.
void TfLiteModel::Impl::prepare_weights() {
    for (Op& op : ops) {
        if (op.folded) continue;
        auto weight = [&](int slot) -> std::optional<mx::array> {
            if (slot >= (int)op.inputs.size() || op.inputs[slot] < 0) return std::nullopt;
            return constants[op.inputs[slot]];
        };
        if (op.code == kDepthwiseConv2D) {
            // [1, KH, KW, C * M] -> [C * M, KH, KW, 1]
            if (auto w = weight(1)) {
                op.prepared[1] = mx::copy(mx::transpose(mx::astype(*w, mx::float32), {3, 1, 2, 0}));
                mx::eval(*op.prepared[1]);
            }
        } else if (op.code == kFullyConnected) {
            // [O, I] -> [I, O]
            if (auto w = weight(1)) {
                op.prepared[1] = mx::copy(mx::transpose(mx::astype(*w, mx::float32)));
                mx::eval(*op.prepared[1]);
            }
        }
    }
}

void TfLiteModel::Impl::run_op(const Op& op, std::vector<std::optional<mx::array>>& values, int batch) const {
    auto in = [&](size_t slot) -> const mx::array& {
        int idx = op.inputs.at(slot);
        if (idx < 0 || !values[idx]) {
            throw std::runtime_error(name + ": missing input " + std::to_string(slot) + " for " + op_name(op.code));
        }
        return *values[idx];
    };
    auto has_in = [&](size_t slot) { return slot < op.inputs.size() && op.inputs[slot] >= 0; };
    auto weight = [&](size_t slot) -> mx::array { return op.prepared[slot] ? *op.prepared[slot] : in(slot); };
    auto set = [&](size_t slot, mx::array value) { values[op.outputs.at(slot)] = std::move(value); };
    auto binary = [&](mx::array result) { set(0, apply_activation(std::move(result), op.activation)); };

    switch (op.code) {
        case kConv2D:
        case kDepthwiseConv2D: {
            const mx::array& x = in(0);
            bool depthwise = op.code == kDepthwiseConv2D;
            mx::array w = weight(1);
            if (depthwise && !op.prepared[1]) w = mx::transpose(w, {3, 1, 2, 0});
            int kh = w.shape(1), kw = w.shape(2);
            std::pair<int, int> ph{0, 0}, pw{0, 0};
            if (op.padding == kPaddingSame) {
                ph = same_padding(x.shape(1), kh, op.stride_h, op.dilation_h);
                pw = same_padding(x.shape(2), kw, op.stride_w, op.dilation_w);
            }
            int groups = depthwise ? x.shape(3) : x.shape(3) / w.shape(3);
            mx::array y = mx::conv_general(x, w, {op.stride_h, op.stride_w}, {ph.first, pw.first},
                                           {ph.second, pw.second}, {op.dilation_h, op.dilation_w}, {1, 1}, groups);
            if (has_in(2)) y = mx::add(y, in(2));
            set(0, apply_activation(y, op.activation));
            break;
        }
        case kFullyConnected: {
            const mx::array& x = in(0);
            mx::array w = op.prepared[1] ? *op.prepared[1] : mx::transpose(in(1));
            int in_features = w.shape(0);
            mx::array y = mx::matmul(mx::reshape(x, {-1, in_features}), w);
            if (has_in(2)) y = mx::add(y, in(2));
            if (op.keep_num_dims) {
                mx::Shape shape = x.shape();
                shape.back() = w.shape(1);
                y = mx::reshape(y, shape);
            }
            set(0, apply_activation(y, op.activation));
            break;
        }
        case kAdd: binary(mx::add(in(0), in(1))); break;
        case kSub: binary(mx::subtract(in(0), in(1))); break;
        case kMul: binary(mx::multiply(in(0), in(1))); break;
        case kDiv: binary(mx::divide(in(0), in(1))); break;
        case kSquaredDifference: set(0, mx::square(mx::subtract(in(0), in(1)))); break;
        case kPRelu: {
            const mx::array& x = in(0);
            set(0, mx::add(mx::maximum(x, mx::array(0.0f)), mx::multiply(in(1), mx::minimum(x, mx::array(0.0f)))));
            break;
        }
        case kRelu: set(0, mx::maximum(in(0), mx::array(0.0f))); break;
        case kRelu6: set(0, mx::clip(in(0), mx::array(0.0f), mx::array(6.0f))); break;
        case kLogistic: set(0, mx::sigmoid(in(0))); break;
        case kTanh: set(0, mx::tanh(in(0))); break;
        case kNeg: set(0, mx::negative(in(0))); break;
        case kSqrt: set(0, mx::sqrt(in(0))); break;
        case kRsqrt: set(0, mx::rsqrt(in(0))); break;
        case kDequantize: set(0, mx::astype(in(0), mx::float32)); break;
        case kMaxPool2D: set(0, pool2d(in(0), op, /*is_max=*/true)); break;
        case kAveragePool2D: set(0, pool2d(in(0), op, /*is_max=*/false)); break;
        case kPad: {
            const mx::array& x = in(0);
            std::vector<int> pads = host_ints(in(1));
            std::vector<int> axes;
            mx::Shape low, high;
            for (int d = 0; d < static_cast<int>(x.ndim()); ++d) {
                if (pads[2 * d] == 0 && pads[2 * d + 1] == 0) continue;
                axes.push_back(d);
                low.push_back(pads[2 * d]);
                high.push_back(pads[2 * d + 1]);
            }
            set(0, axes.empty() ? x : mx::pad(x, axes, low, high, mx::array(0.0f, x.dtype())));
            break;
        }
        case kReshape: {
            const mx::array& x = in(0);
            std::vector<int> shape = has_in(1) ? host_ints(in(1)) : op.new_shape;
            if (!shape.empty() && shape[0] == declared_batch && x.ndim() > 0 && x.shape(0) == batch &&
                batch != declared_batch) {
                shape[0] = batch;
            }
            set(0, mx::reshape(x, mx::Shape(shape.begin(), shape.end())));
            break;
        }
        case kConcatenation: {
            std::vector<mx::array> parts;
            for (size_t i = 0; i < op.inputs.size(); ++i) parts.push_back(in(i));
            int axis = normalize_axis(op.axis, parts[0].ndim());
            set(0, apply_activation(mx::concatenate(parts, axis), op.activation));
            break;
        }
        case kResizeBilinear: {
            std::vector<int> size = host_ints(in(1));
            mx::array y = resize_axis(in(0), 1, size[0], op.align_corners, op.half_pixel_centers);
            set(0, resize_axis(y, 2, size[1], op.align_corners, op.half_pixel_centers));
            break;
        }
        case kMean:
        case kSum: {
            const mx::array& x = in(0);
            std::vector<int> axes = host_ints(in(1));
            for (int& a : axes) a = normalize_axis(a, x.ndim());
            set(0, op.code == kMean ? mx::mean(x, axes, op.keep_dims) : mx::sum(x, axes, op.keep_dims));
            break;
        }
        case kTranspose: {
            std::vector<int> perm = host_ints(in(1));
            set(0, mx::transpose(in(0), perm));
            break;
        }
        case kStridedSlice: {
            const mx::array& x = in(0);
            std::vector<int> begin = host_ints(in(1)), end = host_ints(in(2)), strides = host_ints(in(3));
            int rank = x.ndim();
            mx::Shape start(rank, 0), stop(rank), step(rank, 1);
            std::vector<int> squeeze_axes;
            for (int d = 0; d < rank; ++d) {
                int dim = x.shape(d);
                if (d >= static_cast<int>(begin.size())) {
                    stop[d] = dim;
                    continue;
                }
                step[d] = strides[d];
                if (step[d] <= 0) throw std::runtime_error(name + ": negative STRIDED_SLICE strides unsupported");
                int b = (op.begin_mask >> d) & 1 ? 0 : begin[d];
                int e = (op.end_mask >> d) & 1 ? dim : end[d];
                if (b < 0) b += dim;
                if (e < 0) e += dim;
                if ((op.shrink_axis_mask >> d) & 1) {
                    e = b + 1;
                    squeeze_axes.push_back(d);
                }
                start[d] = std::clamp(b, 0, dim);
                stop[d] = std::clamp(e, 0, dim);
            }
            mx::array y = mx::slice(x, start, stop, step);
            if (!squeeze_axes.empty()) y = mx::squeeze(y, squeeze_axes);
            set(0, y);
            break;
        }
        default:
            throw std::runtime_error(name + ": operator not implemented");
    }
}

std::vector<mx::array> TfLiteModel::Impl::forward(const std::vector<mx::array>& in) const {
    if (in.size() != inputs.size()) {
        throw std::invalid_argument(name + ": expected " + std::to_string(inputs.size()) + " inputs");
    }
    std::vector<std::optional<mx::array>> values = constants;
    for (size_t i = 0; i < inputs.size(); ++i) values[inputs[i]] = mx::astype(in[i], mx::float32);
    int batch = in[0].ndim() ? in[0].shape(0) : 1;
    for (const Op& op : ops) {
        if (!op.folded) run_op(op, values, batch);
    }
    std::vector<mx::array> out;
    for (int idx : outputs) out.push_back(*values[idx]);
    return out;
}

TfLiteModel::TfLiteModel(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {
    const Impl* impl_ptr = impl_.get();
    compiled_ = mx::compile([impl_ptr](const std::vector<mx::array>& in) { return impl_ptr->forward(in); });
}

TfLiteModel::~TfLiteModel() = default;

std::shared_ptr<TfLiteModel> TfLiteModel::load(const std::string& path) {
    std::string name = path.substr(path.find_last_of('/') + 1);
    return from_buffer(read_file(path), name);
}

std::shared_ptr<TfLiteModel> TfLiteModel::from_buffer(std::vector<uint8_t> data, std::string name) {
    auto impl = std::make_unique<Impl>();
    impl->name = std::move(name);
    impl->parse(data);
    impl->fold_constants();
    impl->prepare_weights();
    return std::shared_ptr<TfLiteModel>(new TfLiteModel(std::move(impl)));
}

std::vector<mx::array> TfLiteModel::operator()(const std::vector<mx::array>& inputs) const {
    return compiled_(inputs);
}

std::vector<mx::array> TfLiteModel::run_uncompiled(const std::vector<mx::array>& inputs) const {
    return impl_->forward(inputs);
}

const std::string& TfLiteModel::name() const { return impl_->name; }
size_t TfLiteModel::num_inputs() const { return impl_->inputs.size(); }
size_t TfLiteModel::num_outputs() const { return impl_->outputs.size(); }
std::vector<int> TfLiteModel::input_shape(size_t index) const {
    const auto& s = impl_->tensor_shapes.at(impl_->inputs.at(index));
    return std::vector<int>(s.begin(), s.end());
}
std::vector<int> TfLiteModel::output_shape(size_t index) const {
    const auto& s = impl_->tensor_shapes.at(impl_->outputs.at(index));
    return std::vector<int>(s.begin(), s.end());
}
const std::string& TfLiteModel::output_name(size_t index) const {
    return impl_->tensor_names.at(impl_->outputs.at(index));
}

const std::optional<TfLiteModel::Normalization>& TfLiteModel::input_normalization() const {
    return impl_->normalization;
}

std::pair<float, float> TfLiteModel::input_range(std::pair<float, float> fallback) const {
    if (!impl_->normalization) return fallback;
    const auto& n = *impl_->normalization;
    return {(0.0f - n.mean) / n.std, (255.0f - n.mean) / n.std};
}

}  // namespace mlxpipe
