import sys
import tflite
import re

def clean_name(name):
    # Make a valid C++ variable name
    return "t_" + re.sub(r'[^a-zA-Z0-9_]', '_', name)

def get_op_options(op, builtin_code):
    if builtin_code == tflite.BuiltinOperator.CONV_2D:
        opt = tflite.Conv2DOptions()
        opt.Init(op.CustomOptions().Bytes, op.CustomOptions().Pos) if op.CustomOptions() else None
        # Actually TFLite Flatbuffers have BuiltinOptions
        if op.BuiltinOptionsLength() > 0:
            opt.Init(op.BuiltinOptions().Bytes, op.CustomOptions().Pos) # This is tricky in Python flatbuffers
    return None

def apply_fused_activation(cpp_code, var_name, act_type):
    if act_type == tflite.ActivationFunctionType.RELU:
        cpp_code.append(f"        {var_name} = mlx::core::maximum({var_name}, mlx::core::array(0.0f));")
    elif act_type == tflite.ActivationFunctionType.RELU6:
        cpp_code.append(f"        {var_name} = mlx::core::minimum(mlx::core::maximum({var_name}, mlx::core::array(0.0f)), mlx::core::array(6.0f));")
    elif act_type == tflite.ActivationFunctionType.RELU_N1_TO_1:
        cpp_code.append(f"        {var_name} = mlx::core::minimum(mlx::core::maximum({var_name}, mlx::core::array(-1.0f)), mlx::core::array(1.0f));")

def builtin_code_to_name(code):
    for k, v in tflite.BuiltinOperator.__dict__.items():
        if v == code:
            return k
    return "UNKNOWN"

def main():
    if len(sys.argv) < 2:
        print("Usage: python transpile_tflite_to_mlx.py <model.tflite> [ClassName] [output.h]")
        sys.exit(1)
        
    model_path = sys.argv[1]
    class_name = sys.argv[2] if len(sys.argv) > 2 else "BlazePalm"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "src/blaze_palm.h"

    with open(model_path, 'rb') as f:
        buf = f.read()
        
    model = tflite.Model.GetRootAsModel(bytearray(buf), 0)
    subgraph = model.Subgraphs(0)
    
    # Map tensor index to name
    tensor_names = [subgraph.Tensors(i).Name().decode('utf-8') for i in range(subgraph.TensorsLength())]
    
    # Identify weights (tensors with data)
    weights = set()
    for i in range(subgraph.TensorsLength()):
        t = subgraph.Tensors(i)
        if model.Buffers(t.Buffer()).DataLength() > 0:
            weights.add(tensor_names[i])

    cpp_code = []
    cpp_code.append("#pragma once")
    cpp_code.append("#include <mlx/mlx.h>")
    cpp_code.append("#include <string>")
    cpp_code.append("#include <unordered_map>")
    cpp_code.append("#include <vector>")
    cpp_code.append("#include <algorithm>")
    cpp_code.append("")
    cpp_code.append("namespace mlx_vision {")
    cpp_code.append("")
    cpp_code.append(f"class {class_name} {{")
    cpp_code.append("public:")
    cpp_code.append("    std::unordered_map<std::string, mlx::core::array> weights;")
    cpp_code.append("")
    cpp_code.append(f"    {class_name}(const std::unordered_map<std::string, mlx::core::array>& w) : weights(w) {{}}")
    cpp_code.append("")
    cpp_code.append("    std::vector<int> extract_shape(mlx::core::array arr) {")
    cpp_code.append("        mlx::core::eval(arr);")
    cpp_code.append("        const int32_t* ptr = arr.data<int32_t>();")
    cpp_code.append("        return std::vector<int>(ptr, ptr + arr.size());")
    cpp_code.append("    }")
    cpp_code.append("")
    cpp_code.append("    std::vector<std::pair<int, int>> extract_padding(mlx::core::array arr) {")
    cpp_code.append("        mlx::core::eval(arr);")
    cpp_code.append("        const int32_t* ptr = arr.data<int32_t>();")
    cpp_code.append("        std::vector<std::pair<int, int>> pad_width;")
    cpp_code.append("        for (int i = 0; i < arr.size(); i += 2) {")
    cpp_code.append("            pad_width.push_back({ptr[i], ptr[i+1]});")
    cpp_code.append("        }")
    cpp_code.append("        return pad_width;")
    cpp_code.append("    }")
    cpp_code.append("")
    cpp_code.append("    mlx::core::array apply_same_padding(const mlx::core::array& x, int stride_h, int stride_w, int filter_h, int filter_w) {")
    cpp_code.append("        int in_h = x.shape(1);")
    cpp_code.append("        int in_w = x.shape(2);")
    cpp_code.append("        int out_h = (in_h + stride_h - 1) / stride_h;")
    cpp_code.append("        int out_w = (in_w + stride_w - 1) / stride_w;")
    cpp_code.append("        int pad_h = std::max((out_h - 1) * stride_h + filter_h - in_h, 0);")
    cpp_code.append("        int pad_w = std::max((out_w - 1) * stride_w + filter_w - in_w, 0);")
    cpp_code.append("        if (pad_h == 0 && pad_w == 0) return x;")
    cpp_code.append("        int pad_top = pad_h / 2;")
    cpp_code.append("        int pad_bottom = pad_h - pad_top;")
    cpp_code.append("        int pad_left = pad_w / 2;")
    cpp_code.append("        int pad_right = pad_w - pad_left;")
    cpp_code.append("        return mlx::core::pad(x, {1, 2}, {pad_top, pad_left}, {pad_bottom, pad_right});")
    cpp_code.append("    }")
    cpp_code.append("")
    cpp_code.append("    std::vector<mlx::core::array> forward(mlx::core::array input) {")
    
    # Map from dequantize output back to the original weight name
    dequantize_map = {}
    
    # Execution
    input_idx = subgraph.InputsAsNumpy()[0]
    cpp_code.append(f"        auto {clean_name(tensor_names[input_idx])} = input;")
    
    for i in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(i)
        op_code = model.OperatorCodes(op.OpcodeIndex())
        code_name = builtin_code_to_name(op_code.BuiltinCode())
        
        inputs = op.InputsAsNumpy()
        outputs = op.OutputsAsNumpy()
        
        in_names = [tensor_names[j] for j in inputs if j != -1]
        out_names = [tensor_names[j] for j in outputs if j != -1]
        
        # Replace DEQUANTIZE outputs with the actual weight names in our map
        in_vars = []
        for name in in_names:
            if name in dequantize_map:
                name = dequantize_map[name]
                
            if name in weights:
                in_vars.append(f'weights.at("{name}")')
            else:
                in_vars.append(clean_name(name))
                
        out_var = clean_name(out_names[0])
        
        if code_name == "DEQUANTIZE":
            dequantize_map[out_names[0]] = in_names[0]
            
        elif code_name == "CONV_2D":
            # MLX conv2d expects weights [Out, H, W, In]. 
            # TFLite weights are [Out, H, W, In] natively in the flatbuffer.
            # wait, tf is [H, W, In, Out], but tflite stores them differently.
            # We will handle transposition at weight loading time if needed, for now just call conv2d
            # TFLite conv2d options (strides, padding)
            opt = op.BuiltinOptions()
            conv_opt = tflite.Conv2DOptions()
            conv_opt.Init(opt.Bytes, opt.Pos)
            s_h = conv_opt.StrideH()
            s_w = conv_opt.StrideW()
            pad = conv_opt.Padding() # 0 = SAME, 1 = VALID
            
            w_idx = inputs[1]
            w_shape = subgraph.Tensors(w_idx).ShapeAsNumpy()
            kernel_h = w_shape[1]
            kernel_w = w_shape[2]
            
            in_val = in_vars[0]
            if pad == tflite.Padding.SAME:
                cpp_code.append(f"        auto {out_var}_pad = apply_same_padding({in_val}, {s_h}, {s_w}, {kernel_h}, {kernel_w});")
                in_val = f"{out_var}_pad"
                
            cpp_code.append(f"        auto {out_var}_conv = mlx::core::conv2d({in_val}, {in_vars[1]}, {{{s_h}, {s_w}}});")
            if len(in_vars) > 2:
                cpp_code.append(f"        auto {out_var} = mlx::core::add({out_var}_conv, {in_vars[2]});")
            else:
                cpp_code.append(f"        auto {out_var} = {out_var}_conv;")
                
            apply_fused_activation(cpp_code, out_var, conv_opt.FusedActivationFunction())
                
        elif code_name == "DEPTHWISE_CONV_2D":
            opt = op.BuiltinOptions()
            conv_opt = tflite.DepthwiseConv2DOptions()
            conv_opt.Init(opt.Bytes, opt.Pos)
            s_h = conv_opt.StrideH()
            s_w = conv_opt.StrideW()
            pad = conv_opt.Padding()
            
            in_idx = inputs[0]
            in_shape = subgraph.Tensors(in_idx).ShapeAsNumpy()
            in_channels = in_shape[-1]
            
            w_idx = inputs[1]
            w_shape = subgraph.Tensors(w_idx).ShapeAsNumpy()
            kernel_h = w_shape[1]
            kernel_w = w_shape[2]
            
            in_val = in_vars[0]
            if pad == tflite.Padding.SAME:
                cpp_code.append(f"        auto {out_var}_pad = apply_same_padding({in_val}, {s_h}, {s_w}, {kernel_h}, {kernel_w});")
                in_val = f"{out_var}_pad"
            
            w_var = in_vars[1]
            cpp_code.append(f"        auto {out_var}_conv = mlx::core::conv2d({in_val}, mlx::core::transpose({w_var}, {{3, 1, 2, 0}}), {{{s_h}, {s_w}}}, {{0, 0}}, {{1, 1}}, {in_channels});")
            if len(in_vars) > 2:
                cpp_code.append(f"        auto {out_var} = mlx::core::add({out_var}_conv, {in_vars[2]});")
            else:
                cpp_code.append(f"        auto {out_var} = {out_var}_conv;")
                
            apply_fused_activation(cpp_code, out_var, conv_opt.FusedActivationFunction())
                
        elif code_name == "ADD":
            opt = op.BuiltinOptions()
            add_opt = tflite.AddOptions()
            add_opt.Init(opt.Bytes, opt.Pos)
            cpp_code.append(f"        auto {out_var} = mlx::core::add({in_vars[0]}, {in_vars[1]});")
            apply_fused_activation(cpp_code, out_var, add_opt.FusedActivationFunction())
            
        elif code_name == "PRELU":
            # PReLU: maximum(0, x) + alpha * minimum(0, x)
            cpp_code.append(f"        auto {out_var} = mlx::core::add(mlx::core::maximum({in_vars[0]}, mlx::core::array(0.0f)), mlx::core::multiply({in_vars[1]}, mlx::core::minimum({in_vars[0]}, mlx::core::array(0.0f))));")
            
        elif code_name == "RESHAPE":
            # Reshape needs the new shape
            shape_var = in_vars[1]
            cpp_code.append(f"        auto {out_var} = mlx::core::reshape({in_vars[0]}, extract_shape({shape_var}));")
            
        elif code_name == "CONCATENATION":
            opt = op.BuiltinOptions()
            conc_opt = tflite.ConcatenationOptions()
            conc_opt.Init(opt.Bytes, opt.Pos)
            axis = conc_opt.Axis()
            cpp_code.append(f"        auto {out_var} = mlx::core::concatenate({{{', '.join(in_vars)}}}, {axis});")
            
        elif code_name == "MAX_POOL_2D":
            # 2x2 max pool with stride 2:
            cpp_code.append(f"        auto {out_var}_h = mlx::core::reshape({in_vars[0]}, {{ {in_vars[0]}.shape(0), {in_vars[0]}.shape(1)/2, 2, {in_vars[0]}.shape(2), {in_vars[0]}.shape(3) }});")
            cpp_code.append(f"        auto {out_var}_h_max = mlx::core::max({out_var}_h, 2);")
            cpp_code.append(f"        auto {out_var}_w = mlx::core::reshape({out_var}_h_max, {{ {out_var}_h_max.shape(0), {out_var}_h_max.shape(1), {out_var}_h_max.shape(2)/2, 2, {out_var}_h_max.shape(3) }});")
            cpp_code.append(f"        auto {out_var} = mlx::core::max({out_var}_w, 3);")
            
        elif code_name == "PAD":
            # PAD takes input and paddings. paddings is usually a constant tensor.
            cpp_code.append(f"        auto {out_var} = mlx::core::pad({in_vars[0]}, extract_padding({in_vars[1]}));")
            
        elif code_name == "RESIZE_BILINEAR":
            # Approximate 2x upsampling with repeat
            cpp_code.append(f"        auto {out_var} = mlx::core::repeat(mlx::core::repeat({in_vars[0]}, 2, 1), 2, 2);")
            
        elif code_name == "MEAN":
            opt = op.BuiltinOptions()
            reducer_opt = tflite.ReducerOptions()
            reducer_opt.Init(opt.Bytes, opt.Pos)
            keep_dims = "true" if reducer_opt.KeepDims() else "false"
            cpp_code.append(f"        auto {out_var} = mlx::core::mean({in_vars[0]}, extract_shape({in_vars[1]}), {keep_dims});")
            
        elif code_name == "FULLY_CONNECTED":
            opt = op.BuiltinOptions()
            fc_opt = tflite.FullyConnectedOptions()
            fc_opt.Init(opt.Bytes, opt.Pos)
            cpp_code.append(f"        auto {out_var}_mm = mlx::core::matmul({in_vars[0]}, mlx::core::transpose({in_vars[1]}));")
            if len(in_vars) > 2:
                cpp_code.append(f"        auto {out_var} = mlx::core::add({out_var}_mm, {in_vars[2]});")
            else:
                cpp_code.append(f"        auto {out_var} = {out_var}_mm;")
            apply_fused_activation(cpp_code, out_var, fc_opt.FusedActivationFunction())
                
        elif code_name == "LOGISTIC":
            cpp_code.append(f"        auto {out_var} = mlx::core::sigmoid({in_vars[0]});")
            
        else:
            cpp_code.append(f"        // UNHANDLED OP: {code_name}")
            
    # Outputs
    out_indices = subgraph.OutputsAsNumpy()
    out_vars = [clean_name(tensor_names[i]) for i in out_indices]
    
    out_str = ", ".join(out_vars)
    cpp_code.append(f"        return {{{out_str}}};")
        
    cpp_code.append("    }")
    cpp_code.append("};")
    cpp_code.append("")
    cpp_code.append("} // namespace mlx_vision")
    
    with open(out_path, "w") as f:
        f.write("\n".join(cpp_code))
    
    print(f"Transpiled to {out_path}")

if __name__ == '__main__':
    main()
