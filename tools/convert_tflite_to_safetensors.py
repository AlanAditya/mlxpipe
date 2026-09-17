import sys
import numpy as np
import tflite
from safetensors.numpy import save_file

def get_np_type(tflite_type):
    if tflite_type == tflite.TensorType.FLOAT32:
        return np.float32
    elif tflite_type == tflite.TensorType.FLOAT16:
        return np.float16
    elif tflite_type == tflite.TensorType.INT32:
        return np.int32
    elif tflite_type == tflite.TensorType.INT8:
        return np.int8
    return None

def builtin_code_to_name(code):
    for k, v in tflite.BuiltinOperator.__dict__.items():
        if v == code:
            return k
    return "UNKNOWN"

def main():
    model_path = sys.argv[1]
    with open(model_path, 'rb') as f:
        buf = f.read()
    
    model = tflite.Model.GetRootAsModel(bytearray(buf), 0)
    subgraph = model.Subgraphs(0)
    
    tensors_to_save = {}
    
    for i in range(subgraph.TensorsLength()):
        t = subgraph.Tensors(i)
        name = t.Name().decode('utf-8')
        buffer_idx = t.Buffer()
        
        buffer = model.Buffers(buffer_idx)
        if buffer.DataLength() > 0:
            data = buffer.DataAsNumpy()
            shape = t.ShapeAsNumpy()
            dtype = get_np_type(t.Type())
            
            if dtype is not None:
                tensor_data = data.view(dtype).reshape(shape)
                
                # Dequantize if applicable
                quant = t.Quantization()
                if quant is not None and quant.ScaleLength() > 0:
                    scales = quant.ScaleAsNumpy()
                    zero_points = quant.ZeroPointAsNumpy() if quant.ZeroPointLength() > 0 else np.zeros_like(scales)
                    
                    if len(scales) == 1:
                        tensor_data = (tensor_data.astype(np.float32) - zero_points[0]) * scales[0]
                    else:
                        # Per-channel quantization
                        quant_dim = quant.QuantizedDimension()
                        # Expand dims for broadcasting
                        broadcast_shape = [1] * len(shape)
                        broadcast_shape[quant_dim] = len(scales)
                        scales = scales.reshape(broadcast_shape)
                        zero_points = zero_points.reshape(broadcast_shape)
                        tensor_data = (tensor_data.astype(np.float32) - zero_points) * scales
                    
                tensors_to_save[name] = tensor_data
                print(f"Extracted {name} with shape {shape} (Dequantized: {quant is not None})")
                
    output_path = model_path.replace(".tflite", ".safetensors")
    save_file(tensors_to_save, output_path)
    print(f"Saved {len(tensors_to_save)} tensors to {output_path}")

    print("\n--- Operations ---")
    for i in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(i)
        op_code = model.OperatorCodes(op.OpcodeIndex())
        code_name = builtin_code_to_name(op_code.BuiltinCode())
        
        inputs = [subgraph.Tensors(j).Name().decode('utf-8') for j in op.InputsAsNumpy() if j != -1]
        outputs = [subgraph.Tensors(j).Name().decode('utf-8') for j in op.OutputsAsNumpy() if j != -1]
        
        print(f"{code_name}: Inputs: {inputs} -> Outputs: {outputs}")

if __name__ == '__main__':
    main()
