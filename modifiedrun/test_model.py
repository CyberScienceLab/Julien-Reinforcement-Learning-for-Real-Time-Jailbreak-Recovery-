#!/usr/bin/env python3
"""
Test script to check if the ONNX model can be loaded properly
"""

import onnxruntime as ort
import numpy as np
from pathlib import Path

def test_model_loading():
    """Test if the ONNX model can be loaded"""
    print("🔍 Testing ONNX model loading...")
    
    model_path = Path("modelF.onnx")
    
    if not model_path.exists():
        print("❌ Model file 'modelF.onnx' not found!")
        print("   Please ensure the model file is in the Pipeline directory")
        return False
    
    print(f"✅ Model file found: {model_path}")
    print(f"   Size: {model_path.stat().st_size / 1024:.1f} KB")
    
    try:
        # Try to load the model
        session = ort.InferenceSession("modelF.onnx")
        print("✅ Model loaded successfully!")
        
        # Get model info
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        print(f"📊 Model inputs: {len(inputs)}")
        for i, input_info in enumerate(inputs):
            print(f"   Input {i}: {input_info.name}, shape: {input_info.shape}")
        
        print(f"📊 Model outputs: {len(outputs)}")
        for i, output_info in enumerate(outputs):
            print(f"   Output {i}: {output_info.name}, shape: {output_info.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def test_model_inference():
    """Test if the model can perform inference"""
    print("\n🧪 Testing model inference...")
    
    try:
        session = ort.InferenceSession("modelF.onnx")
        
        # Create a test input
        input_shape = session.get_inputs()[0].shape
        print(f"📊 Expected input shape: {input_shape}")
        
        # Create test data (assuming 389-dimensional input)
        test_input = np.random.randn(1, 389).astype(np.float32)
        print(f"📊 Test input shape: {test_input.shape}")
        
        # Get input name
        input_name = session.get_inputs()[0].name
        print(f"📊 Input name: {input_name}")
        
        # Run inference
        inputs = {input_name: test_input}
        outputs = session.run(None, inputs)
        
        print("✅ Inference successful!")
        print(f"📊 Output shape: {outputs[0].shape}")
        print(f"📊 Output values: {outputs[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False

def main():
    print("🔍 ONNX Model Test")
    print("="*50)
    
    # Test model loading
    if not test_model_loading():
        print("\n❌ Cannot proceed - model loading failed!")
        return
    
    # Test model inference
    if test_model_inference():
        print("\n✅ Model is working correctly!")
        print("   The backend should be able to use this model")
    else:
        print("\n❌ Model inference failed!")
        print("   Check if the model file is corrupted or incompatible")

if __name__ == "__main__":
    main() 