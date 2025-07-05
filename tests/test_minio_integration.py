#!/usr/bin/env python3
"""
Test script for MinIO integration in InSwapper worker
"""

import os
import sys
import base64
import requests
import json
from PIL import Image
import io

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_test_image(width=256, height=256, color=(255, 0, 0)):
    """Create a simple test image"""
    img = Image.new('RGB', (width, height), color)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def test_minio_integration():
    """Test MinIO integration with the worker"""
    
    # Create test images
    print("🖼️ Creating test images...")
    source_image = create_test_image(256, 256, (255, 0, 0))  # Red image
    target_image = create_test_image(256, 256, (0, 255, 0))  # Green image
    
    # Test payload
    test_payload = {
        "source_image": source_image,
        "target_image": target_image,
        "source_indexes": "0",
        "target_indexes": "0",
        "background_enhance": True,
        "face_restore": True,
        "face_upsample": True,
        "upscale": 1,
        "codeformer_fidelity": 0.5,
        "output_format": "JPEG",
        "use_minio_output": True
    }
    
    print("📤 Testing MinIO output...")
    print(f"Payload keys: {list(test_payload.keys())}")
    print(f"Source image length: {len(source_image)}")
    print(f"Target image length: {len(target_image)}")
    
    # Note: This is a mock test since we can't actually call the worker
    # In a real scenario, you would send this to your RunPod endpoint
    
    print("✅ MinIO integration test payload prepared")
    print("📋 To test with actual worker, send payload to your RunPod endpoint")
    
    return test_payload

def test_url_input():
    """Test URL input functionality"""
    
    print("🌐 Testing URL input...")
    
    # Test payload with URL inputs
    test_payload = {
        "source_image": "https://example.com/source.jpg",
        "target_image": "https://example.com/target.jpg",
        "source_indexes": "0",
        "target_indexes": "0",
        "background_enhance": True,
        "face_restore": True,
        "face_upsample": True,
        "upscale": 1,
        "codeformer_fidelity": 0.5,
        "output_format": "JPEG",
        "use_minio_output": True
    }
    
    print("✅ URL input test payload prepared")
    return test_payload

def test_base64_output():
    """Test base64 output (no MinIO)"""
    
    print("📄 Testing base64 output...")
    
    # Create test images
    source_image = create_test_image(256, 256, (255, 0, 0))
    target_image = create_test_image(256, 256, (0, 255, 0))
    
    test_payload = {
        "source_image": source_image,
        "target_image": target_image,
        "source_indexes": "0",
        "target_indexes": "0",
        "background_enhance": True,
        "face_restore": True,
        "face_upsample": True,
        "upscale": 1,
        "codeformer_fidelity": 0.5,
        "output_format": "JPEG",
        "use_minio_output": False  # Use base64 output
    }
    
    print("✅ Base64 output test payload prepared")
    return test_payload

if __name__ == "__main__":
    print("🧪 Running MinIO Integration Tests")
    print("=" * 50)
    
    # Test 1: MinIO output
    minio_payload = test_minio_integration()
    
    # Test 2: URL input
    url_payload = test_url_input()
    
    # Test 3: Base64 output
    base64_payload = test_base64_output()
    
    print("\n📊 Test Summary:")
    print(f"✅ MinIO Output Test: {len(minio_payload)} parameters")
    print(f"✅ URL Input Test: {len(url_payload)} parameters")
    print(f"✅ Base64 Output Test: {len(base64_payload)} parameters")
    
    print("\n🎯 All test payloads prepared successfully!")
    print("💡 Use these payloads to test your RunPod worker deployment") 