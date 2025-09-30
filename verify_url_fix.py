#!/usr/bin/env python3
"""Test current FluxDevVertexProvider URL generation to verify the fix."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_current_url_generation():
    """Test that the current code generates correct URLs."""
    
    print("Testing current FluxDevVertexProvider URL generation...")
    
    # Test the exact URL construction logic from the current code
    endpoint_url = "https://9047233572619943936.us-central1-894885603233.prediction.vertexai.goog"
    
    # Apply the URL construction logic from the provider
    if endpoint_url.endswith('.prediction.vertexai.goog') and endpoint_url.startswith('http'):
        # Already has protocol, ensure it has the correct predict path
        if not endpoint_url.endswith('/predict'):
            endpoint_url = f"{endpoint_url.rstrip('/')}/predict"
    
    print(f"Primary URL: {endpoint_url}")
    
    # Test fallback URL generation
    endpoint_paths_to_try = [endpoint_url]
    
    # If this is a dedicated endpoint, add alternative paths to try
    if '.prediction.vertexai.goog' in endpoint_url:
        # Extract base URL correctly
        if '/predict' in endpoint_url:
            base_url = endpoint_url.rsplit('/predict', 1)[0]
        elif '/v1/predict' in endpoint_url:
            base_url = endpoint_url.rsplit('/v1/predict', 1)[0]
        else:
            base_url = endpoint_url.rstrip('/')
            
        alternative_paths = [
            f"{base_url}/predict",  # Primary path per documentation
            f"{base_url}/v1/predict", 
            f"{base_url}/v1/models/flux-dev/predict"  # Should be /predict not :predict
        ]
        # Add alternatives that aren't already in the list
        for path in alternative_paths:
            if path not in endpoint_paths_to_try:
                endpoint_paths_to_try.append(path)
    
    print("\nGenerated URLs:")
    error_found = False
    for i, url in enumerate(endpoint_paths_to_try):
        if ':predict' in url:
            print(f"  {i+1}. ❌ {url} (CONTAINS :predict - ERROR!)")
            error_found = True
        else:
            print(f"  {i+1}. ✅ {url}")
    
    # Check for the specific problematic URL from the error
    problematic_url = "https://9047233572619943936.us-central1-894885603233.prediction.vertexai.goog:predict"
    if problematic_url in endpoint_paths_to_try:
        print(f"\n❌ ERROR: The problematic URL {problematic_url} is still being generated!")
        error_found = True
    else:
        print(f"\n✅ GOOD: The problematic URL {problematic_url} is NOT being generated.")
    
    if error_found:
        print("\n🚨 There are still issues with URL generation!")
        return False
    else:
        print("\n🎉 URL generation looks correct!")
        return True

if __name__ == "__main__":
    success = test_current_url_generation()
    sys.exit(0 if success else 1)