import feature_extractor as fe

# Test code
test_image = fe.readImage('TestImages/image_k.png')
print(fe.getFeaturesOf(test_image))
fe.showFeaturesOf(test_image)