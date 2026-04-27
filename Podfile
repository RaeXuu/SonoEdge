platform :ios, '16.0'
use_frameworks!

target 'SonoEdge' do
  pod 'TensorFlowLiteSwift'
end

post_install do |installer|
  tensor_path = "Pods/TensorFlowLiteSwift/tensorflow/lite/swift/Sources/Tensor.swift"
  if File.exist?(tensor_path)
    content = File.read(tensor_path)
    unless content.include?("kTfLiteInt8:")
      content.sub!("case kTfLiteUInt8:", "case kTfLiteUInt8:\n      case kTfLiteInt8:\n        self = .uInt8\n")
      File.write(tensor_path, content)
    end
  end
end