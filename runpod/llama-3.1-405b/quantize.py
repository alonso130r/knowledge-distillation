from exllamav2 import ExllamaQuantizer, ExllamaModel

model = ExllamaModel.from_pretrained('/llama-405b')

quantizer = ExllamaQuantizer(model)

quantized_model = quantizer.quantize()

quantized_model.save_pretrained('/llama-405b-quantized')