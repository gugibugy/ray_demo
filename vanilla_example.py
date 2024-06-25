import os
import requests
from datetime import datetime
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification


def get_image(url: str):
  image_data = requests.get(url, stream=True).raw
  image_name = url.split("/")[-1].split(".")[0]
  image = Image.open(image_data)
  return image, image_name


class Predictor:
    def __init__(self, model: ViTForImageClassification, processor: ViTImageProcessor):
      self.processor = processor
      self.model = model

    def predict(self, image_info):
      image, image_name = image_info[0], image_info[1]
      inputs = self.processor(images=image, return_tensors="pt")
      outputs = self.model(**inputs)
      logits = outputs.logits
      # model predicts one of the 1000 ImageNet classes
      predicted_class_idx = logits.argmax(-1).item()
      label = self.model.config.id2label[predicted_class_idx]
      return label, image_name


def write_results(results, destination):
  with open(destination, "w") as output:
    output.write(results)

if __name__ == "__main__":
  start = datetime.now()
  with open("images_to_download.txt") as image_list:
    images = []
    for image_src in image_list.read().splitlines():
      images.append(get_image(image_src))

  predictor = Predictor(ViTForImageClassification.from_pretrained('google/vit-base-patch16-224'),
                        ViTImageProcessor.from_pretrained('google/vit-base-patch16-224'))
  predictions = []
  for image in images:
    predictions.append(predictor.predict(image))

  for prediction in predictions:
    destination = os.path.join("/data/ray_demo/results_vanilla", f"{prediction[1]}.txt")
    write_results(prediction[0], destination)
  end = datetime.now()
  print(f"Total execution time: {end-start}")
