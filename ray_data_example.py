import os
import ray
import requests
from datetime import datetime
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
from typing import Dict, Any


def get_image(row: Dict[str, Any]) -> Dict[str, Any]:
  url = row["text"]
  image_data = requests.get(url, stream=True).raw
  row["image"] = Image.open(image_data)
  row["image_name"] = url.split("/")[-1].split(".")[0]
  return row


class Predictor:
    def __init__(self, model: ViTForImageClassification, processor: ViTImageProcessor):
      self.processor = ray.get(processor)
      self.model = ray.get(model)

    def __call__(self, row: Dict[str, Any]):
      image = row["image"]
      inputs = self.processor(images=image, return_tensors="pt")
      outputs = self.model(**inputs)
      logits = outputs.logits
      # model predicts one of the 1000 ImageNet classes
      predicted_class_idx = logits.argmax(-1).item()
      row["label"] = self.model.config.id2label[predicted_class_idx]
      return row


def write_results(row: Dict[str, Any]):
  destination = os.path.join("/data/ray_demo/results_ray_data", f"{row['image_name']}.txt")
  with open(destination, "w") as output:
    output.write(row["label"])
  return row

if __name__ == "__main__":
  start = datetime.now()
  dataset = ray.data.read_text("images_to_download.txt", concurrency=6)
  dataset = dataset.map(get_image, concurrency=6)
  processor_ref = ray.put(ViTImageProcessor.from_pretrained('google/vit-base-patch16-224'))
  model_ref = ray.put(ViTForImageClassification.from_pretrained('google/vit-base-patch16-224'))
  dataset = dataset.map(Predictor, fn_constructor_args=[model_ref, processor_ref], concurrency=6)
  dataset = dataset.map(write_results, concurrency=6)
  dataset.materialize()
  end = datetime.now()
  print(f"Total execution time: {end-start}")
