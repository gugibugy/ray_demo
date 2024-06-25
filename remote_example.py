import os
import ray
import requests
from ray.util.actor_pool import ActorPool
from datetime import datetime
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification


@ray.remote
def get_image(url: str):
  image_data = requests.get(url, stream=True).raw
  image_name = url.split("/")[-1].split(".")[0]
  image = Image.open(image_data)
  return image, image_name


@ray.remote
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


@ray.remote
def write_results(results, destination):
  with open(destination, "w") as output:
    output.write(results)

if __name__ == "__main__":
  start = datetime.now()
  with open("images_to_download.txt") as image_list:
    image_futures = []
    start = datetime.now()
    for image_src in image_list.read().splitlines():
      image_futures.append(get_image.remote(image_src))

  images = ray.get(image_futures)

  processor_ref = ray.put(ViTImageProcessor.from_pretrained('google/vit-base-patch16-224'))
  model_ref = ray.put(ViTForImageClassification.from_pretrained('google/vit-base-patch16-224'))
  actor_pool = ActorPool(Predictor.remote(model_ref, processor_ref) for _ in range(4))
  for image, image_name in images:
    actor_pool.submit(lambda actor, value: actor.predict.remote(value), (image, image_name))

  predictions = []
  while actor_pool.has_next():
      predictions.append(actor_pool.get_next())

  for prediction in predictions:
    destination = os.path.join("/data/ray_demo/results_remote", f"{prediction[1]}.txt")
    write_results.remote(prediction[0], destination)
  end = datetime.now()
  print(f"Total execution time: {end-start}")
