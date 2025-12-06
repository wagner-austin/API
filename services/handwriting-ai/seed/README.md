Place your seeded model artifacts here so the container can copy them to the
mounted volume on first boot.

Expected layout (relative to this folder):

  digits/models/<MODEL_ID>/model.pt
  digits/models/<MODEL_ID>/manifest.json

Example:

  digits/models/mnist_resnet18_v1/model.pt
  digits/models/mnist_resnet18_v1/manifest.json

At runtime, the container entrypoint copies files from /seed/digits/models/<MODEL_ID>/
into /data/digits/models/<MODEL_ID>/ only if the destination files are missing. This
allows you to attach a persistent volume at /data and keep your models across
restarts; the seed is used only the first time.

