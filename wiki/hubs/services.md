# Services

The FastAPI ML/NLP/media services that make up the api platform surface. Each service runs on its own port with a per-service README as the front door — this hub covers cross-service architecture, shared patterns, and subsystem-depth pages that expand on what each README opens with. Services: data-bank-api (8001, content-addressed storage), Model-Trainer (8005, LM training), Art-Trainer (8011, image LoRAs), handwriting-ai (8004, MNIST), turkic-api (8000, Turkic language detection + IPA), transcript-api (8003, YouTube transcription), qr-api (8002), music-wrapped-api (8006), covenant-radar-api (8007, risk prediction + Kafka), grandma-api (8008, multi-language audio-to-English), github-stats-api (8009, SVG cards), opportunity-radar-api (8010, hackathon discovery), procart-api (port not yet fixed).

[Service Port Map](../pages/service-port-map.md) -- every FastAPI service + its assigned 80xx port + purpose; hypercorn `--bind` port-binding convention + Traefik routing

[Covenant Radar's backend registry](../pages/covenant-radar-backend-registry.md) -- one interface over seven classifiers and five regressors, split across `covenant_ml` and `covenant_nn`; why the `Literal` is canonical and not either registry; and the README count that drifted one low

[Cartridge composition ceiling](../pages/model-trainer-composition-ceiling.md) -- Model-Trainer's composition-scaling sweep: two compartments is the measured limit (63% retention at n2, negative by n4, erased by n8, bit-identical record), the cost flips from structural to content interference with scale, and the cross-gain arm caught its own roster's leakage

[Cartridge question set](../pages/model-trainer-cartridge-question-set.md) -- the arm that asks whether a cartridge's knowledge is USABLE rather than merely less surprising: it halves answer-token surprise (18.46 -> 10.68, p = 0.0066) while its accuracy gain sits inside its own seed spread, oracle retrieval answers every item, and the distractor policy alone flipped the base model between chance and 0.5417

[A declared batch size trained at four times itself](../pages/model-trainer-declared-config-is-not-a-suggestion.md) -- the worker silently rewrote any declared batch of 4 or less on CUDA, and every record kept saying what the payload declared

[Companioned training recipe](../pages/model-trainer-companioned-training-recipe.md) -- the intervention that moves the composition ceiling: train every cartridge with a frozen held-out companion at p=0.5 and four-compartment retention goes from -45.4% to +44.6% (bit-identical record, 15x its floor) for a solo cost of four hundredths, with a dose curve and an overdose endpoint in both companion kinds; the arc ends with the base itself trained to read crowds twice over: the LM-objective LoRA solves the structural half at both scales, and crowd-invariance distillation closes the content half at depth (medium n8 -79.4% -> +38.1%, n4 +63.2% the program record; eight compartments deliverable on both bases, both levers base-side)

<!-- Add pages here as they're written. Format: [Title](../pages/<slug>.md) -- one-line description -->
