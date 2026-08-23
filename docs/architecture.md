# Architecture

The thermal, carburizing, and hardening components are independent Tesseract APIs. The app can run them locally through `Tesseract.from_tesseract_api` or use built images through environment variables. `apply_tesseract` carries gradients across the two composition boundaries.
