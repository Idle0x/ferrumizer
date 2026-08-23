# ADR-002: Lumped Thermal Surrogate in Calibration

Status: ACCEPTED

The full explicit thermal PDE is the production/simulation model. Calibration
uses the closed-form lumped-capacitance limit, which is independently checked
by V1. This keeps NUTS tractable on CPU while retaining a differentiable
emissivity path. The full three-Tesseract composition remains available via
`FerrumizerPipeline.forward_containers` and is used for boundary verification.

The surrogate is not presented as a replacement for the resolved thermal
field. Its use is called out in calibration reports and is a v1 limitation.
