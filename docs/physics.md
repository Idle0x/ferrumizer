# Physics

Thermal transport uses explicit FTCS conduction with a nonlinear convective/radiative Robin boundary. Carbon transport uses Arrhenius diffusivity:

```text
D(T) = D0 exp(-Q / (R T))
```

Hardening applies Andrews Ms(C), Koistinen-Marburger martensite, an optional Scheil/JMAK fraction, and a smoothstep carbon-to-hardness mixing rule. ECD is the first inward depth crossing 550 HV under ISO 2639 practice.
