# Security and Production Testing

## Security Considerations

The FastAPI request schema validates all eight clinical attributes before
inference. Missing fields, invalid types, and values outside defined ranges are
rejected with HTTP 422. Unexpected internal failures are logged while clients
receive a generic HTTP 500 response; stack traces, file paths, and model details
are not exposed.

The deployed service should process only the clinical attributes required for
prediction, avoid storing request payloads, and exclude patient inputs from
operational logs. In a real deployment, model artefacts, training data, API
access, and logs should be protected using authentication, role-based access,
encryption, retention controls, and audit monitoring. Authentication is a
deployment recommendation and is not claimed as part of this prototype.

## Shadow Deployment

A new model can be evaluated using shadow deployment. Live requests continue
to receive official predictions from the existing production model, while a
copy of each request is also evaluated by the candidate model. Candidate
outputs are recorded for comparison but do not affect user-facing or clinical
decisions. Accuracy, latency, reliability, and error rates can then be compared
under realistic workloads. The candidate model is promoted only after it meets
predefined quality and operational thresholds.

