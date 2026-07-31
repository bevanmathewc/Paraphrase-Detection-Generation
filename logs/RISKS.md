# Project Risk Register

| ID | Risk | Likelihood | Impact | Mitigation | Status |
|----|------|------------|--------|------------|--------|
| R1 | Google Colab GPU unavailable | Medium | High | Continue with local development tasks until GPU is available | Open |
| R2 | Dataset download failure | Low | Medium | Cache datasets locally after first successful download | Open |
| R3 | Version incompatibility between Python packages | High | High | Use a fixed `environment.yml` and pin package versions | Open |
| R4 | Model training interrupted | Medium | High | Save checkpoints after each epoch | Open |
| R5 | Out-of-memory (OOM) during transformer training | Medium | High | Reduce batch size or sequence length | Open |
| R6 | Overfitting on small datasets | Medium | Medium | Monitor validation metrics and use early stopping | Open |
| R7 | Poor paraphrase generation quality | Medium | Medium | Compare multiple decoding strategies and inspect outputs | Open |
| R8 | Loss of work due to missing backups | Low | High | Commit changes regularly and push to GitHub | Open |
| R9 | Scope creep | Medium | Medium | Prioritize core deliverables before stretch goals | Open |
| R10 | Inconsistent experimental results | Medium | Medium | Fix random seeds and track experiments systematically | Open |