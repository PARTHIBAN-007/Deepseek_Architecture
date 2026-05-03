# DeepSeek Architecture
A PyTorch implementation of core DeepSeek architectures, including attention mechanisms, positional encoding, multi-token prediction, mixture-of-experts, and quantization.

## Attention Mechanism
The attention mechanism allows a model to dynamically focus on the most relevant parts of its input when producing an output.
- Self Attention
- Causal Attention
- Multi Head Attention
- KV Cache
- Multi Query Attention
- Grouped Query Attention
- Multi-Head Latent Attention

## Positional Encoding
Positional encoding injects information about token order into a sequence model, enabling it to capture positional relationships in otherwise permutation-invariant architectures.
- Integer Position Encoding
- Binary Position Encoding
- Sinusoidal Position Encoding
- Rotary Position Encoding

## Mixture of Experts
Mixture of Experts is a neural network architecture that consists of multiple specialized sub-networks (experts) and a gating mechanism that dynamically selects or weights a subset of experts for each input, enabling conditional computation and increased model capacity with reduced computational cost.
- Sparse MoE
- Balancing Techniques
- Auxiliary Loss
- Load Balancing Loss
- Capacity Factor
- Auxiliary Loss-Free Load Balancing
- Shared Expert
- DeepSeek MoE

## Multi Token Prediction
Multi-token prediction is a training or inference strategy in which a model predicts multiple future tokens simultaneously, rather than one token at a time, to improve efficiency, parallelism, or contextual coherence.


## Quantization
Quantization is the process of reducing the numerical precision of model parameters, activations, or gradients (e.g., from 32-bit floating point to lower-bit representations) to decrease memory usage and improve computational efficiency, often with minimal impact on model performance.
- Mixed Precision
- Fine Grained Quantization
- Increasing Accumulation Precision
- Mantissa Over Exponent
- Online Quantization

## References:
- Youtube: [Vizura](https://youtube.com/playlist?list=PLPTV0NXA_ZSiOpKKlHCyOq9lnp-dLvlms&si=FhNHIMy_oTLrnr76)
- Github: [Vizura AI Labs](https://github.com/VizuaraAILabs/DeepSeek-From-Scratch)

### Contributions
Contributions are welcome. If you'd like to add new notebooks, improve implementations, or fix issues, feel free to submit a pull request or raise issues.


