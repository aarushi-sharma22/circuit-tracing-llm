# Understanding How Language Models Perform Arithmetic: A Mechanistic Interpretability Study

## Overview

This project investigates how Large Language Models (LLMs) perform basic arithmetic operations internally, inspired by Anthropic'sresearch "Tracing the thoughts of a large language model". We aim to uncover the hidden computational circuits that enable models to solve problems like "55+56" despite being trained primarily on text prediction.


## Background: The Mystery of Emergent Computation

Language models aren't explicitly programmed to do math. They're trained to predict the next word in a sequence. Yet somehow, through this training process, they develop internal mechanisms for arithmetic. Anthropic's research revealed that models like Claude use:

- **Parallel computational pathways** rather than sequential human-like algorithms
- **One pathway for rough approximation** of the answer
- **Another pathway for precise digit calculation**
- **Internal representations** that the model itself isn't "aware" of

This project replicates and extends these findings by building tools to peer inside the model's computation process.

## Goals

1. **Identify computational circuits** responsible for arithmetic in transformer models
2. **Map information flow** from input tokens ("2", "3", "+", "2", "8", "=") to output ("9", "5")
3. **Discover parallel pathways** for approximation vs. exact computation
4. **Verify findings** through targeted interventions
5. **Build interpretable understanding** of model internals

## Methodology

Our approach follows a three-stage process, progressively building understanding from simple observations to complex feature extraction.

### Stage 1: Direct Analysis (No SAE)

This stage uses built-in model properties to trace information flow without additional complexity.

#### 1.1 Attention-Based Tracking

**What it does**: Traces how tokens "pay attention" to each other across layers, revealing information highways.

```python
def track_information_flow(model, prompt="23+28="):
    # Get attention patterns
    attentions = model.get_attention_patterns(prompt)
    
    # Track how "23" forms
    digit_binding = attentions["2"]["3"] + attentions["3"]["2"]
    
    # Track operation recognition
    operation_attention = attentions["+"]["="]
    
    return digit_binding, operation_attention
```

**Why this matters**: 
- Shows how individual digits ("2", "3") bind together to form numbers ("23")
- Reveals when the model recognizes this is an addition problem
- Identifies which tokens are "talking to" each other

**What we're looking for**:
- Strong attention between consecutive digits (forming multi-digit numbers)
- Attention from "=" to number tokens (triggering computation)
- Attention patterns that suggest grouping or calculation

#### 1.2 Logit Lens Technique

**What it does**: Projects internal representations at each layer into vocabulary space to see when the answer starts forming.

```python
def find_computation_layer(model, prompt="23+28="):
    for layer in range(model.n_layers):
        # Project to vocabulary space
        hidden_state = model.get_hidden_state(prompt, layer)
        logits = model.unembed(hidden_state)
        
        # Check if "95" starts appearing
        if logits["9"] > threshold:
            return layer
```

**Why this matters**:
- Pinpoints the exact layer where computation happens
- Shows how confidence in the answer builds across layers
- Helps identify critical layers for deeper analysis

**What we're looking for**:
- Sudden appearance of correct answer tokens
- Gradual build-up vs. sudden emergence
- Competition between different possible answers

### Stage 2: Targeted SAE Analysis

After identifying key layers and positions, we use Sparse Autoencoders to extract interpretable features.

#### 2.1 Feature Extraction with SAEs

**What it does**: Decomposes dense neural activations into sparse, interpretable features.

```python
def extract_math_features(model, prompts, key_layers=[10, 15, 20]):
    # Train SAEs on specific layers
    for layer in key_layers:
        activations = collect_activations(model, prompts, layer)
        sae = train_sparse_autoencoder(activations)
        
        # Find math-relevant features
        math_features = identify_features(sae, 
            tests=["carries", "magnitude", "digits"])
```

**Why this matters**:
- Transforms opaque neural activations into human-interpretable concepts
- Reveals features like "detecting carry operations" or "computing tens digit"
- Enables circuit discovery by following features across layers

**What we're looking for**:
- Features that activate for specific number ranges
- Separate features for approximation vs. exact calculation
- Features that detect mathematical operations

### Stage 3: Circuit Mapping and Verification

Combines insights from previous stages to map complete computational circuits.

## Key Concepts Explained

### Information Flow in Transformers
- **Parallel Processing**: All input tokens are processed simultaneously
- **Attention Mechanism**: Tokens can "communicate" with each other
- **Layer-wise Refinement**: Each layer builds upon previous representations

### Why These Methods?

1. **Attention Patterns**: Natural starting point - directly interpretable
2. **Logit Lens**: Shows us WHAT the model is computing
3. **SAEs**: Reveals HOW the model is computing it

### Expected Discoveries

Based on Anthropic's findings, we expect to discover:

1. **Parallel Pathways**:
   - Fast approximation circuit (gets rough magnitude)
   - Precise calculation circuit (computes exact digits)

2. **Key Computational Moments**:
   - Number formation (layers 0-5)
   - Operation recognition (layers 5-10)
   - Actual computation (layers 10-20)
   - Answer crystallization (layers 20+)

3. **Non-Human Algorithms**:
   - Model doesn't use standard carry-based addition
   - Develops its own efficient shortcuts
   - May process digits in parallel rather than sequentially

## Success Criteria

1. **Identify distinct computational pathways** for arithmetic
2. **Predict model behavior** based on circuit understanding
3. **Manipulate computations** through targeted interventions
4. **Build generalizable understanding** that applies to other math operations

## Applications

Understanding these mechanisms helps us:
- Build more interpretable AI systems
- Detect when models might make arithmetic errors
- Design better mathematical reasoning capabilities
- Ensure AI systems are doing what we think they're doing

## Getting Started

1. Install required libraries (TransformerLens, PyTorch, etc.)
2. Load a pre-trained model
3. Run Stage 1 analysis to identify key layers
4. Focus deeper analysis on discovered hotspots
5. Validate findings through intervention experiments

## References

- [Anthropic: Tracing the thoughts of a large language model](https://www.anthropic.com/research/tracing-thoughts-language-model)
- [Transformer Circuits Thread](https://transformer-circuits.pub/)
- [Mechanistic Interpretability Resources](https://www.neelnanda.io/mechanistic-interpretability/getting-started)

---

