Here is a comprehensive textual guideline for building **GWM-RNN**, a resource-efficient Graph World Model using Recurrent Neural Networks.

This guideline covers the conceptual framework, data engineering, architecture design, and training strategy without using code.

---

### Phase 1: Conceptual Framework
To build this model, you must reframe the standard graph problem (Link Prediction) into a **Sequential World Model** problem.

1.  **The World Model Analogy:**
    *   **The Environment:** The Graph.
    *   **The State ($s_t$):** A node's semantic and structural embedding.
    *   **The Trajectory:** A path traversing from a Source Node to a Target Node.
    *   **The Goal:** The model must process the trajectory (Source $\to$ Target) and determine if this is a valid transition (i.e., does a link exist?).

2.  **The Efficiency Hypothesis:**
    *   Graph edges are local interactions. You do not need the global attention mechanism of a multi-billion parameter LLM to determine if two papers share a topic.
    *   A recurrent mechanism (LSTM/GRU) can capture the dependency between the Source and Target features with significantly fewer parameters.

---

### Phase 2: Data Engineering & Processing
This is the most critical step. Unlike LLMs, which can handle raw text, RNNs require dense numerical vectors. You must pre-process your data into a "Feature Store."

#### 1. Text Encoding (The "Encoder")
*   **Objective:** Convert raw text (Titles/Abstracts) into fixed-size vectors.
*   **Tool:** Use a domain-specific Pre-trained Language Model (PLM). For medical graphs, use **PubMedBERT**.
*   **Strategy:** Run inference *once* on all nodes to generate embeddings ($d=768$). Save these to disk. **Do not fine-tune this encoder** during the RNN training; keep it frozen to maintain speed.

#### 2. Structural Aggregation (The "Context")
*   **Objective:** A node is defined not just by its text, but by its neighbors.
*   **Strategy:** Compute "Multi-Hop" embeddings.
    *   **Hop 0:** The node’s own BERT embedding.
    *   **Hop 1:** The average (mean pooling) of its direct neighbors' embeddings.
*   **Result:** Each node is now represented by a sequence of vectors (e.g., Self + Context).

#### 3. Sequence Construction (The Input)
*   **Objective:** Format the data as a time-series sequence for the RNN.
*   **Format:** For every edge pair (Source $u$, Target $v$), construct a sequence tensor.
    *   *Input Sequence:* `[Embedding_u, Context_u, Embedding_v, Context_v]`
*   **Shape:** If your embedding dimension is 768, your input per sample is a matrix of shape `[4, 768]`.

#### 4. Sampling Strategy
*   **Positive Samples:** Use existing edges in the graph.
*   **Negative Samples:** Randomly sample pairs of nodes that do not have an edge.
*   **Balance:** Maintain a 1:1 ratio between positive and negative samples for balanced training.

---

### Phase 3: Model Architecture
The architecture is a **Discriminative Sequential Model**. It consists of three distinct blocks.

#### Block 1: The Feature Projector (Compression)
*   **Purpose:** Reduce dimensionality and adapt the BERT space to the RNN space.
*   **Component:** A simple Linear Layer (MLP) with Layer Normalization.
*   **Function:** Projects the input (e.g., 768 dimensions) down to a manageable size for an RNN (e.g., 256 or 512 dimensions). This acts as a bottleneck to force feature selection.

#### Block 2: The Transition Core (The RNN)
*   **Purpose:** The "brain" of the model. It models the interaction between the Source and the Target.
*   **Component:** A **Bi-Directional LSTM** (Long Short-Term Memory).
*   **Why Bi-Directional?** Link prediction in citation graphs is often symmetric in terms of semantic relevance. The model should analyze the flow from Source $\to$ Target and Target $\to$ Source simultaneously.
*   **Depth:** 2 stacked layers are usually sufficient. More layers may lead to overfitting on small graphs.

#### Block 3: The Decision Head (Classifier)
*   **Purpose:** Convert the RNN's internal state into a probability.
*   **Input:** The final hidden state of the LSTM (or a pooling of all hidden states).
*   **Component:** A standard Feed-Forward Network (MLP) ending in a linear layer.
    *   *For Link Prediction:* Output dimension is 2 (Binary Classification).
    *   *For Relation Prediction:* Output dimension is $K$ (where $K$ is the number of relation types).

---

### Phase 4: Training Strategy
This architecture is lightweight, allowing for aggressive optimization strategies that are impossible with LLMs.

1.  **Optimization:**
    *   **Loss Function:** Cross-Entropy Loss.
    *   **Optimizer:** AdamW (Adam with Weight Decay) is standard.
    *   **Learning Rate:** You can use a much higher learning rate than LLMs (e.g., `1e-3` vs `1e-5`).

2.  **Regularization (Crucial):**
    *   Since RNNs can overfit easily on small datasets (like Cora), applying **Dropout** is essential. Apply it between the Projector and the RNN, and between the RNN layers.
    *   Use **Weight Decay** to prevent the weights from growing too large.

3.  **Performance Tuning:**
    *   **Batch Size:** Because the model is small, you can use very large batch sizes (e.g., 256, 512, or even 1024). This makes training extremely fast.
    *   **Gradient Clipping:** RNNs can suffer from exploding gradients. Always clip the gradients (e.g., norm value of 1.0) during backpropagation.

---

### Phase 5: Experimental Comparison (For Publication)
If you release this as a paper, your evaluation must focus on the **Performance vs. Efficiency** trade-off.

1.  **Metric 1: Accuracy/AUC:** Compare against standard GNNs (GCN, GAT) and LLM-based models (GWM, LLaGA).
2.  **Metric 2: Parameter Count:** Highlight that your model has ~10-20 million parameters vs. 3-8 billion for LLMs.
3.  **Metric 3: Inference Speed:** Measure how many edges per second you can predict. This will be your strongest selling point (likely 100x faster than LLMs).
4.  **Metric 4: Hardware Requirements:** Emphasize that your model can train on a standard consumer GPU or even a CPU, democratizing access to Graph World Models.