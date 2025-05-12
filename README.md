# 🧠✨ Distill Like a Pro: Knowledge Distillation with JAX & FLAX NNX 🚀

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mridul-sahu/knowledge_distillation_intuitions/blob/main/Knowledge_Distillation_Intuitions.ipynb) 

Ever wondered how to cram the wisdom of a colossal AI (the "teacher" 🧑‍🏫) into a nimble, efficient "student" 🎓 model without losing (too much) performance? This Colab notebook is your dojo!

We're diving deep into the alchemical art of **Knowledge Distillation (KD)**, translating complex understanding into compact power. Forget arcane PyTorch incantations for a moment; here, we wield the sheer power of `JAX` for ludicrous speed, the modern elegance of `FLAX NNX` for a more explicit, stateful PyTorch-like feel, and the precision of `Optax` for optimization. Our proving ground? The classic CIFAR-10 dataset.

This isn't just another KD tutorial. It's an intuitive journey, a JAX/FLAX NNX reimplementation inspired by the official PyTorch KD tutorial, designed to build your gut feeling for *why* and *how* these techniques actually work.

---

## 📜 What Arcane Knowledge Will You Uncover?

* **The Gnosis of Knowledge Distillation:** Understand the core tenets – what it means to learn from "soft targets" and the teacher's "dark knowledge."
* **FLAX NNX Mastery:** Define and train neural networks with the new `flax.nnx` API, embracing its more explicit state management.
* **Building Baselines:** Forge your own teacher and student CNNs from scratch.
* **Classic Distillation Rituals:** Implement standard KD by matching output logits (Hinton et al. style).
* **Intermediate Sorcery with `flax.nnx.Intermediate`:** Learn to capture and leverage the hidden wisdom within your models using `self.sow()`:
    * **Cosine Similarity Charms:** Make student hidden states dance in tune with the teacher's (with a dash of teacher feature pooling).
    * **FitNets Finesse:** Conjure an intermediate regressor to guide the student's feature learning using MSE loss.
* **JAX & FLAX Statecraft:** Navigate the intricacies of managing state, parameters, and RNGs in the NNX paradigm.

---

## 🛠️ Your Arsenal (Tech Stack):

* **`JAX`**: For high-performance numerical computation, automatic differentiation, and XLA-compilation to GPUs/TPUs. Pure functions FTW!
* **`FLAX NNX (`flax.nnx`)`**: The new, more stateful API within FLAX for building neural networks with a PyTorch-esque, object-oriented feel.
* **`Optax`**: For powerful and flexible gradient processing and optimization.
* **`TensorFlow Datasets` & `tf.data`**: For efficient and robust data loading and preprocessing pipelines.
* **`Orbax Checkpointing`**: For saving and loading your precious model states.
* **`Matplotlib` & `TQDM`**: For visualizing our glorious training progress and results.

---

## 🧪 Distillation Experiments You'll Conduct:

1.  **The Ground Truth:** Train a formidable Teacher (`DeepNN_NNX`) and a baseline Student (`LightNN_NNX`) independently. Know your benchmarks!
2.  **Classic Logit Mimicry:** The student learns from the teacher's final output probabilities (soft targets + temperature scaling).
3.  **Hidden State Harmonics (Cosine Loss):**
    * Capture intermediate feature maps from both teacher and student using `self.sow()`.
    * Student learns to align its (flattened) features with a *pooled* version of the teacher's (flattened) features via Cosine Similarity loss.
4.  **FitNets-Style Regression:**
    * Student sprouts a dedicated "regressor" network.
    * This regressor learns to transform the student's intermediate features to match the teacher's "hint" features, optimized with MSE loss.

---

## 🔥 Ignite the Cauldron (How to Run):

1.  **Click the Badge!** 👇 (The "Open in Colab" badge at the top of the notebook).

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mridul-sahu/knowledge_distillation_intuitions/blob/main/Knowledge_Distillation_Intuitions.ipynb) 
3.  The notebook handles all necessary library installations (`!pip install ...`).
4.  Pre-trained model checkpoints are automatically downloaded to `/tmp/flax_nnx_kd_checkpoints/` to speed things up or let you jump to specific experiments. If you want to train from scratch, simply skip the restore steps or modify the code.
5.  Run the cells sequentially and watch the distillation magic unfold!

---

## 🙏 Acknowledgement:

This tutorial is a humble JAX & FLAX NNX translation and adaptation of the concepts presented in the excellent [PyTorch Knowledge Distillation Tutorial](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html).

---

Ready to become a Knowledge Distillation adept? Dive in and let the learning (and distilling) commence!
