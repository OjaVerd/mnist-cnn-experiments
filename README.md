# MNIST CNN Experiments

A Jupyter notebook project exploring convolutional neural networks (CNNs) for handwritten digit classification with the MNIST dataset. The main notebook builds a baseline CNN, explains its layer dimensions and parameter counts, and compares several training and regularization choices.

## Project contents

- [`ConvNeuralNetwork.ipynb`](./ConvNeuralNetwork.ipynb) — the main experiment: a baseline CNN, dropout and L1/L2 regularization comparisons, SGD versus Adam, a Global Average Pooling architecture variant, and test-set error analysis.
- [`JupyterPractice.ipynb`](./JupyterPractice.ipynb) — an additional introductory MNIST and Keras practice notebook. Some cells expect a local `testing.keras` model file, which is not included.

## Model and experiments

The baseline network uses this sequence:

```text
Input (28 x 28 x 1)
  -> Conv2D (32 filters, 3 x 3, ReLU)
  -> Conv2D (64 filters, 3 x 3, ReLU)
  -> MaxPooling2D (2 x 2)
  -> Flatten
  -> Dense (128, ReLU)
  -> Dense (10, softmax)
```

The notebook uses the 60,000 MNIST training images, with 50,000 for training and 10,000 for validation, and evaluates a selected model on MNIST's 10,000-image test set. Experiments include dense dropout, spatial dropout, L1/L2 regularization, SGD versus Adam, and replacing `Flatten` with `GlobalAveragePooling2D`.

### Recorded results

The notebook's comparison table reports best validation accuracy of up to **98.95%** for dense dropout (0.5) and Adam. The Global Average Pooling variant reached **76.58%** after 15 epochs with far fewer parameters; its validation accuracy was still rising at that point. These are results recorded in the notebook, not a guarantee that a fresh run will reproduce them exactly.

## Run the main notebook

Use Python 3.12 or a compatible Python version supported by your TensorFlow installation.

1. Clone or download this repository.
2. Create and activate a virtual environment.
3. Install the notebook dependencies:

   ```bash
   python -m pip install tensorflow numpy matplotlib jupyter
   ```

4. Start Jupyter from the project directory:

   ```bash
   jupyter notebook
   ```

5. Open `ConvNeuralNetwork.ipynb` and run its cells from top to bottom. TensorFlow/Keras downloads MNIST automatically when it is first requested.

Training all experiment cells can take substantially longer than running only the baseline because multiple models are trained for 15 epochs each.

## Notes

- The notebooks contain saved cell outputs; those outputs may differ from a fresh run.
- `JupyterPractice.ipynb` includes model-loading cells that need a `testing.keras` file. That artifact is not part of this project.
- This is an educational experiment, not a packaged application or a production-ready model.
