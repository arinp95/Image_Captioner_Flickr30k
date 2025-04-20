# 🧠 Image Caption Generator using DenseNet201 and LSTM

This project is an end-to-end deep learning solution that automatically generates natural language descriptions for images. It integrates a powerful feature extractor (DenseNet201) with a language modeling component (LSTM), trained on the **Flickr30k** dataset. The final model is deployed using a **Flask** web app for real-time image captioning.

---

## 📂 Project Structure
```
Image-Captioner/
├── 01_data_preparation.ipynb     # Caption preprocessing and dataset organization
├── 02_model_training.ipynb       # Model definition, training pipeline, and checkpointing
├── 03_model_evaluaion.ipynb      # Caption inference and BLEU evaluation script
├── app.py                        # Flask application backend
├── static/uploads/               # Directory to store uploaded images via UI
├── templates/index.html          # HTML UI for image upload and caption display
├── model.keras                   # Saved Keras model
├── tokenizer.pkl                 # Trained tokenizer object
├── config.pkl                    # Model configuration (e.g., max_length, vocab size)
├── features.pkl.xz               # Compressed DenseNet feature representations
├── captions.pkl                  # Cleaned caption DataFrame
├── requirements.txt              # Python dependencies
```

---

## 🧠 Model Architecture

### 🔹 Encoder (Feature Extractor)
- Pretrained **DenseNet201** from Keras Applications
- Global Average Pooling applied to last convolutional layer output

### 🔹 Decoder (Language Generator)
- Embedding Layer initialized randomly
- LSTM Layer with 256 hidden units
- Dense Layer with ReLU followed by Softmax over vocabulary

The model is trained to predict the next word given the image embedding and previously generated words.

---

## 📈 Evaluation Metrics
Model performance is evaluated using BLEU-n metrics over a held-out validation set:
```
BLEU-1: 0.4355
BLEU-2: 0.2256
BLEU-3: 0.1191
BLEU-4: 0.0594
```
> Scores were calculated using 5 reference captions per image with NLTK’s BLEU implementation and smoothing function.

---

## 🌐 Deployment with Flask

A simple web UI is included for testing the model by uploading an image.

### Features
- Upload any image from your device
- Automatically view the predicted caption

### Run the App Locally
```bash
# Step 1: Set up environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Launch app
python app.py
```

Then navigate to `http://127.0.0.1:5000` in your browser.

---

## 📦 Dependencies
- `tensorflow`
- `flask`
- `numpy`, `pandas`, `matplotlib`, `Pillow`
- `nltk`, `tqdm`

To install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 🗃️ Dataset
- [Flickr30k Dataset](https://www.kaggle.com/datasets/eeshawn/flickr30k)

---

## 👤 Author
**Arindam Phatowali**  
B.Tech + M.Tech (Mathematics & Data Science), MANIT Bhopal

---

## 📬 Contact & Links
📧 Email: arindamphatowali@gmail.com  
🐍 GitHub: [github.com/arinp95](https://github.com/arinp95)

---

## ⭐ Like this project?
Please consider starring ⭐ the repository to show your support and help others discover it!
