# Image Caption Generator using Xception + LSTM

This project generates captions for input images by combining image feature extraction using a pre-trained Xception model and caption generation via a Long Short-Term Memory (LSTM) network. It illustrates the encoder-decoder structure applied to vision and language tasks.

---

## 🛠️ Tools & Techniques

- Python, TensorFlow / Keras  
- Pre-trained Xception model (feature extraction)  
- LSTM-based decoder for sequence generation  
- Tokenization, padding, and word index mapping using Keras and `pickle`  
- Matplotlib for displaying generated captions with images

---

## 🧪 How It Works

- Extract image features from input images using Xception  
- Feed features to an LSTM decoder trained on caption sequences  
- Generate captions word-by-word starting from the "start" token  
- Stop when "end" token is predicted  
- Visualize the result with the original image and generated caption

---

## 📄 Files

- `image_caption_generator.ipynb` — Full workflow including training, evaluation, and generation  
- `test_image_caption_generator.py` — Script to test a saved model on a new image  
- `README.md` — Project overview and usage instructions

---

## 🔗 Notes

- This project is **self-contained**: all required files (tokenizer, model weights) are generated when you run the notebook
- The notebook:
  - Loads image data  
  - Extracts features using the **Xception** model  
  - Trains the LSTM decoder  
  - Saves the tokenizer and model files (`tokenizer.p`, `model_9.keras`)
- The `.py` script loads the saved model/tokenizer and generates a caption for a single test image
- **No files are pre-uploaded** — dataset (e.g., Flickr8k), tokenizer, and model weights must be generated locally by running the notebook

---

## 🚫 Limitations

- Dataset (e.g., Flickr8k) must be obtained separately  
- The vocabulary is limited to what's seen during training  
- Requires a GPU or strong CPU for training the LSTM component
