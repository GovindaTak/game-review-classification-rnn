# game-review-classification-rnn
This project is a Game Review Classification model using RNN and Word2Vec embeddings.  It processes user reviews to predict whether a player would recommend the game or not.  The pipeline includes text preprocessing (lemmatization, stopword removal, vectorization),  feature extraction using TF-IDF and Word2Vec, and training an RNN model.
-----------

## **Game Review Classification using RNN** 🎮  
📌 **Predicting Game Review Sentiments with Word2Vec & RNN**  

### **Overview**  
This project focuses on classifying **game reviews** based on user sentiment. We analyze user feedback and predict whether they would **recommend** the game or not. The model is built using **Recurrent Neural Networks (RNNs)** with **Word2Vec embeddings**.  

----

### **Project Pipeline**
✔️ **Step 1: Importing Dataset**  
- Used a dataset containing **game reviews** and their corresponding **recommendation labels (0/1)**.  

✔️ **Step 2: Text Preprocessing**  
- **Tokenization & Lemmatization** using `spaCy`.  
- **Stopword removal & punctuation cleanup**.  
- **POS Tagging & Named Entity Recognition (NER)** to enhance features.  

✔️ **Step 3: Embeddings (Feature Engineering)**  
- Used **TF-IDF Vectorization** for feature extraction.  
- Built **Word2Vec embeddings** (100-dimensional) trained on our dataset.  

✔️ **Step 4: Modeling with RNN**  
- Implemented an **RNN Model in PyTorch**.  
- Used **Leaky ReLU activation** and **dropout regularization**.  
- Trained for **30 epochs with early stopping**.  

---

### **Dataset Details**  
- **Features:** `user_review` (game review text)  
- **Target Variable:** `user_suggestion` (0 = Not Recommended, 1 = Recommended)  
- **Total Records:** 17,877 (Training), 1,000+ (Validation & Testing)  

---

### **Model Architecture**  
- Input Size: `100` (Word2Vec embedding dimension)  
- Hidden Size: `128`  
- Layers: `1`  
- Activation: `Leaky ReLU`  
- Dropout: `0.5`  
- Optimizer: `Adam (lr=0.001)`  
- Loss Function: `Binary Cross-Entropy Loss`  

---

### **Results & Accuracy**  
🏆 **Best Model Performance:**  
✅ **Training Accuracy:** `56.72%`  
✅ **Validation Accuracy:** `56.87%`  
✅ **Lowest Validation Loss:** `0.6840`  

📊 **Next Steps:**  
- Experiment with **LSTMs & GRUs** to improve accuracy.  
- Fine-tune **Word2Vec embeddings** with external datasets.  
- Explore **Transformer-based models (BERT, GPT)**.  

---

### **Installation & Usage**  
#### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/GovindaTak/game-review-classification-rnn.git
cd game-review-classification-rnn
```
#### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```
#### **3️⃣ Run the Model**
```bash
python train_model.py
```

---

### **Technologies Used**  
🛠 **Libraries & Tools:**  
- **Python** (Pandas, NumPy, Matplotlib, Seaborn)  
- **Natural Language Processing (NLP):** `NLTK`, `spaCy`, `TF-IDF`, `Word2Vec`  
- **Machine Learning:** `scikit-learn`  
- **Deep Learning:** `PyTorch`  
- **Data Visualization:** `Matplotlib`, `Seaborn`  

---

### **Contributing**  
Want to improve the model? Feel free to fork the repo, create a new branch, and submit a pull request. 🚀  

---

### **Connect with Me**  
📌 Gmail :- govindatak19@gmail.com

💡 **Let's build better AI models together!** 🚀🔥  
