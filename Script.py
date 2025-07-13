import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_text
import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import callbacks 

# nltk.download('punkt_tab')

df = pd.read_csv("tweetfile.csv")
print(df.dtypes)
print(df.head(5))

#data imbalance??
print(df['text_label'].value_counts())
#percentage
print(df['text_label'].value_counts(normalize=True))
#plot
df['text_label'].value_counts(normalize=True).plot(kind='bar')
plt.show()
#Balance the dataset
bullying = df[df['text_label'] == 'bullying']
nonbullying = df[df['text_label'] == 'nonbullying']
nonbullying_df= nonbullying.sample(n=len(bullying),random_state=55)
#new dataset
data= pd.concat([nonbullying_df,bullying]).reset_index(drop=True)
#rechecking
data['text_label'].value_counts(normalize=True).plot(kind='bar')
plt.show()
print(data.head())
#data shuffling( frac=1 means we must include all rows)
review_df= data.sample(frac=1, random_state=42)
#cleaning
stop_words= set(stopwords.words('english'))
custom_remove = ["''",'``',"rt","https","’","“”,","\u200b","--","n't","'s","...","//t.c"]
def clean_tweet(text):
    text = text.lower()  # lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    words = word_tokenize(text)  # tokenize
    clean_words = [word for word in words if word not in stop_words and word not in custom_remove]
    return " ".join(clean_words)
#apply the cleaning
data['clean_tweet'] = data['tweet'].apply(clean_tweet)
print(data.head(5))

#text embedding
use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
sent_1 = ['fuck you']
sent_2 = ['go kill yourself']

emb_1 = use(sent_1)
emb_2 = use(sent_2)

print(np.inner(emb_1, emb_2).flatten()[0])

#one hot encoding
label_encoder = OneHotEncoder(sparse_output=False)
encoded_labels = label_encoder.fit_transform(data['text_label'].values.reshape(-1, 1))

#train/test set
train_texts, test_texts, y_train, y_test= train_test_split(data['clean_tweet'],
                                                            encoded_labels,
                                                            test_size=0.1,
                                                            random_state=42)
#Embedding
X_train=[]
for tweet in tqdm(train_texts):
    emb=use([tweet])
    emb_np=tf.reshape(emb,[-1]).numpy()
    X_train.append(emb_np)
X_train = np.array(X_train)

X_test=[]
for tweet in tqdm(test_texts):
    emb=use([tweet])
    emb_np=tf.reshape(emb,[-1]).numpy()
    X_test.append(emb_np)
X_test = np.array(X_test)

#Model
model = keras.Sequential([
    keras.layers.Dense(256,activation='relu',input_shape=(512,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(2, activation='softmax')
])
#compile
model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#train
elst=callbacks.EarlyStopping(monitor='val_loss',patience=3,mode='min',restore_best_weights=True)
history=model.fit(X_train,y_train,epochs=10,batch_size=16,validation_split=0.1,shuffle=True,
                  callbacks=[elst],verbose=1)
#evaluation
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.show()

loss, acc = model.evaluate(X_test, y_test)
print("Test accuracy:", acc)
#save
model.save("FinalModel.keras")