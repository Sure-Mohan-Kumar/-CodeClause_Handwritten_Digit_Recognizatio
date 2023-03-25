# importing all the necessary libraries

from tkinter import *
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tkinter import messagebox
import io

# Loading the MNIST dataset

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizing and reshaping the images

x_train = x_train.astype('float32') / 255.
x_train = np.expand_dims(x_train, axis=-1)
x_test = x_test.astype('float32') / 255.
x_test = np.expand_dims(x_test, axis=-1)

# Creating the model

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compiling the model

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# Training the model

model.fit(x_train, y_train, epochs=5)

# Evaluating the model on the test set

loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}')

# Creating the GUI Interface

class PaintApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Handwritten Digit Recognition")
        self.master.geometry("500x500")
        self.master.resizable(0, 0)

        self.canvas = Canvas(self.master, width=280, height=280, bg="white", bd=5, relief="groove")
        self.canvas.place(x=110, y=10)

        self.button_clear = Button(self.master, text="Clear", width=10, command=self.clear)
        self.button_clear.place(x=110, y=300)

        self.button_predict = Button(self.master, text="Predict", width=10, command=self.predict)
        self.button_predict.place(x=220, y=300)

        self.image = Image.new("RGB", (280, 280), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

        # Loading the dataset
        digits = load_digits()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(digits.data, digits.target, test_size=0.3, random_state=42)

        # Training the model
        self.model = SVC(kernel='linear', C=1)
        self.model.fit(self.X_train, self.y_train)

    def paint(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.canvas.create_oval(x1, y1, x2, y2, fill="black", width=0)
        self.draw.ellipse([x1, y1, x2, y2], fill="black")

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (280, 280), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)

    def predict(self):
        # Resizing image
        img = self.image.resize((28, 28)).convert('L')

        # Inverting the image and converting into the numpy array
        img = np.array(img)
        img = np.invert(img)

        # Normalizing the image
        img = img.astype('float32') / 255.

        # Reshaping the image that matches the input shape of the model
        img = np.reshape(img, (1, 28, 28, 1))

        # Making a prediction
        pred = model.predict(img)

        # Showing the predicted digit
        messagebox.showinfo("Prediction",f"The predicted digit is: {np.argmax(pred[0])}")

root = Tk()
app = PaintApp(root)
root.mainloop()
