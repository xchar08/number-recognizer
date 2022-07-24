import numpy as np
from matplotlib import pyplot as plt
import random

msgs = [
  "Do a barrel roll!",
  "Pretty Soary, ey?",
  "Brought to you by the MIT Mostec trio :)",
  "Let's move onto another one, that was too easy!"
]

def get_mnist():
    with np.load('mnist.npz') as f:
        images, labels = f["x_train"], f["y_train"]
    images = images.astype("float32") / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]
    return images, labels

images, labels = get_mnist()
input_hidden_weights = np.random.uniform(-0.5,0.5,(20,784))
hidden_output_weights = np.random.uniform(-0.5,0.5,(10,20))
input_hidden_bias = np.zeros((20,1))
hidden_output_bias = np.zeros((10,1))

learning_rate = 0.01
correct_evaluations = 0

#for epoch in (how many times we're looping through the learning set)
for epoch in range(10):
  #iterates through image pairs
  for img, lbl in zip(images,labels):
    #adds on the next row of image vals
    img.shape += (1,)
    #adds on the next row of label vals
    lbl.shape += (1,)
    #forwards propogation (input to hidden layer)
    hidden_prelevel = input_hidden_bias + input_hidden_weights @ img
    hidden_level = 1 / (1 + np.exp(-hidden_prelevel))
    #forwards propogation (hidden to output layer)
    output_prelevel = hidden_output_bias + hidden_output_weights @ hidden_level
    output_level = 1 / (1 + np.exp(-output_prelevel))
     #done with the weights
    #compare output values with the label (what number it is)
    #calculates difference between output and the corresponding label value, 
    #squaring and summing each difference, then divdes by number of output neurons
    e = 1 / len(output_level) * np.sum((output_level - lbl) ** 2, axis = 0)
    correct_evaluations += int(np.argmax(output_level) == np.argmax(lbl))
    #resulting value is "error value"

    #backwards propogation
    change_in_output = output_level - lbl 
    hidden_output_weights += -learning_rate * change_in_output @ np.transpose(hidden_level)
    hidden_output_bias += -learning_rate * change_in_output

    #backwards propogation hidden layer to input layer via derivatives 
    change_in_hidden = np.transpose(hidden_output_weights) @ change_in_output * (hidden_level * (1 - hidden_level))
    input_hidden_weights += -learning_rate * change_in_hidden @ np.transpose(img)
    input_hidden_bias += -learning_rate * change_in_hidden

  #show accuracy for this run (60,000 imgs)
  print(f"Acc: {round((correct_evaluations/images.shape[0])*100, 2)}%")
  correct_evaluations = 0

#display results 😋
while True:
  input_index = int (input("Give me a number between 1 and 60000: "))
  index = input_index - 1
  img = images[index]
  plt.imshow(img.reshape(28,28), cmap = "Greys")

  img.shape += (1, )
  #forwards propogation from input to hidden layer 
  hidden_prelevel = input_hidden_bias + input_hidden_weights @ img.reshape(784,1)
  hidden_level = 1 / (1 + np.exp(-hidden_prelevel))
  #forwards propogation from hidden level to output level
  output_prelevel = hidden_output_bias + hidden_output_weights @ hidden_level
  output_level = 1 / (1 + np.exp(-output_prelevel))

  plt.title(f"It is a {output_level.argmax()}. " + random.choice(msgs))
  plt.show()
