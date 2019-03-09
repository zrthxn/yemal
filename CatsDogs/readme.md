### Classifier
# Cats vs Dogs

Microsoft Cats vs Dogs challenge on Kaggle

Dataset https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765

Try to classify an image of a dog or a cat 
using a CNN (Convolutional Neural Network).

## Notes
- The initial accuracy was only 0.65 due to a missing activation function after the dense layer
- After the activation function, the accuracy was around 0.75 after 3 epochs
- Bumping up to 10 epochs, the accuracy was 0.85 but the validation loss was big and val accuracy was 0.7
- Some signs of overfitting were observed. Last recorded loss was only 0.1
- Maximum accuracy was obtained after
  - 32 batch size
  - 16 epochs (takes forever to train)
  - test split of 25%

After testing with out of sample images, accuracy was 0.7 with cats being recognized less often.
Insample test gave 100% accuracy, so some overfitting definitely happened.