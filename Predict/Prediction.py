import torch
import pandas as pd

def predict( image, model):

    # Pass the image through our model
    output = model.forward(image)

    # Reverse the log function in our output
    output = torch.exp(output)

    # Get the top predicted class, and the output percentage for
    # that class
    probs, classes = output.topk(1, dim=1)
    print(probs, classes)
    x, y = probs.item(), classes.item()
    z = list([x,y])

    result = pd.DataFrame(z, index = ['accuracy', 'class']).T

    path = "Predictions.csv"
    result.to_csv(path)
    return probs.item(), classes.item(), path, result.head().to_json(orient="records")