from flask import Flask, Response, render_template, request
import os
from wsgiref import simple_server
from imagePreprocessing.imageTransformation import DataTransform
from models.deep_model import Trainmodel
import torch
import imageView
from Predict import Prediction
#from flask_cors import cross_origin
import json

app = Flask(__name__)

@app.route("/", methods=['GET'])

def home():
    return render_template('index.html')

@app.route('/train' , methods = ['GET'])
def train():
    try:
        datatransform = DataTransform(batch_size = 32)
        train_loader, validation_loader = datatransform.transformation(size = (100,100))
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        train_model = Trainmodel()
        model = train_model.classified(num_labels = 3)
        train_model.training(model,train_loader, validation_loader, device )

        print('Training of the model is complete')
        print('File is saved in SavedModel directory')
        return 'Training Complete!!!'
    except ValueError:
        return Response("Error Occurred! %s" % ValueError)
    except KeyError:
        return Response("Error Occurred! %s" % KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" % e)
    return Response("Training successfull!!")

@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':

        try:
            if request.form is not None:
                path = request.form['filepath']
                print(path)
                image = imageView.process_image(path)
                # Give image to model to predict output
                model = torch.load('SavedModel/model.pth')
                model.eval()
                top_prob, top_class, path, json_prediction = Prediction.predict(image, model)
                # Show the image
                imageView.show_image(image)
                # Print the results
                print("The model is ", top_prob*100, "% certain that the image has a predicted class of ", top_class)
                prediction = "The model is " + str(top_prob * 100) + "% certain that the image has a predicted class of " + str(top_class)
                return Response(prediction + str(path)  + str(json.loads(json_prediction) ))

            else:
                print('Nothing Matched!!!')

        except ValueError:
            return Response("Error Occurred! %s" % ValueError)
        except KeyError:
            return Response("Error Occurred! %s" % KeyError)
        except Exception as e:
            return Response("Error Occurred! %s" % e)

port = int(os.getenv("PORT",5000))
if __name__ == '__main__':
    app.run(debug = True)
    host = '0.0.0.0'
    #port = 5000
    httpd = simple_server.make_server(host, port, app)
    # print("Serving on %s %d" % (host, port))
    httpd.serve_forever()