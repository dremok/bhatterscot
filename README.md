# Bhatterscot

A trainable chatbot implemented using PyTorch.

The bot can be re-trained using any text data. 
It has speech recognition to parse spoken language and has a web interface with text-2-speech capabilities.

## Configuring the model

In `config.py` you can set the hyperparameters of the model. Set `BASE_DIR` to the location of where to save training data and trained models.

## Get data

The model can be trained on any text data. Currently, the entire corpus will be parsed as one long conversation, where each pair of lines will be regarded as an input/output pair.

You can use `scrape.py` to build a training corpus from a bunch of video game dialogues. It will be saved in the configured `BASE_DIR`.

## Training

Run `train.py` to train the model using the current configuration. The model will be saved in the configured `BASE_DIR` under `/models/<CORPUS_NAME>`.

## Inference

You can use the `ChatBot` class in `chatbot.py` to chat with the trained bot. Instantiate the class with a file containing a trained model. Then call `chat(query)` to get a response. You can change the value of `EXPLORE_PROB` to adjust how unpredictable the responses of the chatbot will be.

## Web API

If you run `main.py` in a web server, e.g. [gunicorn](https://gunicorn.org/), you can chat with the bot in a web interface.

Example: `gunicorn main:app -b localhost:8000`

Now, browse to `localhost:8000` to chat with your bot.

Cheers!
