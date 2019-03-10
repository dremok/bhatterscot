import os

import falcon
from yattag import Doc

from bhatterscot.chatbot import ChatBot
from bhatterscot.config import save_dir, model_name, CORPUS_NAME, encoder_n_layers, decoder_n_layers, hidden_size, \
    checkpoint_to_load


class ChatResource(object):
    def __init__(self):
        load_filename = os.path.join(save_dir, model_name, CORPUS_NAME,
                                     '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                                     '{}_checkpoint.tar'.format(checkpoint_to_load))
        self._chat_bot = ChatBot(load_filename)

    def on_post(self, req, resp):
        input_query = req.get_param('query')
        resp.status = falcon.HTTP_200
        resp.content_type = 'text/html'
        bot_response = self._chat_bot.chat(input_query)
        resp.body = (self.html(input_query, bot_response))

    def on_get(self, req, resp):
        resp.status = falcon.HTTP_200
        resp.content_type = 'text/html'
        resp.body = (self.html('', 'Hello. I am a bot.'))

    def html(self, you_said, bot_says):
        doc, tag, text = Doc().tagtext()
        with tag('html'):
            with tag('head'):
                with tag('title'):
                    text('Chat with a Video Game NPC')
                with tag('script'):
                    text("onload = function() {\n")
                    text(f"var msg = new SpeechSynthesisUtterance('{bot_says}');\n")
                    text('window.speechSynthesis.speak(msg);')
                    text('}')
            with tag('body', id='chat'):
                with tag('h1'):
                    text('Chat with Video Game NPC')
                with tag('h3'):
                    if len(you_said) > 0:
                        text(f'You said: {you_said}')
                        doc.stag('br')
                    text(f'Bot says: {bot_says}')
                with tag('form', action='/chat', method='post', enctype='application/json'):
                    doc.input(name='query', id='query', type='text', value='')
                    doc.stag('br')
                    doc.stag('input',
                             type='submit',
                             value='Say it!',
                             style='-webkit-appearance: button; height:50px; width:300px; font-size:20px')
        return doc.getvalue()


app = falcon.API()
app.req_options.auto_parse_form_urlencoded = True

app.add_route('/chat', ChatResource())
