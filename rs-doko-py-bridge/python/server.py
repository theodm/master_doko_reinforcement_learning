
from bottle import route, run, template, app, request
from bottle_cors_plugin import cors_plugin

from rs_doko_py_bridge import p_create_game_for_web_view

@route('/game', method='GET')
def game():
    seed_param = request.query['seed']

    seed = int(seed_param)

    return p_create_game_for_web_view(
        seed
    )


app = app()
# ToDo: Warum?
app.install(cors_plugin('*'))

run(host='localhost', port=8081)
