# application.py
from src.app_current_month import dash_app 

app = dash_app()
application = app.server

if __name__ == '__main__':
    # app.run_server(debug=True)
    application.run(debug=True, port=8080)