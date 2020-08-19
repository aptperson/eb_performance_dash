# application.py
from src.app_current_month import app 

# app = dash_app()
app.title='APTCapital ASX Performance'########### Set up the layout
application = app.server

if __name__ == '__main__':
    # app.run_server(debug=True)
    application.run(debug=True, port=8080)