# application.py
from src.app_current_month import dash_app 

if __name__ == '__main__':
    app = dash_app()
    application = app.server
    # app.run_server(debug=True)
    application.run(debug=True, port=8080)