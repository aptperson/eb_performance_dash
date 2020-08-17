# application.py
from src.app_current_month import dash_app 

if __name__ == '__main__':
    app = dash_app()
    app.run_server(debug=True)