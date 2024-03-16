# gunicorn_config.py

bind = '127.0.0.1:8001'  # Specify the host and port to bind to
workers = 1  # Number of worker processes
timeout = 120000  # Timeout for worker processes