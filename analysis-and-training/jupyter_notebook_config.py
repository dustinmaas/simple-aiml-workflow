import os

c = get_config()

# Allow connections from configurable IP address
c.NotebookApp.ip = os.environ.get('JUPYTER_IP', '0.0.0.0')

# Set port from environment variable
c.NotebookApp.port = int(os.environ.get('JUPYTER_PORT', 8888))

# Don't automatically open a browser
c.NotebookApp.open_browser = False

# Set the notebook directory
c.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager"
c.ContentsManager.default_jupytext_formats = "ipynb,py:percent"

# Allow remote access without a token or password
# This is not secure but simplifies development
c.NotebookApp.token = ''
c.NotebookApp.password = ''
