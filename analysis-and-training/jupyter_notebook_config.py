c = get_config()

# Allow connections from all IP addresses
c.NotebookApp.ip = '0.0.0.0'

# Don't automatically open a browser
c.NotebookApp.open_browser = False

# Set the notebook directory
c.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager"
c.ContentsManager.default_jupytext_formats = "ipynb,py:percent"

# Allow remote access without a token or password
# This is not secure but simplifies development
c.NotebookApp.token = ''
c.NotebookApp.password = ''
