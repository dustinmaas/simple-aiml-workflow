c = get_config()

# Allow connections from all IP addresses
c.NotebookApp.ip = '0.0.0.0'

# Don't automatically open a browser
c.NotebookApp.open_browser = False

# Set the notebook directory
c.NotebookApp.notebook_dir = '/app/notebooks'

# Allow remote access without a token or password
# This is not secure but simplifies development
c.NotebookApp.token = ''
c.NotebookApp.password = ''
