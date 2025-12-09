import launch

# We need the gallery-dl LIBRARY, not just the tool.
if not launch.is_installed("gallery_dl"):
    launch.run_pip("install gallery-dl", "requirements for Booru Grabber")