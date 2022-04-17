from app import app
from app.utils import get_port
from app.views import index, style_transfer_view  # noqa: F401

if __name__ == "__main__":
    app.run(debug=True, port=get_port("BACKEND"))
