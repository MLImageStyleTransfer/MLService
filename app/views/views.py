import flask
import typing as tp
from flask import request, jsonify

from app import app
from app.utils import get_port
from app.controller import controller


@app.route("/")
def index() -> flask.Response:
    """
    This function draws hello-page for ml service
    """
    main_page: str = f"""
        <h1 width="100%" align="center">
            Backend ml-service!
            <br/>
            Started on port: {get_port('BACKEND')}
        </h1>
    """
    return flask.Response(main_page)


@app.route("/style_transfer/", methods=["POST"])
def style_transfer_view() -> flask.Response:
    """
    This function processes request for style transfer
    Request structure:
    {
        "content_image_code": str,
        "style_image_code": str,
        "params" : {
            "transfer_coefficient": float
        }
    }
    :return: flask.Response with structure:
    {
        "image_code": str
    }
    """
    request_data: tp.Optional[tp.Any] = request.get_json()
    response: flask.Response

    if isinstance(request_data, dict) and \
            ({"content_image_code", "style_image_code", "params"} <= request_data.keys()) and \
            isinstance(request_data["params"], dict) and \
            ("transfer_coefficient" in request_data["params"]):
        try:
            result_image_code: str = controller(
                request_data["content_image_code"],
                request_data["style_image_code"],
                request_data["params"]["transfer_coefficient"]
            )
            response = jsonify({"image_code": result_image_code})
        except AssertionError:
            response = flask.Response(status=400)
    else:
        response = flask.Response(status=400)

    response.headers.add("Access-Control-Allow-Origin", f"http://localhost:{get_port('FRONTEND')}")
    return response
