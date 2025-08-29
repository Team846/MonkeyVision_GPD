import math
from dash import Dash, html, dcc, Input, Output
import cv2
from flask import Flask, Response
from pipeline.visionmain import VisionMain
import time
from threading import Thread
from util.config import ConfigCategory, Config
import os
from localization.visiony import CONF, ASPECT_THRESH
from localization.partial_solution import WD, ONT


def make_slider(slider_id, label, min_val, max_val, step, value, marks):
    return html.Div(
        [
            html.Label(
                label,
                style={
                    "color": "#CCC9CA",
                    "font-size": "16px",
                    "margin-bottom": "10px",
                    "padding": "15px 20px 0 15px",
                    "white-space": "nowrap",
                    "max-width": "250px",
                },
            ),
            dcc.Slider(
                id=slider_id,
                min=min_val,
                max=max_val,
                step=step,
                value=value,
                marks=marks,
                tooltip={"placement": "bottom", "always_visible": True},
                className="funky-slider",
            ),
            html.Br(),
        ]
    )


class HTMLServer:
    config_category = ConfigCategory("HTMLServer")
    framecomp_slider = config_category.getFloatConfig("framecomp_slider", 0.5)

    def __init__(self, vision_main: VisionMain):
        self.vision_main = vision_main
        self.server = Flask(__name__)
        self.app = Dash(__name__, server=self.server)
        self.app.index_string = self.index_string()

        self.SLIDERS = [
            {
                "id": "framecomp-slider",
                "label": "Displayed Frame Quality",
                "min": 0.05,
                "max": 1,
                "step": 0.001,
                "value": HTMLServer.framecomp_slider.valueFloat(),
                "marks": {0.05: "5%", 1: "100%"},
                "setter": lambda v: HTMLServer.framecomp_slider.setFloat(v),
            },
            {
                "id": "conf_s",
                "label": "Confidence Thresh (%)",
                "min": 0.05,
                "max": 1.0,
                "step": 0.01,
                "value": CONF.valueFloat(),
                "marks": {0.05: "5%", 1: "100%"},
                "setter": lambda v: CONF.setFloat(v),
            },
            {
                "id": "aspect_slider",
                "label": "Aspect Thresh (%)",
                "min": 1.0,
                "max": 3.0,
                "step": 0.1,
                "value": ASPECT_THRESH.valueFloat(),
                "marks": {1.01: "1f", 2.99: "3f"},
                "setter": lambda v: ASPECT_THRESH.setFloat(v - 1.0),
            },
            {
                "id": "wd-slider",
                "label": "Object Width (in)",
                "min": 10.0,
                "max": 20.0,
                "step": 0.2,
                "value": WD.valueFloat(),
                "marks": {0.1: "10in", 19.9: "20in"},
                "setter": lambda v: WD.setFloat(v),
            },
            {
                "id": "ont-slider",
                "label": "Is-On-Top Threshold (in)",
                "min": -100.0,
                "max": 100.0,
                "step": 1.0,
                "value": ONT.valueFloat(),
                "marks": {-99.9: "-100in", 99.9: "100in"},
                "setter": lambda v: ONT.setFloat(v),
            },
        ]

        slider_components = [
            make_slider(
                cfg["id"],
                cfg["label"],
                cfg["min"],
                cfg["max"],
                cfg["step"],
                cfg["value"],
                cfg["marks"],
            )
            for cfg in self.SLIDERS
        ]

        self.app.layout = html.Div(
            [
                html.Div(
                    [
                        html.H1(
                            "MonkeyVision GPD",
                            style={
                                "textAlign": "left",
                                "color": "#CCC9CA",
                                "font-size": "32px",
                                "font-weight": "bold",
                                "padding": "10px 20px 0 25px",
                                "margin-bottom": "5px",
                            },
                        ),
                        html.H4(
                            "By Team 846 • The Funky Monkeys",
                            style={
                                "textAlign": "left",
                                "color": "#CCC9CA",
                                "font-size": "14px",
                                "margin": "5px 0 10px 25px",
                            },
                        ),
                        html.Img(
                            src="/assets/logo.svg",
                            style={
                                "position": "absolute",
                                "top": "20px",
                                "right": "20px",
                                "width": "50px",
                                "height": "50px",
                            },
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H4(
                                    "Detections",
                                    style={
                                        "color": "#CCC9CA",
                                        "font-size": "18px",
                                        "padding": "0px 0px 0px 7px",
                                    },
                                ),
                                html.Div(id="detections-container"),
                                html.Br(),
                                html.H4(
                                    "Settings",
                                    style={
                                        "color": "#CCC9CA",
                                        "font-size": "18px",
                                        "padding": "0px 0px 0px 7px",
                                    },
                                ),
                                html.Br(),
                                *slider_components,
                                html.Br(),
                                html.Div(
                                    html.Button(
                                        "Test Pipeline",
                                        id="test-image-button",
                                        style={
                                            "margin-top": "10px",
                                            "font-size": "14px",
                                            "color": "#161616",
                                            "background-color": "rgba(255, 204, 74, 1)",
                                            "border": "none",
                                            "padding": "8px 16px",
                                            "width": "300px",
                                            "height": "40px",
                                            "border-radius": "10px",
                                            "cursor": "pointer",
                                            "font-weight": "bold",
                                        },
                                    ),
                                    style={
                                        "display": "flex",
                                        "justify-content": "center",
                                        "align-items": "center",
                                        "width": "100%",
                                    },
                                ),
                            ],
                            style={
                                "flex": "1",
                                "padding": "10px",
                                "max-width": "35%",
                                "overflow-y": "auto",
                                "height": "100vh",
                            },
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label(
                                            f"Pipeline #{self.vision_main.get_pipeline_number()}",
                                            style={
                                                "textAlign": "right",
                                                "color": "#CCC9CA",
                                                "font-size": "24px",
                                                "font-weight": "bold",
                                            },
                                        ),
                                        html.Img(
                                            src="/video_feed",
                                            style={
                                                "width": "100%",
                                                "max-width": "650px",
                                                "max-height": "600px",
                                                "border": "3px solid #CDA646",
                                                "border-radius": "9px",
                                            },
                                        ),
                                        html.Div(
                                            id="metrics-display",
                                            style={
                                                "width": "100%",
                                                "height": "40px",
                                                "margin-top": "5px",
                                            },
                                        ),
                                        html.Br(),
                                        html.Button(
                                            "Reboot",
                                            id="reboot-button",
                                            style={
                                                "margin-top": "10px",
                                                "font-size": "14px",
                                                "color": "#161616",
                                                "background-color": "rgba(255, 204, 74, 1)",
                                                "border": "none",
                                                "padding": "8px 16px",
                                                "width": "300px",
                                                "height": "40px",
                                                "border-radius": "20px",
                                                "cursor": "pointer",
                                                "font-weight": "bold",
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "flex",
                                        "flex-direction": "column",
                                        "align-items": "center",
                                    },
                                )
                            ],
                            style={"flex": "2"},
                        ),
                    ],
                    style={"display": "flex", "flex-direction": "row"},
                ),
                *[
                    html.Div(id=f"fake-output1-{i}", style={"display": "none"})
                    for i in range(1, len(self.SLIDERS) + 1)
                ],
                html.Div(id=f"fake-output-2", style={"display": "none"}),
                html.Div(id=f"fake-output-3", style={"display": "none"}),
                dcc.Interval(id="update-interval", interval=1000, n_intervals=0),
            ],
            style={
                "background-color": "#161616",
                "color": "#FFF",
                "font-family": "'Inter', sans-serif",
            },
        )

        self.app.callback(
            Output("fake-output-2", "children"), [Input("reboot-button", "n_clicks")]
        )(self.reboot_system)
        self.app.callback(
            Output("fake-output-3", "children"),
            [Input("test-image-button", "n_clicks")],
        )(self.test_image)
        self.app.callback(
            Output("metrics-display", "children"),
            [Input("update-interval", "n_intervals")],
        )(self.update_metrics)
        self.app.callback(
            Output("detections-container", "children"),
            [Input("update-interval", "n_intervals")],
        )(self.update_detections)

        for i, cfg in enumerate(self.SLIDERS, start=1):

            def make_callback(setter, sid=cfg["id"]):
                def callback(value):
                    print(f"{sid} updated -> {value}")
                    setter(value)
                    return f"Slider {sid} value is {value}"

                return callback

            self.app.callback(
                Output(f"fake-output1-{i}", "children"), [Input(cfg["id"], "value")]
            )(make_callback(cfg["setter"]))

        self.server.add_url_rule("/video_feed", "video_feed", self.video_feed)
        self.start_server_thread()

    def reboot_system(self, n_clicks):
        if n_clicks:
            os.system("sudo reboot")
        return "Rebooting system"

    def test_image(self, n_clicks):
        if n_clicks:
            self.vision_main.cam.setUseTestImage(50)
        return "Testing image"

    def start_server(self):
        self.app.run_server(
            host="0.0.0.0",
            port=5800 + self.vision_main.get_pipeline_number(),
            debug=True,
            use_reloader=False,
        )

    def start_server_thread(self):
        Thread(target=self.start_server, daemon=True).start()

    def video_feed(self):
        return Response(
            self.generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    def generate_frames(self):
        while True:
            time.sleep(0.08)
            frame = self.vision_main.get_frame()
            if frame is None:
                continue
            rs_dim = HTMLServer.framecomp_slider.valueFloat()
            frame = cv2.resize(
                frame, (0, 0), fx=rs_dim, fy=rs_dim, interpolation=cv2.INTER_AREA
            )
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    def get_metrics(self):
        return {
            "framerate": self.vision_main.get_framerate(),
            "processing_latency": self.vision_main.get_processing_latency() * 1e3,
        }

    def update_metrics(self, n_intervals):
        metrics = self.get_metrics()
        return [
            html.Div(
                [
                    html.Span(
                        f"Framerate: {metrics['framerate']:.2f} fps",
                        style={
                            "position": "absolute",
                            "left": "0",
                            "bottom": "0",
                            "color": "rgba(255, 255, 255, 0.8)",
                            "font-size": "18px",
                            "font-style": "italic",
                            "padding": "5px 10px",
                            "width": "50%",
                        },
                    ),
                    html.Span(
                        f"Latency: {metrics['processing_latency']:.2f} ms",
                        style={
                            "position": "absolute",
                            "right": "0",
                            "bottom": "0",
                            "color": "rgba(255, 255, 255, 0.8)",
                            "font-size": "18px",
                            "font-style": "italic",
                            "padding": "5px 10px",
                        },
                    ),
                ],
                style={
                    "position": "relative",
                    "width": "60%",
                    "height": "40px",
                    "marginLeft": "20%",
                    "marginRight": "20%",
                },
            )
        ]

    def update_detections(self, n_intervals):
        detections = self.vision_main.get_detections()
        if not detections:
            return [
                html.Div(
                    "No detections",
                    style={
                        "color": "#CCC9CA",
                        "font-size": "14px",
                        "text-align": "center",
                        "border": "2px solid rgba(255, 255, 255, 0.5)",
                        "padding": "10px",
                        "margin": "0 0px 20px 20px",
                        "border-radius": "10px",
                    },
                )
            ]
        return [
            html.Div(
                [
                    html.Span(
                        f"Detection #{i + 1}:",
                        style={
                            "color": "#CCC9CA",
                            "margin-right": "10px",
                            "font-weight": "medium",
                        },
                    ),
                    html.Span(
                        f"R {d.r:.2f}in",
                        style={"color": "#CCC9CA", "margin-right": "5px"},
                    ),
                    html.Span(
                        f"θ {d.theta:.2f}deg",
                        style={"color": "#CCC9CA", "margin-right": "15px"},
                    ),
                    html.Span(f"On: {d.isOnTop()}", style={"color": "#CCC9CA"}),
                ],
                style={
                    "border": "2px solid rgba(255, 255, 255, 0.5)",
                    "border-radius": "10px",
                    "padding": "10px",
                    "font-size": "14px",
                    "margin": "0 0px 20px 20px",
                    "display": "flex",
                    "gap": "10px",
                },
            )
            for i, d in enumerate(detections)
        ]

    def index_string(self):
        return """
        <!DOCTYPE html>
        <html lang=\"en\">
        <head>
            <meta charset=\"UTF-8\">
            <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
            <title>MonkeyVision GPD</title>
            <style>
                body { background-color: #161616; color: white; font-family: 'Inter', sans-serif; margin: 0; padding: 0; }
                .funky-slider .rc-slider-track { background-color: #CDA646; }
                .funky-slider .rc-slider-rail { background-color: rgba(255, 255, 255, 0.5); }
                .funky-slider .rc-slider-handle { border-color: #CDA646; background-color: #CDA646; }
            </style>
        </head>
        <body>
            {%app_entry%}
            {%config%}
            {%scripts%}
            {%renderer%}
        </body>
        </html>
        """
