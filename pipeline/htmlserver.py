from dash import Dash, html, dcc, Input, Output
import cv2
from flask import Flask, Response
from pipeline.visionmain import VisionMain
import time
from threading import Thread

class HTMLServer:
    def __init__(self, vision_main: VisionMain):
        self.vision_main = vision_main
        self.server = Flask(__name__)

        self.app = Dash(__name__, server=self.server)

        self.app.index_string = self.index_string()

        self.app.layout = html.Div([
            html.Div([
                html.Div([
                    html.H1("MonkeyVision", style={
                        'textAlign': 'left',
                        'color': '#CCC9CA',
                        'font-size': '32px',
                        'font-weight': 'bold',
                        'padding': '10px 20px 0 25px',
                        'margin-bottom': '5px',
                    }),
                    html.H4("By Team 846 • The Funky Monkeys", style={
                        'textAlign': 'left',
                        'color': '#CCC9CA',
                        'font-size': '14px',
                        'font-weight': 'regular',
                        'margin': '5px 0 10px 25px',
                    }),
                    html.Img(
                        src="/assets/logo.svg",
                        style={
                            "position": "absolute",
                            "top": "20px",
                            "right": "20px",
                            "width": "50px",
                            "height": "50px"
                        }
                    ),
                ]),
                html.Div([
                    html.Div([
                        html.H4("Detections", style={
                            'textAlign': 'left',
                            'color': '#CCC9CA',
                            'font-size': '18px',
                            'font-weight': 'medium',
                            'padding': '0px 0px 0px 15px',
                        }),
                        html.Div(
                            id='detections-container',
                            style={
                                'flex-direction': 'column',
                                'gap': '5px',
                                'margin-top': '5px',
                                'padding': '0 0px',
                            }
                        ),
                        html.Label("Dynamic Brightness Target", style={
                            "color": "#CCC9CA",
                            "font-size": "16px",
                            "margin-bottom": "10px",
                            'padding': '15px 20px 0 15px',
                            "white-space": "nowrap",
                            "max-width": "250px",
                        }),
                        dcc.Slider(
                            id="brightness-slider",
                            min=0,
                            max=255,
                            step=1,
                            value=128,
                            marks={0: '0', 255: '255'},
                            tooltip={"placement": "bottom", "always_visible": True},
                            className="funky-slider"
                        ),
                        html.Br(),
                        html.Label("Frame Resolution", style={
                            "color": "#CCC9CA",
                            "font-size": "16px",
                            "margin-bottom": "10px",
                            'padding': '15px 20px 0 15px',
                            "white-space": "nowrap",
                            "max-width": "250px",
                        }),
                        dcc.Slider(
                            id="other-setting-slider",
                            min=0.125,
                            max=1,
                            step=0.001,
                            value=0.562,
                            marks={0.125: '12.5%', 1: '100%'},
                            tooltip={"placement": "bottom", "always_visible": True},
                            className="funky-slider"
                        ),
                    ], style={
                        "flex": "1",
                        "padding": "10px",
                        "color": "#FFF",
                        "font-family": "'Inter', sans-serif",
                        "flex-direction": "column",
                        "display": "flex",
                        "flex-grow": "1",
                        "max-width": "35%",
                        "box-sizing": "border-box",
                    }),
                    html.Div([
                        html.Div([
                            html.Label("Pipeline", style={
                                "textAlign": "right",
                                "color": "#CCC9CA",
                                "font-size": "24px",
                                "font-weight": "bold",
                                "margin-bottom": "5px",
                                "padding-right": "5px"
                            }),
                            html.Img(src="/video_feed", style={
                                "width": "100%",
                                "max-width": "650px",
                                "max-height": "600px",
                                "border": "3px solid #CDA646",
                                "border-radius": "9px",
                            }),
                            html.Div(id="metrics-display", style={
                                "position": "relative",
                                "width": "100%",
                                "max-width": "600px",
                                "height": "40px",
                                "margin-top": "5px",
                            }),
                            html.Button("Reboot", id="reboot-button", style={
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
                                "font-weight": "bold"
                            }),
                        ], style={
                            "display": "flex",
                            "flex-direction": "column",
                            "align-items": "center",
                            "justify-content": "center",
                            "padding": "0px 10px 0 10px", 
                        })
                    ], style={"flex": "2"}),
                ], style={"display": "flex", "flex-direction": "row"}),
                html.Br(),
            ]),

            dcc.Interval(
                id="update-interval",
                interval=1000,
                n_intervals=0
            ),

        ], style={
            "background-color": "#161616",
            "color": "#FFF",
            "font-family": "'Inter', sans-serif",
            "display": "flex",
            "flex-direction": "column",
            "flex-wrap": "wrap",
            "justify-content": "space-between",
            "align-items": "stretch",
            "height": "100vh",
            "width": "100%",
            "padding": "0",
            "margin": "0",
        })

        self.app.callback(
            output=Output("metrics-display", "children"),
            inputs=[Input("update-interval", "n_intervals")]
        )(self.update_metrics)

        self.app.callback(
            Output('detections-container', 'children'),
            [Input('update-interval', 'n_intervals')]
        )(self.update_detections)

        self.server.add_url_rule('/video_feed', 'video_feed', self.video_feed)

        self.start_server_thread()

    def start_server(self):
        self.app.run_server(port=5801, debug=False, use_reloader=False)

    def start_server_thread(self):
        Thread(target=self.start_server, daemon=True).start()

    def video_feed(self):
        return Response(self.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def generate_frames(self):
        while True:
            time.sleep(0.05)
            frame = self.vision_main.get_frame()
            if frame is None:
                continue

            frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def get_metrics(self):
        return {
            "framerate": self.vision_main.get_framerate(),
            "processing_latency": self.vision_main.get_processing_latency() * 1e3
        }

    def update_metrics(self, n_intervals):
        metrics = self.get_metrics()
        return [
            html.Div([
                html.Span(f"FPS: {metrics['framerate']:.2f}", style={
                    "position": "absolute", 
                    "left": "0", 
                    "bottom": "0",
                    "color": "rgba(255, 255, 255, 0.8)",
                    "font-size": "24px",
                    "font-weight": "regular",
                    "padding": "5px 10px",
                }),
                html.Span(f"Latency: {metrics['processing_latency']:.2f} ms", style={
                    "position": "absolute", 
                    "right": "0", 
                    "bottom": "0",
                    "color": "rgba(255, 255, 255, 0.8)",
                    "font-size": "24px",
                    "font-weight": "regular",
                    "padding": "5px 10px",
                    "border-radius": "5px",
                })
            ], style={
                "position": "relative",
                "width": "100%",
                "height": "40px",
            })
        ]
    
    def update_detections(self, n_intervals):
        detections = self.vision_main.get_detections()
        if not detections:
            return [
                html.Div("Nothing Detected", style={
                    'color': '#CCC9CA',
                    'font-size': '20px',
                    'text-align': 'center',
                    'border-radius': '24px',
                    'border': '2px solid rgba(255, 255, 255, 0.5)',
                    'padding': '15px',
                    'margin': '0 0px 20px 20px',
                    'display': 'flex',
                    'align-items': 'center',
                    'gap': '10px',
                    'width': '100%',
                })
            ]
        detection_items = []
        for detection in detections:
            detection_items.append(
                html.Div([
                    html.Span(f"• Distance: {detection.r:.2f} m", style={
                        'color': '#CCC9CA',
                        'margin-right': '10px'
                    }),
                    html.Span(f"• Angle: {detection.theta:.2f}°", style={
                        'color': '#CCC9CA',
                        'margin-right': '10px'
                    })
                ], style={
                    'border': '2px solid rgba(255, 255, 255, 0.5)',
                    'border-radius': '24px',
                    'padding': '15px',
                    'font-size': '20px',
                    'margin': '0 0px 20px 20px',
                    'display': 'flex',
                    'align-items': 'center',
                    'width': '100%',
                    'gap': '10px'
                })
            )
        return detection_items



    def index_string(self):
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>MonkeySee</title>
            <style>
                body {
                    background-color: #161616;
                    color: white;
                    font-family: 'Inter', sans-serif;
                    font-size: 15px;
                    margin: 0;
                    padding: 0;
                }
                h1 {
                    color: #F0C808;
                    text-align: center;
                    font-size: 48px;
                    font-weight: bold;
                }
                .video-container {
                    display: flex;
                    justify-content: center;
                    margin-bottom: 20px;
                }
                .funky-slider .rc-slider-track {
                    background-color: #CDA646;
                }
                .funky-slider .rc-slider-rail {
                    background-color: rgba(255, 255, 255, 0.5);
                }
                .funky-slider .rc-slider-handle {
                    border-color: #CDA646;
                    background-color: #CDA646;
                }
                .funky-slider .rc-slider-tooltip {
                    font-size: 14px;
                    color: rgba(255, 255, 255, 0.5);
                    background-color: #CDA646;
                    border-radius: 8px;
                    box-shadow: none;
                }
                .video-feed {
                    width: 100%;
                    max-width: 650px;
                    border: 5px solid #F0C808;
                    border-radius: 10px;
                }
                .container {
                    background-color: #161616;
                    height: 100vh;
                }
                .metrics {
                    color: #F0C808;
                    font-size: 20px;
                    font-weight: bold;
                    position: absolute;
                    top: 20px;
                    right: 20px;
                    background-color: rgba(0, 0, 0, 0.5);
                    padding: 10px;
                    border-radius: 5px;
                }
                .footer {
                    position: absolute;
                    bottom: 20px;
                    width: 100%;
                    text-align: center;
                    font-size: 14px;
                }
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
