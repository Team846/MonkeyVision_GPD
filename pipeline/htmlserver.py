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
                    html.H1("MonkeySee", style={
                        'textAlign': 'left', 
                        'color': '#F0C808', 
                        'font-size': '48px', 
                        'font-weight': 'bold', 
                        'padding-top': '20px',
                        'padding-left': '40px',
                        'margin-bottom': '0px',
                        'padding-bottom': '0px'
                    }),
                    html.H4("Developed by FRC Team 846", style={
                        'textAlign': 'left', 
                        'color': '#F0C808', 
                        'font-size': '14px',
                        'padding-top': '20px',
                        'padding-left': '75px',
                        'margin-top': '0px',
                        'padding-top': '3px'
                    })
                ]),
                html.Div(
                        html.Img(src="/video_feed", style={
                        "width": "100%", 
                        "max-width": "1200px", 
                        "border": "5px solid #F0C808", 
                        "border-radius": "10px"
                    }),
                    
                    style={
                        "display": "flex", 
                        "justify-content": "center", 
                        "margin-bottom": "20px"
                    }
                ),
                html.Br(),
            ]),

            html.Div(id="metrics-display", style={
                "position": "absolute", 
                "top": "20px", 
                "right": "20px", 
                "font-size": "20px", 
                "color": "#F0C808", 
                "font-weight": "bold", 
                "background-color": "rgba(0, 0, 0, 0.5)", 
                "padding": "10px", 
                "border-radius": "5px",
            }),

            dcc.Interval(
                id="update-interval",
                interval=1000,
                n_intervals=0
            ),

        ], style={
            "background-color": "#161616", 
            "height": "100vh", 
            "color": "#FFF", 
            "font-family": "'Verdana', sans-serif", 
            "position": "relative"
        })

        self.app.callback(
            output=Output("metrics-display", "children"),
            inputs=[Input("update-interval", "n_intervals")]
        )(self.update_metrics)


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
            f"{metrics['framerate']:.2f} FPS. {metrics['processing_latency']:.2f} ms processing latency.",
            html.Br(),
        ]

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
                    font-family: 'Verdana', sans-serif;
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
                .video-feed {
                    width: 100%;
                    max-width: 1200px;
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
