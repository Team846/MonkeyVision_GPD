import math
from dash import Dash, html, dcc, Input, Output, no_update
import cv2
from flask import Flask, Response
from pipeline.visionmain import VisionMain
import time
from threading import Thread
from util.config import ConfigCategory, Config
import os
from camera.preprocess import GET_DIVERGENCE_GAIN, SET_DIVERGENCE_GAIN, GET_TARGET_BRIGHTNESS, SET_TARGET_BRIGHTNESS, GET_NUM_BINS, SET_NUM_BINS, GET_MIN_CORR_STRENGTH, SET_MIN_CORR_STRENGTH
from localization.detection import GET_THRESH_STEP, SET_THRESH_STEP, GET_THRESH_WIN, SET_THRESH_WIN

class HTMLServer:
    config_category = ConfigCategory("HTMLServer")
    framecomp_slider = config_category.getFloatConfig("framecomp_slider", 0.5)

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
                    'padding': '0px 0px 0px 7px',
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
                html.Br(),
                html.H4("Settings", style={
                    'textAlign': 'left',
                    'color': '#CCC9CA',
                    'font-size': '18px',
                    'font-weight': 'medium',
                    'padding': '0px 0px 0px 7px',
                }),
                html.Label("Dynamic Frame Correction Target", style={
                    "color": "#CCC9CA",
                    "font-size": "16px",
                    "margin-bottom": "10px",
                    'padding': '15px 20px 0 15px',
                    "white-space": "nowrap",
                    "max-width": "250px",
                }),
                dcc.Slider(
                    id="dfc-target-slider",
                    min=50,
                    max=200,
                    step=1,
                    value=GET_TARGET_BRIGHTNESS(),
                    marks={50: '50', 200: '200'},
                    tooltip={"placement": "bottom", "always_visible": True},
                    className="funky-slider"
                ),
                html.Br(),
                html.Label("ASLC Num Bins", style={
                    "color": "#CCC9CA",
                    "font-size": "16px",
                    "margin-bottom": "10px",
                    'padding': '15px 20px 0 15px',
                    "white-space": "nowrap",
                    "max-width": "250px",
                }),
                dcc.Slider(
                    id="aslc-num-bins-slider",
                    min=100,
                    max=1600,
                    step=100,
                    value=GET_NUM_BINS(),
                    marks={50: '50', 1600: '1600'},
                    tooltip={"placement": "bottom", "always_visible": True},
                    className="funky-slider"
                ),
                html.Br(),
                html.Label("Min ASLC Correction", style={
                    "color": "#CCC9CA",
                    "font-size": "16px",
                    "margin-bottom": "10px",
                    'padding': '15px 20px 0 15px',
                    "white-space": "nowrap",
                    "max-width": "250px",
                }),
                dcc.Slider(
                    id="aslc-min-corr-slider",
                    min=0.02,
                    max=0.5,
                    step=0.01,
                    value=GET_MIN_CORR_STRENGTH(),
                    marks={0.02: '0.02', 0.5: '0.5'},
                    tooltip={"placement": "bottom", "always_visible": True},
                    className="funky-slider"
                ),
                html.Br(),
                html.Label("Divergence Gain", style={
                    "color": "#CCC9CA",
                    "font-size": "16px",
                    "margin-bottom": "10px",
                    'padding': '15px 20px 0 15px',
                    "white-space": "nowrap",
                    "max-width": "250px",
                }),
                dcc.Slider(
                    id="divergence-gain-slider",
                    min=0.5,
                    max=4.0,
                    step=0.1,
                    value=GET_DIVERGENCE_GAIN(),
                    marks={0.5: '0.5', 4.0: '4.0'},
                    tooltip={"placement": "bottom", "always_visible": True},
                    className="funky-slider"
                ),
                html.Br(),
                html.Label("Thresholding Steps", style={
                    "color": "#CCC9CA",
                    "font-size": "16px",
                    "margin-bottom": "10px",
                    'padding': '15px 20px 0 15px',
                    "white-space": "nowrap",
                    "max-width": "250px",
                }),
                dcc.Slider(
                    id="thresh-step-slider",
                    min=11,
                    max=33,
                    step=2,
                    value=GET_THRESH_STEP(),
                    marks={11: '11', 33: '33'},
                    tooltip={"placement": "bottom", "always_visible": True},
                    className="funky-slider"
                ),
                html.Br(),
                html.Label("Maximum threshold window", style={
                    "color": "#CCC9CA",
                    "font-size": "16px",
                    "margin-bottom": "10px",
                    'padding': '15px 20px 0 15px',
                    "white-space": "nowrap",
                    "max-width": "250px",
                }),
                dcc.Slider(
                    id="max-thresh-slider",
                    min=11,
                    max=33,
                    step=2,
                    value=GET_THRESH_WIN(),
                    marks={11: '11', 33: '33'},
                    tooltip={"placement": "bottom", "always_visible": True},
                    className="funky-slider"
                ),
                html.Br(),
                html.Label("Displayed Frame Quality", style={
                    "color": "#CCC9CA",
                    "font-size": "16px",
                    "margin-bottom": "10px",
                    'padding': '15px 20px 0 15px',
                    "white-space": "nowrap",
                    "max-width": "250px",
                }),
                dcc.Slider(
                    id="framecomp-slider",
                    min=0.05,
                    max=1,
                    step=0.001,
                    value=HTMLServer.framecomp_slider.valueFloat(),
                    marks={0.05: '5%', 1: '100%'},
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
                    html.Label(f"Pipeline #{self.vision_main.get_pipeline_number()}", style={
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
                    html.Br(),
                    html.Div([
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
                        "justify-content": "center",
                        "align-items": "center",
                        "padding": "10px",
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
            html.Div(id='fake-output', style={'display': 'none'}),
            html.Div(id='fake-output-2', style={'display': 'none'}),
            html.Div(id='fake-output-3', style={'display': 'none'}),
            html.Div(id='fake-output-4', style={'display': 'none'}),
            html.Div(id='fake-output-5', style={'display': 'none'}),
            html.Div(id='fake-output-6', style={'display': 'none'}),
            html.Div(id='fake-output-7', style={'display': 'none'}),
            html.Div(id='fake-output-8', style={'display': 'none'}),
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
            output=Output('fake-output-2', 'children'),
            inputs=[Input('reboot-button', 'n_clicks')]
        )(self.reboot_system)

        self.app.callback(
            output=Output("metrics-display", "children"),
            inputs=[Input("update-interval", "n_intervals")]
        )(self.update_metrics)

        self.app.callback(
            Output('detections-container', 'children'),
            [Input('update-interval', 'n_intervals')]
        )(self.update_detections)

        self.app.callback(
            Output("fake-output", "children"),
            [Input("framecomp-slider", "value")]
        )(self.framecomp_callback)

        self.app.callback(
            Output("fake-output-3", "children"),
            [Input("dfc-target-slider", "value")]
        )(self.dfc_target_callback)

        self.app.callback(
            Output("fake-output-4", "children"),
            [Input("aslc-num-bins-slider", "value")]
        )(self.aslc_num_bins_callback)

        self.app.callback(
            Output("fake-output-5", "children"),
            [Input("aslc-min-corr-slider", "value")]
        )(self.aslc_min_corr_callback)

        self.app.callback(
            Output("fake-output-6", "children"),
            [Input("divergence-gain-slider", "value")]
        )(self.divergence_gain_callback)

        self.app.callback(
            Output("fake-output-7", "children"),
            [Input("thresh-step-slider", "value")]
        )(self.thresh_step_callback)

        self.app.callback(
            Output("fake-output-8", "children"),
            [Input("max-thresh-slider", "value")]
        )(self.max_thresh_callback)

        self.server.add_url_rule('/video_feed', 'video_feed', self.video_feed)

        self.start_server_thread()

    def reboot_system(self, n_clicks):
        if n_clicks:
            os.system('sudo reboot')
        return f'Rebooting system'

    def start_server(self):
        self.app.run_server(host="0.0.0.0", port=5801, debug=True, use_reloader=False)

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
            
            rs_dim = HTMLServer.framecomp_slider.valueFloat()
            frame = cv2.resize(frame, (0, 0), fx=rs_dim, fy=rs_dim, interpolation=cv2.INTER_AREA)

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
                html.Span(f"Framerate: {metrics['framerate']:.2f} fps", style={
                    "position": "absolute", 
                    "left": "0", 
                    "bottom": "0",
                    "color": "rgba(255, 255, 255, 0.8)",
                    "font-size": "18px",
                    "font-weight": "regular",
                    "font-style": "italic",
                    "padding": "5px 10px",
                }),
                html.Span(f"Latency: {metrics['processing_latency']:.2f} ms", style={
                    "position": "absolute", 
                    "right": "0", 
                    "bottom": "0",
                    "color": "rgba(255, 255, 255, 0.8)",
                    "font-size": "18px",
                    "font-weight": "regular",
                    "padding": "5px 10px",
                    "font-style": "italic",
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
                html.Div("No detections", style={
                    'color': '#CCC9CA',
                    'font-size': '14px',
                    'text-align': 'center',
                    'border-radius': '10px',
                    'border': '2px solid rgba(255, 255, 255, 0.5)',
                    'padding': '10px',
                    'margin': '0 0px 20px 20px',
                    'display': 'flex',
                    'align-items': 'center',
                    'gap': '10xpx',
                    'width': '100%',
                })
            ]
        detection_items = []
        for i in range(len(detections)):
            detection = detections[i]
            detection_items.append(
                html.Div([
                    html.Span(f"Detection #{i + 1}:", style={
                        'color': '#CCC9CA',
                        'margin-right': '10px',
                        'font-weight': 'medium'
                    }),
                    html.Span(f"R {detection.r:.1f}in", style={
                        'color': '#CCC9CA',
                        'margin-right': '5px'
                    }),
                    html.Span(f"θ {detection.theta:.2f}deg", style={
                        'color': '#CCC9CA',
                        'margin-right': '10px'
                    })
                ], style={
                    'border': '2px solid rgba(255, 255, 255, 0.5)',
                    'border-radius': '10px',
                    'padding': '10px',
                    'font-size': '14px',
                    'margin': '0 0px 20px 20px',
                    'display': 'flex',
                    'align-items': 'center',
                    'width': '100%',
                    'gap': '10px'
                })
            )
        return detection_items

    def framecomp_callback(self, value):
        print("Frame compression value updated")
        HTMLServer.framecomp_slider.setFloat(value)
        return f'Slider value is {value}'
    
    def dfc_target_callback(self, value):
        print("Target brightness value updated")
        SET_TARGET_BRIGHTNESS(value)
        return f'Slider value is {value}'
    
    def aslc_num_bins_callback(self, value):
        print("ASLC num bins value updated")
        SET_NUM_BINS(value)
        return f'Slider value is {value}'
    
    def aslc_min_corr_callback(self, value):
        print("ASLC min correction value updated")
        SET_MIN_CORR_STRENGTH(value)
        return f'Slider value is {value}'
    
    def divergence_gain_callback(self, value):
        print("Divergence gain value updated")
        SET_DIVERGENCE_GAIN(value)
        return f'Slider value is {value}'
    
    def thresh_step_callback(self, value):
        print("Thresh step value updated")
        SET_THRESH_STEP(value)
        return f'Slider value is {value}'
    
    def max_thresh_callback(self, value):
        print("Max thresh win value updated")
        SET_THRESH_WIN(value)
        return f'Slider value is {value}'

    def index_string(self):
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>MonkeyVision</title>
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
