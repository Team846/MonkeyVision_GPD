import pipeline.htmlserver
import pipeline.ntables
from pipeline.visionmain import VisionMain
import argparse
from time import sleep
import threading

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=int, required=True)
    args = parser.parse_args()

    vision_main = VisionMain(args.pipeline)
    server = pipeline.htmlserver.HTMLServer(vision_main)

    def main_loop():
        while True:
            vision_main.execute()
            sleep(0.001)

    vision_thread = threading.Thread(target=main_loop, daemon=True)
    vision_thread.start()
    try:
        while True:
            sleep(1)
    except KeyboardInterrupt:
        exit(0)
