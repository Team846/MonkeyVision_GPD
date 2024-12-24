import pipeline.htmlserver
import pipeline.ntables
from pipeline.visionmain import VisionMain
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=int, required=True)
    args = parser.parse_args()
                        
    vision_main = VisionMain(args.pipeline)
    server = pipeline.htmlserver.HTMLServer(vision_main)
    vision_main.execute()