# import pipeline.htmlserver
from pipeline.visionmain import VisionMain

if __name__ == "__main__":
    vision_main = VisionMain()
    # server = pipeline.htmlserver.HTMLServer(vision_main)
    vision_main.execute()