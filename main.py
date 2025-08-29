import pipeline.htmlserver
import pipeline.ntables
from pipeline.visionmain import VisionMain
import argparse
from time import sleep
import asyncio

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=int, required=True)
    args = parser.parse_args()

    vision_main = VisionMain(args.pipeline)
    server = pipeline.htmlserver.HTMLServer(vision_main)

    async def main_loop():
        while True:
            await vision_main.execute()

    asyncio.run(main_loop())
