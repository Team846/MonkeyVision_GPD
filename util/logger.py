from time import time

zero_time = time()

class Logger:
    LOGGING_ENABLED = True

    def __init__(self, name):
        self.name = name

    @staticmethod
    def get_time() -> float:
        global zero_time
        return time() - zero_time
    @staticmethod
    def format_time(num_dp: int = 1) -> str:
        return f"{round(Logger.get_time(), num_dp)}s"

    
    def Log(self, message: str) -> None:
        if not self.LOGGING_ENABLED: return
        print(f"LOG {Logger.get_time()} [{self.name}] {message}")
    def Warn(self, message: str) -> None:
        if not self.LOGGING_ENABLED: return
        print(f"WARN {Logger.get_time()} [{self.name}] {message}")
    def Error(self, message: str) -> None:
        if not self.LOGGING_ENABLED: return
        print(f"ERROR {Logger.get_time()} [{self.name}] {message}")