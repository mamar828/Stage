from asyncio import run as asyncio_run
from telegram_send import send as telegram_send
from time import time


def telegram_send_message(message: str):
    """
    Sends a notification message via Telegram. This function is called by the notify function.
    Note: messages can also be sent directly with a terminal command at the end of the execution 
    {cmd} ; telegram-send "{message}"

    Parameters
    ----------
    message : str
        The message to be sent.
    """
    try:
        asyncio_run(telegram_send(messages=[message]))
    except:
        print("No telegram bot configuration was available.")

def notify(func):
    """
    Decorates a function to notify when it has finished running.
    """
    def inner_func(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        telegram_send_message(f"{func.__name__} has finished running in {round(time()-start_time)}s.")
        return result
    
    return inner_func
