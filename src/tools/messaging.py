from asyncio import run as asyncio_run
from time import time
from datetime import timedelta
from tqdm import tqdm
from tqdm.contrib.telegram import tqdm as tqdm_telegram
try:
    import telegram_send as _telegram_send
except Exception:
    _telegram_send = None


def telegram_send_message(message: str):
    """
    Sends a notification message via Telegram. This function is called by the notify function.

    ..note::
        Messages can also be sent directly with a terminal command at the end of the execution
        e.g. : {cmd} ; telegram-send "{message}"

    Parameters
    ----------
    message : str
        The message to be sent.
    """
    try:
        asyncio_run(_telegram_send.send(messages=[message]))
    except:
        print("No telegram bot configuration was available.")

def notify_function_end(func):
    """
    Decorates a function to notify when it has finished running.
    """
    def inner_func(*args, **kwargs):
        start_time = time()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            telegram_send_message(f"{func.__name__} has failed after {format_time(time()-start_time)} with error: {e}.")
            raise e
        telegram_send_message(f"{func.__name__} has finished running in {format_time(time()-start_time)}.")
        return result

    return inner_func

def format_time(seconds: float) -> str:
    """
    Formats a time duration in seconds into a human-readable string.

    Parameters
    ----------
    seconds : float
        The time duration in seconds.

    Returns
    -------
    str
        The time duration formatted as a timedelta string, stripped of trailing zeros.
    """
    if seconds > 0.01:
        seconds = round(seconds, 2)
    return str(timedelta(seconds=seconds)).rstrip("0")

def get_telegram_config():
    try:
        return _telegram_send.telegram_send.get_config_settings()
    except Exception:
        return False

def smart_tqdm(iterable=None, desc=None, total=None, leave=True, file=None,
               ncols=None, mininterval=0.1, maxinterval=10.0, miniters=None,
               ascii=None, disable=False, unit='it', unit_scale=False,
               dynamic_ncols=False, smoothing=0.3, bar_format=None, initial=0,
               position=None, postfix=None, unit_divisor=1000, write_bytes=False,
               lock_args=None, nrows=None, colour=None, delay=0.0, gui=False,
               **kwargs) -> tqdm:
    """
    A smart wrapper around tqdm that checks for a telegram_send configuration and uses it if available. This allows to
    display progress bars in Telegram messages.
    For a full list of arguments, please refer to the tqdm documentation at https://tqdm.github.io/docs/tqdm/.
    """
    kwargs.update({
        "iterable" : iterable, "desc" : desc, "total" : total, "leave" : leave, "file" : file,
        "ncols" : ncols, "mininterval" : mininterval, "maxinterval" : maxinterval, "miniters" : miniters,
        "ascii" : ascii, "disable" : disable, "unit" : unit, "unit_scale" : unit_scale,
        "dynamic_ncols" : dynamic_ncols, "smoothing" : smoothing, "bar_format" : bar_format, "initial" : initial,
        "position" : position, "postfix" : postfix, "unit_divisor" : unit_divisor, "write_bytes" : write_bytes,
        "lock_args" : lock_args, "nrows" : nrows, "colour" : colour, "delay" : delay, "gui" : gui,
    })
    config = get_telegram_config()
    if config:
        kwargs["colour"] = None  # Telegram does not support colors
        return tqdm_telegram(**kwargs, token=config.token, chat_id=config.chat_id)
    else:
        return tqdm(**kwargs)
