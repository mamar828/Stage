def format_time(total_seconds: float, precision: int=2) -> str:
    """
    Format a given time in seconds into a human-readable string. An example of the output format is "13h05m06.23s".

    Parameters
    ----------
    total_seconds : float
        The total time in seconds to be formatted.
    precision : int, default=2
        The number of decimal places for the seconds.

    Returns
    -------
    str
        A human-readable string representing the formatted time.
    """
    end_str = ""
    # Explicitly convert the numbers to int as python's "integer division" does not always output integer results...
    hours, minutes, seconds = int(total_seconds//3600), int((total_seconds%3600)//60), (total_seconds%3600)%60
    if hours:
        end_str += f"{hours}h"
    if minutes:
        if hours:
            # Force the minutes format to have two digits
            end_str += f"{minutes:02d}m"
        else:
            end_str += f"{minutes}m"
        # Force the seconds format to have two digits
        end_str += f"{seconds:05.{precision}f}s"
    else:
        end_str += f"{seconds:.{precision}f}s"
    return end_str
