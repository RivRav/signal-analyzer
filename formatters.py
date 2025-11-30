def format_size(size_bytes):
    """
        - format a size into a readable string

        @Parameters
        size_bytes : int or float
            - the size value in bytes

        @Returns
        str
            - the size given as text
    """

    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / (1024**2):.1f} MB"
    else:
        return f"{size_bytes / (1024**3):.1f} GB"


def format_time(seconds):
    """
        - format a duration into a readable string

        @Parameters
        seconds : float
            - time in seconds

        @Returns
        str
            - a formatted time string
    """
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)} min {s:.1f} s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)} h {int(m)} min"
