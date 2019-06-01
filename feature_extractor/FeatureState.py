GOOD = 0  # value is perfectly A-okay
INSUF_HIST = 1  # value best-guess was made due to insufficient history (ex, acceleration set to zero due to one timestamp)
MISSING = 2  # value is missing (no car in front, etc.)
CENSORED_HI = 3  # value is past an operating threshold
CENSORED_LO = 4  # value is below an operating threshold
