# Hyper parameters
EAR_THRESH = 0.3     # EAR max value indicating blink
EAR_CONSEC_FRAMES = 3 # Number of consecutive frames the eye must be below the threshold

FRAME_MAX_SIDE = 800 # The max size of the longest sides of a frame
EAR_WINDOW = {
    'blink': 13, # Window for storing last EAR values
    'open': 30  # Window for storing last EAR values of opened eyes
}

# Plot parameters
EAR_PLOT_PARAMS = {
    'height' : 200,
    'width' : 500
}

# Color scheme definition
COLORS = {
    'ok': (100,255,0),       # Green
    'blink':(0,165,255),     # Orange
    'alert':(0,0,255),        # Red
    'text':(3, 169, 252)
}