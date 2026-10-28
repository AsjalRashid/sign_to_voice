import cv2 
import numpy as np 
import time 
from dataclasses import dataclass
import threading
from typing import Optional

import time
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Frame:
    """
    An immutable data container representing a single captured video frame.

    Attributes:
        id (int): A monotonically increasing sequential identifier assigned to 
            the frame upon capture.
        timestamp (float): The high-resolution system monotonic time (via 
            `time.perf_counter()`) recorded at the exact moment the frame was decoded.
        image (np.ndarray): The raw multi-dimensional NumPy array containing the 
            pixel data (typically BGR format from OpenCV).
    """
    id: int                 
    timestamp: float        
    image: np.ndarray       
 
    @property
    def age_ms(self) -> float:
        """
        Calculates the age of this specific frame relative to the current system time.

        Useful for identifying data pipeline latency, bottleneck delays, or 
        backlogs in consumer loops.

        Returns:
            float: The elapsed time in milliseconds since the frame was captured.
        """
        return (time.perf_counter() - self.timestamp) * 1000.0
    
class CaptureError(RuntimeError):
    """
    Raised when an unrecoverable failure occurs during threaded video stream capture.

    This exception is thrown during pipeline initialization faults (e.g., hardware 
    disconnects, invalid streaming URLs) or when a live stream violates the 
    maximum read failure threshold.
    """
    pass


class ThreadedCapture():
    """
    A thread-safe, high-performance video wrapper for OpenCV capture streams.

    This class decouples frame decoding from your main processing loop by running 
    a background worker thread. It minimizes latency by restricting the driver-side 
    buffer size, monitors framing errors, calculates exponential moving average 
    (EMA) frame rates, and exposes a block-and-wait API for consumer loops.

    Attributes:
        source (Union[int, str]): The identifier for the video stream or device.
        width (int): Target frame width configuration.
        height (int): Target frame height configuration.
        target_fps (int): Intended frame rate for the hardware stream capture.
        mirror (bool): Flips frames horizontally if enabled.
        max_read_failures (int): Allowed continuous read drop frames before shutdown.
        start_timeout (float): Max seconds to wait for the first hardware frame.
    """    
    def __init__(self,
                 source: int|str = 0,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30,
                 mirror: bool = True,
                 max_read_failures: int = 30,
                 start_timeout: float = 5.0):
        """ Initializes configuration parameters for threaded video stream capture.
        
        Arguments:
            source (int | str): Camera index or path to a video file / streaming URL. Defaults to 0.
            width (int): Frame width resolution. Defaults to 640.
            height (int): Frame height resolution. Defaults to 480.
            fps (int): Target frames per second. Defaults to 30.
            mirror (bool): Mirror flip frames horizontally if True. Defaults to True.
            max_read_failures (int): Allowed drop frames before closing. Defaults to 30.
            start_timeout (float): Seconds to wait for hardware initialization. Defaults to 5.0. 
        """
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = fps
        self.mirror = mirror
        self.max_read_failures = max_read_failures
        self.start_timeout = start_timeout        
        
        #Internal Threading States
        self._cap : Optional[cv2.VideoCapture] = None #Holds the OpenCV camera handle
        self._thread: Optional[threading.Thread] = None #Holds the capture thread to be used 
        self._stop = threading.Event() #An internal boolean flag that allows one thread to signal a state change
        self._cond = threading.Condition() #Lets the consumer sleep until a new frame arrives
        self._frame: Optional[Frame] = None #The latest frame
        self._count = 0 #A running counter for input frames
        self._fps = 0.0 #Smoothed capture rate
        self._ended = False #Flags wether the source has ended or not
        self._error: Optional[str] = None # A str that describes what error has ocurred
        self._is_file: bool = isinstance(source, str) #A flag that check if a file path (a string) is passed as source

    def read(self, after_id: int = 0, timeout: float = 1.0) -> Optional[Frame]:
        """
        Blocks until a frame newer than a specific ID is captured.

        Args:
            after_id: The ID threshold to check against. Only frames with an ID 
                strictly greater than this value will be returned.
            timeout: The maximum time to wait in seconds before giving up.

        Returns:
            Optional[Frame]: The newest `Frame` object if available before the 
            timeout; `None` if a timeout occurs and the end of the file is reached, 
            the camera connection is lost, or `stop()` is called.
        """
        def ready() -> bool:
            newer = self._frame is not None and self._frame.id > after_id
            return newer or self._ended or self._stop.is_set()

        with self._cond:
            self._cond.wait_for(ready, timeout=timeout)
            if self._frame is not None and self._frame.id > after_id:
                return self._frame
            return None
         
    def _loop (self) -> None:
        """
        The core background worker loop that continuously pulls frames from the video source.

        Runs inside a dedicated thread. It decodes frames, applies optional mirror 
        flipping, tracks the instantaneous FPS, computes an Exponential Moving 
        Average (EMA) for the overall stream FPS, and broadcasts updates to waiting 
        threads. It also handles mock real-time playback pacing for file sources 
        and tracks sequential frame delivery failures for hardware sources.
        """        
        failure = 0 
        file_delay = 0.0
        
        if self._is_file:
            native = self._cap.get(cv2.CAP_PROP_FPS) or self.target_fps # A fallback added incase reading the metadata fails
            file_delay = 1.0 / native
            
        last_timestamp = time.perf_counter()
        
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            
            if not ret: 
                
                if self._is_file: 
                    # Normal Ending of a file
                    return self._finish(None)
                
                failure +=1
                if failure >= self.max_read_failures:
                    return self._finish("Camera stopped delivering frames")
                
                time.sleep(0.01)
                continue
            failure = 0
            
            if self.mirror:
                frame = cv2.flip(frame, 1)
                
            now = time.perf_counter()
            inst_fps = 1.0/max(now - last_timestamp, 1e-6)
            last_timestamp = now 
            
            with self._cond:
                self._count +=1
                self._frame = Frame(self._count, now, frame)
                self._fps = inst_fps if self._fps == 0 else 0.9 * self._fps + 0.1 * inst_fps # Calculates Exponential Moving Average
                self._cond.notify_all()
            
            if file_delay:
                # Forces the loop to wait before fetching the next frame so the video plays back at it's normal speed
                time.sleep(file_delay)

    def _finish(self, error: Optional[str]) -> None:
        """
        Cleans up the internal states when the capture loop terminates.

        Sets the completion flags, saves the exit status error message ,if any and signals 
        all threads blocked on the condition variable that processing has halted.

        Args:
            error: A descriptive error string if the loop crashed or timed out, 
                or `None` if the stream naturally reached the end of a file.
        """        
        with self._cond:
            self._ended = True
            self._error = error
            self._cond.notify_all()
            
    def start(self) -> 'ThreadedCapture':
        """
        Starts the background thread to capture video frames.

        Initializes the OpenCV video capture, configures stream properties 
        (if the source is a live camera), spins up a daemon thread to consume 
        frames, and blocks briefly until the first frame arrives or the 
        timeout expires.

        Returns:
            ThreadedCamera: The current instance, allowing for method chaining.

        Raises:
            CameraError: If the video source cannot be opened, or if no frames 
                are received within the designated startup timeout window.
        """
        
        if self._thread is not None: 
            return self
        
        self._cap = cv2.VideoCapture(self.source)
        
        if not self._cap.isOpened():
            self._cap.release()
            raise CaptureError(f"Cannot open video source {self.source}.")
        
        if not self._is_file : 
            #Configurations for real time video stream
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Keeps the driver queue short
            
        
        self._stop.clear() # Signals that the the below code is allowed to run
        self._thread = threading.Thread(target=self._loop, name='camera-capture', daemon=True)
        self._thread.start()     
    
        if self.read(after_id=0, timeout=self.start_timeout) is None:
            self.stop()
            raise CaptureError(self._error or "Camera produced no frames.")
        
        return self
    
    def stop(self) -> None:
        """
        Stops the frame capture loop and releases all system resources.

        Signals the worker thread to exit, notifies waiting threads via the 
        condition variable, joins the background thread with a timeout, 
        and safely releases the OpenCV hardware handle.
        """

        self._stop.set() #Gives the signal to stop the background thread
        
        with self._cond:
            self._cond.notify_all()
        
        if self._thread is not None: 
            self._thread.join(timeout=2.0)
            self._thread = None
        
        if self._cap is not None:
            self._cap.release()
            self._cap = None
                
    def __enter__(self) -> 'ThreadedCapture':
        """Enters the runtime context window for safe camera management.
        
        Returns: 
            ThreadedCapture: The threaded, initialized capture instance.
        """
        
        return self.start()
    
    def __exit__(self, *exc) -> None:
        """
        Exits the runtime context window, ensuring resources are safely closed.

        Args:
            *exc: Variable length argument list capturing exception types, 
                values, and tracebacks if an error occurred inside the block.
        """
        return self.stop()

    @property
    def fps(self) -> float:
        """
        The smoothed frame capture rate.

        Returns:
            float: The Exponential Moving Average (EMA) of the frames per second 
            delivered by the video source. Returns 0.0 if no frames have been processed.
        """
        return self._fps
 
    @property
    def frames_captured(self) -> int:
        """
        The total number of frames captured since initialization.

        Returns:
            int: A rolling counter incremented each time a frame is successfully 
            read and processed from the stream.
        """
        return self._count
 
    @property
    def ended(self) -> bool:
        """
        Indicates whether the video source stream has completed.

        Returns:
            bool: True if the end of a video file was reached, a critical hardware 
            read failure occurred, or the thread was safely torn down. False otherwise.
        """
        return self._ended
 
    @property
    def error(self) -> Optional[str]:
        """
        The exception description or failure message encountered during streaming.

        Returns:
            Optional[str]: A descriptive string detailing why the camera capture loop 
            terminated early (e.g., "Camera stopped delivering frames"). Returns `None` 
            if the stream ended gracefully or is actively running without issue.
        """
        return self._error

